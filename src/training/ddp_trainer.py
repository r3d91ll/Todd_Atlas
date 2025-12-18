"""
Distributed Data Parallel (DDP) trainer for Atlas.

Efficient multi-GPU training using PyTorch DDP.
Reference: TNT paper, PyTorch DDP best practices

Usage:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/atlas_50m.yaml
"""

import os
import math
import signal
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List
from pathlib import Path
from tqdm import tqdm
import time
import json
from datetime import datetime

from ..model.atlas import Atlas, AtlasConfig
from .metrics import MetricsLogger
from .alerts import TrainingAlerts, create_alerts_from_config


def setup_ddp(rank: int, world_size: int, backend: str = "nccl"):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up DDP process group."""
    dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


class DDPTrainer:
    """
    Distributed Data Parallel trainer for Atlas.

    Features:
    - Multi-GPU training with proper gradient synchronization
    - Mixed precision (bf16) for memory efficiency
    - TNT two-stage training support
    - Gradient accumulation for large effective batch sizes
    - COMPREHENSIVE LOGGING for debugging
    """

    def __init__(
        self,
        model: Atlas,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Optimization
        lr: float = 4e-4,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        total_steps: int = 50000,
        grad_accum_steps: int = 8,
        grad_clip: float = 1.0,
        # DDP
        rank: int = 0,
        world_size: int = 1,
        device: torch.device = None,
        # Mixed precision
        use_amp: bool = True,
        # TNT
        use_tnt: bool = True,
        stage1_chunk_size: int = 2048,
        stage1_steps: int = 45000,
        stage2_chunk_size: int = 256,
        stage2_steps: int = 5000,
        # Logging
        output_dir: Path = Path("runs/atlas"),
        log_every: int = 10,
        val_every: int = 500,
        save_every: int = 5000,
        # Early stopping
        val_patience: int = 10,
        # Memory reset
        memory_reset_every: int = 5000,
    ):
        self.rank = rank
        self.world_size = world_size
        self.device = device or torch.device(f"cuda:{rank}")
        self.is_main = (rank == 0)

        # Move model to device and wrap with DDP
        self.model = model.to(self.device)
        if world_size > 1:
            self.model = DDP(self.model, device_ids=[rank])

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimization params
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip = grad_clip

        # TNT params
        self.use_tnt = use_tnt
        self.stage1_chunk_size = stage1_chunk_size
        self.stage1_steps = stage1_steps
        self.stage2_chunk_size = stage2_chunk_size
        self.stage2_steps = stage2_steps
        if use_tnt:
            self.total_steps = stage1_steps + stage2_steps

        # Logging params
        self.output_dir = Path(output_dir)
        self.log_every = log_every
        self.val_every = val_every
        self.save_every = save_every

        # Only main process does logging/saving
        if self.is_main:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger = MetricsLogger(self.output_dir / "metrics")
            # Separate detailed log file for training steps
            self.train_log_file = self.output_dir / "metrics" / "train_steps.jsonl"
            self.component_log_file = self.output_dir / "metrics" / "component_stats.jsonl"
        else:
            self.logger = None
            self.train_log_file = None
            self.component_log_file = None

        # Optimizer with separate parameter groups for different components
        # This addresses the gradient scale mismatch between components
        param_groups = self._create_param_groups(lr, weight_decay)
        self.optimizer = AdamW(param_groups, betas=betas)

        if self.is_main:
            print(f"\nParameter groups:")
            for i, pg in enumerate(param_groups):
                print(f"  Group {i} ({pg.get('name', 'unnamed')}): {pg['lr']:.2e} LR, {len(pg['params'])} params")

        # LR scheduler - applies to all groups proportionally
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - warmup_steps,
            eta_min=lr * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Mixed precision
        self.use_amp = use_amp
        self.scaler = GradScaler() if use_amp else None

        # State
        self.global_step = 0
        self.current_stage = 1
        self.memory_states = None

        # Alerting (will be configured separately)
        self.alerts: Optional[TrainingAlerts] = None

        # Timing for budgeting
        self.training_start_time: Optional[float] = None
        self.last_log_time: Optional[float] = None
        self.last_good_loss: float = float('inf')
        self.steps_since_last_progress: int = 0
        self.progress_report_interval: int = 500  # Report progress every N steps

        # Early stopping on validation loss
        self.best_val_loss: float = float('inf')
        self.val_patience: int = val_patience  # Stop if no improvement for N validations
        self.val_patience_counter: int = 0
        self.early_stopped: bool = False

        # Memory reset frequency (prevent memorization)
        self.memory_reset_every: int = memory_reset_every  # Reset memory states periodically
        self.steps_since_memory_reset: int = 0

        # Signal handling for graceful shutdown with checkpoint save
        self._interrupt_received: bool = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers to save checkpoint on interrupt."""
        if not self.is_main:
            return  # Only main process handles signals

        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            print(f"\n[SIGNAL] Received {sig_name} - will save checkpoint and exit gracefully")
            self._interrupt_received = True

        # Register handlers for common termination signals
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command

        if self.is_main:
            print("[SIGNAL] Signal handlers registered for graceful checkpoint save on interrupt")

    def set_alerts(self, alerts: TrainingAlerts):
        """Configure alerting system."""
        self.alerts = alerts
        if self.is_main:
            print(f"[ALERTS] SMS alerts enabled to {alerts.to_address}")

    def _create_param_groups(self, base_lr: float, weight_decay: float) -> list:
        """
        Create parameter groups with different learning rates.

        Based on diagnostic analysis:
        - Retention gates have ~1000x smaller gradients → need higher LR
        - Memory/attention gates have ~10x smaller gradients → need higher LR
        - Memory, attention, FFN have similar gradient scales → use base LR

        LR multipliers tuned based on gradient norm ratios.
        """
        model = self._get_model()

        # Categorize parameters
        retention_params = []
        gate_params = []
        memory_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Retention gate parameters (local_logit, global_logit)
            # These are the retention decay parameters that control memory forgetting
            if 'retention' in name and ('logit' in name):
                retention_params.append(param)
            # Memory/attention gating parameters
            # Also includes Titans input-dependent params (alpha_proj, eta_proj, theta_proj)
            elif ('gate' in name and 'retention' not in name) or \
                 ('memory' in name and '_proj' in name and any(p in name for p in ['alpha', 'eta', 'theta'])):
                gate_params.append(param)
            # Memory module parameters (projections for k, v, q, qk - but not alpha/eta/theta)
            elif 'memory' in name:
                memory_params.append(param)
            # Everything else (attention, FFN, embeddings, norms)
            else:
                other_params.append(param)

        # Build parameter groups with different LRs
        # Rationale for multipliers:
        # - Retention: gradients ~1000x smaller, but we don't want to destabilize
        #   Use 25x higher LR as a conservative start
        # - Gates: gradients ~10x smaller, use 5x higher LR
        # - Memory: similar to others, but no weight decay (it's a state matrix)
        param_groups = []

        if other_params:
            param_groups.append({
                "name": "main",
                "params": other_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
            })

        if memory_params:
            param_groups.append({
                "name": "memory",
                "params": memory_params,
                "lr": base_lr,
                "weight_decay": 0.0,  # No weight decay on memory projections
            })

        if retention_params:
            param_groups.append({
                "name": "retention",
                "params": retention_params,
                "lr": base_lr * 5.0,  # 5x higher LR for retention gates (reduced from 10x to prevent explosion)
                "weight_decay": 0.0,  # No weight decay on gates
            })

        if gate_params:
            param_groups.append({
                "name": "gate",
                "params": gate_params,
                "lr": base_lr * 5.0,  # 5x higher LR for memory/attn gates
                "weight_decay": 0.0,  # No weight decay on gates
            })

        return param_groups

    def _get_model(self) -> Atlas:
        """Get underlying model (unwrap DDP if needed)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norms for different components."""
        model = self._get_model()

        norms = {
            "total": 0.0,
            "memory": 0.0,
            "attention": 0.0,
            "ffn": 0.0,
            "embedding": 0.0,
            "retention": 0.0,
            "gate": 0.0,
        }

        counts = {k: 0 for k in norms}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                norms["total"] += grad_norm ** 2
                counts["total"] += 1

                # Categorize by component
                if "memory" in name:
                    norms["memory"] += grad_norm ** 2
                    counts["memory"] += 1
                elif "attention" in name:
                    norms["attention"] += grad_norm ** 2
                    counts["attention"] += 1
                elif "ffn" in name:
                    norms["ffn"] += grad_norm ** 2
                    counts["ffn"] += 1
                elif "emb" in name:
                    norms["embedding"] += grad_norm ** 2
                    counts["embedding"] += 1
                elif "retention" in name:
                    norms["retention"] += grad_norm ** 2
                    counts["retention"] += 1
                elif "gate" in name:
                    norms["gate"] += grad_norm ** 2
                    counts["gate"] += 1

        # Convert to actual norms
        for k in norms:
            norms[k] = norms[k] ** 0.5 if norms[k] > 0 else 0.0

        return norms

    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics from memory module."""
        model = self._get_model()
        stats = {}

        for i, block in enumerate(model.blocks):
            # Memory module stats
            mem = block.memory
            if hasattr(mem, 'lr') and mem.lr is not None:
                stats[f"layer_{i}/memory_lr"] = mem.lr.item()

            # Omega memory specific stats
            if hasattr(mem, 'alpha'):
                stats[f"layer_{i}/omega_alpha"] = mem.alpha.item()
            if hasattr(mem, 'theta'):
                stats[f"layer_{i}/omega_theta"] = mem.theta.item()
            if hasattr(mem, 'eta'):
                stats[f"layer_{i}/omega_eta"] = mem.eta.item()

            # Retention gate stats (only for original AtlasBlock, not AtlasOmegaBlock)
            if hasattr(block, 'retention'):
                ret = block.retention
                if hasattr(ret, 'lambda_local'):
                    stats[f"layer_{i}/lambda_local"] = ret.lambda_local.item()
                if hasattr(ret, 'lambda_global'):
                    stats[f"layer_{i}/lambda_global"] = ret.lambda_global.item()

            # Gate values (if available from last forward)
            if hasattr(block, 'gate'):
                gate = block.gate
                if hasattr(gate, 'last_gate_mean'):
                    stats[f"layer_{i}/gate_mean"] = gate.last_gate_mean

            # Gamma gate for AtlasOmegaBlock
            if hasattr(block, 'gamma_gate'):
                gamma = block.gamma_gate
                if hasattr(gamma, 'last_gamma_mean'):
                    stats[f"layer_{i}/gamma_mean"] = gamma.last_gamma_mean

        return stats

    def _get_memory_state_stats(self) -> Dict[str, Any]:
        """Get statistics from current memory states."""
        if not self.memory_states:
            return {}

        stats = {}
        for i, state in enumerate(self.memory_states):
            if state is None:
                continue

            # Handle different state tuple formats:
            # Original memory: (W, m) - 2 elements
            # Omega memory: (M, S, context_buffer) - 3 elements
            if len(state) == 2:
                W, m = state
                # W matrix stats
                W_sample = W[0].detach()  # First batch item
                stats[f"layer_{i}/W_norm"] = W_sample.norm().item()
                stats[f"layer_{i}/W_mean"] = W_sample.mean().item()
                stats[f"layer_{i}/W_std"] = W_sample.std().item()
                stats[f"layer_{i}/W_max"] = W_sample.abs().max().item()

                # Momentum stats
                m_sample = m[0].detach()
                stats[f"layer_{i}/m_norm"] = m_sample.norm().item()
            elif len(state) >= 3:
                # Omega memory: (M, S, context_buffer)
                M, S, context_buffer = state[0], state[1], state[2]

                # M (Memory) matrix stats
                M_sample = M[0].detach()  # First batch item
                stats[f"layer_{i}/M_norm"] = M_sample.norm().item()
                stats[f"layer_{i}/M_mean"] = M_sample.mean().item()
                stats[f"layer_{i}/M_std"] = M_sample.std().item()
                stats[f"layer_{i}/M_max"] = M_sample.abs().max().item()

                # S (Momentum) stats
                S_sample = S[0].detach()
                stats[f"layer_{i}/S_norm"] = S_sample.norm().item()

                # Context buffer stats
                if context_buffer and len(context_buffer) > 0:
                    stats[f"layer_{i}/context_len"] = len(context_buffer)

        return stats

    def _log_training_step(
        self,
        step: int,
        loss: float,
        grad_norms: Dict[str, float],
        memory_stats: Dict[str, Any],
        state_stats: Dict[str, Any],
        metrics: Optional[Dict] = None,
        elapsed_hours: float = 0.0,
        steps_per_sec: float = 0.0,
        eta_hours: float = 0.0,
    ):
        """Log detailed training step information."""
        if not self.is_main or not self.train_log_file:
            return

        # Get LR for each parameter group
        lrs = {}
        for pg in self.optimizer.param_groups:
            name = pg.get('name', 'unnamed')
            lrs[f"lr_{name}"] = pg['lr']

        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "stage": self.current_stage,
            "loss": loss,
            "perplexity": min(torch.exp(torch.tensor(loss)).item(), 1e6),
            # Timing metrics
            "elapsed_hours": round(elapsed_hours, 3),
            "steps_per_sec": round(steps_per_sec, 3),
            "eta_hours": round(eta_hours, 2),
            # Per-group learning rates
            **lrs,
            # Gradient norms
            "grad_norm_total": grad_norms.get("total", 0),
            "grad_norm_memory": grad_norms.get("memory", 0),
            "grad_norm_attention": grad_norms.get("attention", 0),
            "grad_norm_ffn": grad_norms.get("ffn", 0),
            "grad_norm_retention": grad_norms.get("retention", 0),
            "grad_norm_gate": grad_norms.get("gate", 0),
        }

        # Add memory module stats
        record.update(memory_stats)

        # Add memory state stats
        record.update(state_stats)

        # Add any additional metrics from forward pass
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    record[k] = v
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, (int, float)):
                            record[f"{k}/{k2}"] = v2

        with open(self.train_log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def train(self):
        """Run training loop."""
        if self.use_tnt:
            self._train_tnt()
        else:
            self._train_standard()

    def _train_standard(self):
        """Standard training without TNT stages."""
        self.model.train()
        data_iter = iter(self.train_loader)

        pbar = tqdm(range(self.total_steps), desc="Training", disable=not self.is_main)
        accum_loss = 0.0
        start_time = time.time()

        for step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.memory_states = None

            loss, metrics = self._train_step(batch, chunk_size=self.stage1_chunk_size)
            accum_loss += loss

            # Optimizer step after accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Get gradient norms BEFORE optimizer step
                grad_norms = self._compute_gradient_norms()

                self._optimizer_step()

                # Log every N optimizer steps (not raw steps)
                optimizer_step = (step + 1) // self.grad_accum_steps
                if self.is_main and optimizer_step % self.log_every == 0:
                    # Get component stats
                    memory_stats = self._get_memory_stats()
                    state_stats = self._get_memory_state_stats()

                    # Log everything
                    self._log_training_step(
                        step, accum_loss, grad_norms,
                        memory_stats, state_stats, metrics
                    )

                    pbar.set_postfix({
                        "loss": f"{accum_loss:.4f}",
                        "grad": f"{grad_norms['total']:.2e}",
                    })

                accum_loss = 0.0

            # Validation
            if self.val_loader is not None and step > 0 and step % self.val_every == 0:
                self._validate(step)

            # Checkpoint
            if self.is_main and step > 0 and step % self.save_every == 0:
                self._save_checkpoint(step)

            self.global_step = step

        if self.is_main:
            self._save_checkpoint(self.global_step)
            elapsed = time.time() - start_time
            print(f"\nTraining complete in {elapsed/3600:.2f} hours")

    def _train_tnt(self):
        """TNT two-stage training with resume support."""
        # Calculate remaining steps based on global_step (set by load_checkpoint)
        stage1_end = self.stage1_steps
        stage2_end = self.stage1_steps + self.stage2_steps

        # Determine where to resume from
        if self.global_step >= stage2_end:
            if self.is_main:
                print(f"\n[RESUME] Training already complete (step {self.global_step} >= {stage2_end})")
            return
        elif self.global_step >= stage1_end:
            # Resume in Stage 2
            remaining_stage2 = stage2_end - self.global_step
            if self.is_main:
                print(f"\n[RESUME] Resuming Stage 2 at step {self.global_step}, {remaining_stage2} steps remaining")
                print(f"=== Stage 2: chunk_size={self.stage2_chunk_size} ===")
            self.current_stage = 2
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr * 0.1
            self._train_stage(
                steps=remaining_stage2,
                chunk_size=self.stage2_chunk_size,
            )
        else:
            # Resume in Stage 1 (or start fresh)
            remaining_stage1 = stage1_end - self.global_step
            if self.global_step > 0:
                if self.is_main:
                    print(f"\n[RESUME] Resuming Stage 1 at step {self.global_step}, {remaining_stage1} steps remaining")
            if self.is_main:
                print(f"\n=== Stage 1: chunk_size={self.stage1_chunk_size} ===")
            self.current_stage = 1
            self._train_stage(
                steps=remaining_stage1,
                chunk_size=self.stage1_chunk_size,
            )

            # Check for early stopping before Stage 2
            if self.early_stopped:
                if self.is_main:
                    print(f"\n[EARLY STOPPING] Skipping Stage 2 due to early stopping in Stage 1")
            else:
                # Stage 2 - reduce LR
                if self.is_main:
                    print(f"\n=== Stage 2: chunk_size={self.stage2_chunk_size} ===")
                self.current_stage = 2
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1

                self._train_stage(
                    steps=self.stage2_steps,
                    chunk_size=self.stage2_chunk_size,
                )

        # Training complete - save final checkpoint
        if self.is_main:
            self._save_checkpoint(self.global_step)
            print(f"\nSaved final checkpoint at step {self.global_step}")

            total_time = time.time() - self.training_start_time if self.training_start_time else 0
            duration_hours = total_time / 3600
            print(f"\nTNT training complete!")
            print(f"Total time: {duration_hours:.2f} hours")
            print(f"Final loss: {self.last_good_loss:.4f}")

            # Send completion alert
            if self.alerts:
                self.alerts.complete(self.total_steps, self.last_good_loss, duration_hours)

    def _train_stage(self, steps: int, chunk_size: int):
        """Train for one TNT stage."""
        self.model.train()
        data_iter = iter(self.train_loader)

        # Initialize timing on first stage
        if self.training_start_time is None:
            self.training_start_time = time.time()
            self.last_log_time = time.time()

        pbar = tqdm(range(steps), desc=f"Stage {self.current_stage}", disable=not self.is_main)
        accum_loss = 0.0

        for stage_step in pbar:
            # Check for early stopping
            if self.early_stopped:
                if self.is_main:
                    print(f"\n[EARLY STOPPING] Breaking training loop at step {self.global_step + stage_step}")
                    # CRITICAL: Save checkpoint before early stop exit
                    self._save_checkpoint(self.global_step + stage_step)
                    print(f"Saved early-stop checkpoint at step {self.global_step + stage_step}")
                break

            # Check for signal interrupt (Ctrl+C, kill)
            if self._interrupt_received:
                if self.is_main:
                    print(f"\n[INTERRUPT] Signal received - saving checkpoint and exiting gracefully")
                    self._save_checkpoint(self.global_step + stage_step)
                    print(f"Saved interrupt checkpoint at step {self.global_step + stage_step}")
                    if self.alerts:
                        self.alerts.critical(f"Training interrupted at step {self.global_step + stage_step}. Checkpoint saved.")
                raise KeyboardInterrupt("Training interrupted by signal - checkpoint saved")

            step = self.global_step + stage_step

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.memory_states = None

            # Periodic memory reset to prevent memorization
            self.steps_since_memory_reset += 1
            if self.steps_since_memory_reset >= self.memory_reset_every:
                if self.is_main and stage_step % 100 == 0:  # Only log occasionally
                    print(f"\n[MEMORY RESET] Resetting memory states at step {step} to prevent memorization")
                self.memory_states = None
                self.steps_since_memory_reset = 0

            loss, metrics = self._train_step(batch, chunk_size=chunk_size)
            accum_loss += loss

            if (stage_step + 1) % self.grad_accum_steps == 0:
                # Get gradient norms BEFORE optimizer step
                grad_norms = self._compute_gradient_norms()

                self._optimizer_step()

                # Log every N optimizer steps (not raw steps)
                optimizer_step = (stage_step + 1) // self.grad_accum_steps
                avg_loss = accum_loss / self.grad_accum_steps

                # === NaN DETECTION ===
                if math.isnan(avg_loss) or math.isinf(avg_loss):
                    if self.is_main:
                        print(f"\n[FATAL] NaN/Inf detected at step {step}!")
                        print(f"Last good loss: {self.last_good_loss:.4f}")
                        # CRITICAL: Save checkpoint before NaN exit (use last good step)
                        last_good_step = max(0, step - self.grad_accum_steps)
                        self._save_checkpoint(last_good_step)
                        print(f"Saved pre-NaN checkpoint at step {last_good_step}")
                        if self.alerts:
                            self.alerts.nan_detected(step, self.last_good_loss)
                    # Stop training
                    raise RuntimeError(f"NaN detected at step {step}. Training stopped.")

                # === LOSS EXPLOSION DETECTION ===
                if self.last_good_loss < float('inf') and avg_loss > self.last_good_loss * 3:
                    if self.is_main:
                        print(f"\n[WARNING] Loss explosion at step {step}: {self.last_good_loss:.4f} -> {avg_loss:.4f}")
                        if self.alerts:
                            self.alerts.loss_explosion(step, self.last_good_loss, avg_loss)

                # Track last good loss
                if not math.isnan(avg_loss) and not math.isinf(avg_loss):
                    self.last_good_loss = avg_loss

                if self.is_main and optimizer_step % self.log_every == 0:
                    # Get component stats
                    memory_stats = self._get_memory_stats()
                    state_stats = self._get_memory_state_stats()

                    # Calculate timing metrics
                    current_time = time.time()
                    elapsed_total = current_time - self.training_start_time
                    steps_completed = step + 1
                    steps_per_sec = steps_completed / elapsed_total if elapsed_total > 0 else 0
                    eta_seconds = (self.total_steps - steps_completed) / steps_per_sec if steps_per_sec > 0 else 0
                    eta_hours = eta_seconds / 3600

                    # Log everything (with timing)
                    self._log_training_step(
                        step, accum_loss, grad_norms,
                        memory_stats, state_stats, metrics,
                        elapsed_hours=elapsed_total / 3600,
                        steps_per_sec=steps_per_sec,
                        eta_hours=eta_hours,
                    )

                    pbar.set_postfix({
                        "loss": f"{accum_loss:.4f}",
                        "grad": f"{grad_norms['total']:.2e}",
                        "stage": self.current_stage,
                    })

                    # === PROGRESS REPORT ===
                    self.steps_since_last_progress += self.log_every
                    if self.alerts and self.steps_since_last_progress >= self.progress_report_interval:
                        self.alerts.progress(step, self.total_steps, avg_loss, eta_hours)
                        self.steps_since_last_progress = 0

                accum_loss = 0.0

            if self.val_loader is not None and stage_step > 0 and stage_step % self.val_every == 0:
                self._validate(step)

            if self.is_main and stage_step > 0 and stage_step % self.save_every == 0:
                self._save_checkpoint(step)
                # === CHECKPOINT ALERT ===
                if self.alerts:
                    self.alerts.checkpoint(step, self.last_good_loss)

        self.global_step += steps

    def _detach_memory_state(self, state):
        """Detach memory state tensors from the computation graph.

        Handles both Atlas (2-tuple) and AtlasOmega (3-tuple) memory formats.
        """
        if state is None:
            return None

        # Handle tuple of tensors (could be 2, 3, or more elements)
        if isinstance(state, tuple):
            detached = []
            for item in state:
                if isinstance(item, torch.Tensor):
                    detached.append(item.detach())
                elif isinstance(item, list):
                    # Context buffer in AtlasOmega is a list of tensors
                    detached.append([t.detach() if isinstance(t, torch.Tensor) else t for t in item])
                else:
                    detached.append(item)
            return tuple(detached)

        # Single tensor
        if isinstance(state, torch.Tensor):
            return state.detach()

        return state

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        chunk_size: int,
    ) -> tuple:
        """Single training step with optional chunking."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch.get("labels", input_ids).to(self.device)

        model = self._get_model()

        # Process in chunks for TNT
        seq_len = input_ids.shape[1]
        n_chunks = (seq_len + chunk_size - 1) // chunk_size

        total_loss = 0.0
        metrics = None

        with autocast(device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, seq_len)

                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                # Detach memory states to avoid backprop through previous chunks
                if self.memory_states:
                    self.memory_states = [
                        self._detach_memory_state(state)
                        for state in self.memory_states
                    ]

                loss, self.memory_states, chunk_metrics = model.compute_loss(
                    chunk_input,
                    chunk_labels,
                    memory_states=self.memory_states,
                    return_metrics=(i == 0),
                )

                # Scale and backward per chunk to avoid graph accumulation
                chunk_loss = loss / (n_chunks * self.grad_accum_steps)

                if self.use_amp:
                    self.scaler.scale(chunk_loss).backward()
                else:
                    chunk_loss.backward()

                total_loss += loss.item() / n_chunks

                if chunk_metrics:
                    metrics = chunk_metrics

        return total_loss, metrics

    def _optimizer_step(self):
        """Optimizer step with gradient clipping."""
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.scheduler.step()
        self.optimizer.zero_grad()

    def _log_step(self, step: int, loss: float, metrics: Optional[Dict]):
        """Log training metrics (main process only)."""
        if self.logger:
            lr = self.scheduler.get_last_lr()[0]
            log_metrics = metrics.copy() if metrics else {}
            log_metrics["stage"] = self.current_stage
            self.logger.log_step(step, loss, log_metrics, lr)

    @torch.no_grad()
    def _validate(self, step: int):
        """Run validation with accumulated memory (matching training behavior)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        # Initialize fresh memory state for validation
        # This mirrors how training starts with fresh memory per epoch
        val_memory_states = None

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            model = self._get_model()
            with autocast(device_type="cuda", enabled=self.use_amp, dtype=torch.bfloat16):
                # Pass and accumulate memory states (matching training behavior)
                loss, val_memory_states, _ = model.compute_loss(
                    input_ids,
                    labels,
                    memory_states=val_memory_states,
                )

            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:
                break

        # Aggregate across GPUs
        if self.world_size > 1:
            loss_tensor = torch.tensor([total_loss, n_batches], device=self.device)
            dist.all_reduce(loss_tensor)
            total_loss = loss_tensor[0].item()
            n_batches = int(loss_tensor[1].item())

        avg_loss = total_loss / max(n_batches, 1)
        val_ppl = torch.exp(torch.tensor(avg_loss)).item()

        if self.is_main and self.logger:
            self.logger.log_validation(step, avg_loss, val_ppl)
            print(f"\nStep {step} | Val Loss: {avg_loss:.4f} | Val PPL: {val_ppl:.2f}")

        # Early stopping check
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.val_patience_counter = 0
            if self.is_main:
                print(f"  New best validation loss! Resetting patience counter.")
        else:
            self.val_patience_counter += 1
            if self.is_main:
                print(f"  No improvement. Patience: {self.val_patience_counter}/{self.val_patience}")

            if self.val_patience_counter >= self.val_patience:
                self.early_stopped = True
                if self.is_main:
                    print(f"\n[EARLY STOPPING] No improvement for {self.val_patience} validations.")
                    print(f"Best validation loss: {self.best_val_loss:.4f}")
                    if self.alerts:
                        self.alerts.critical(f"EARLY STOPPED at step {step}. Best val loss: {self.best_val_loss:.4f}")

        self.model.train()

    def _save_checkpoint(self, step: int):
        """Save checkpoint (main process only)."""
        if not self.is_main:
            return

        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        model = self._get_model()
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": model.config,
            "stage": self.current_stage,
        }

        path = checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint: {path}")

    def load_checkpoint(self, path: Path):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        model = self._get_model()
        model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["step"]
        self.current_stage = checkpoint.get("stage", 1)

        if self.is_main:
            print(f"Loaded checkpoint from {path}, step {self.global_step}")

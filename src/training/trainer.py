"""
Atlas trainer with TNT-style two-stage training.

Stage 1: Large chunks, hierarchical memory, periodic resets
Stage 2: Small chunks, fine-tuning for accuracy

Reference: TNT paper (arXiv:2511.07343)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from tqdm import tqdm
import time

from ..model.atlas import Atlas, AtlasConfig
from .metrics import MetricsLogger, MemoryProfiler


class AtlasTrainer:
    """
    Basic Atlas trainer without TNT modifications.

    Good for initial debugging and baseline comparison.
    """

    def __init__(
        self,
        model: Atlas,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        total_steps: int = 50000,
        grad_accum_steps: int = 1,
        output_dir: Path = Path("runs/atlas"),
        device: torch.device = torch.device("cuda"),
        log_every: int = 10,
        val_every: int = 1000,
        save_every: int = 5000,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.grad_accum_steps = grad_accum_steps

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.log_every = log_every
        self.val_every = val_every
        self.save_every = save_every

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        # LR scheduler: warmup then cosine
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=lr * 0.1,
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        # Metrics
        self.logger = MetricsLogger(self.output_dir / "metrics")
        self.profiler = MemoryProfiler(device)

        # State
        self.global_step = 0
        self.memory_states = None

    def train(self):
        """Run training loop."""
        self.model.train()

        data_iter = iter(self.train_loader)
        pbar = tqdm(range(self.total_steps), desc="Training")

        accum_loss = 0.0
        start_time = time.time()

        for step in pbar:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                # Reset memory states on epoch boundary
                self.memory_states = None

            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Forward
            loss, self.memory_states, metrics = self.model.compute_loss(
                input_ids,
                labels,
                memory_states=self.memory_states,
                return_metrics=(step % self.log_every == 0),
            )

            # Backward (with gradient accumulation)
            loss = loss / self.grad_accum_steps
            loss.backward()
            accum_loss += loss.item()

            # Optimizer step
            if (step + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                # Log
                if step % self.log_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.log_step(
                        step=step,
                        loss=accum_loss * self.grad_accum_steps,
                        metrics=metrics,
                        lr=current_lr,
                    )

                    # Update progress bar
                    pbar.set_postfix({
                        "loss": f"{accum_loss * self.grad_accum_steps:.4f}",
                        "lr": f"{current_lr:.2e}",
                    })

                accum_loss = 0.0

            # Validation
            if self.val_loader and step > 0 and step % self.val_every == 0:
                val_loss = self.validate()
                self.logger.log_validation(step, val_loss, torch.exp(torch.tensor(val_loss)).item())
                self.model.train()

            # Checkpoint
            if step > 0 and step % self.save_every == 0:
                self.save_checkpoint(step)

            self.global_step = step

        # Final save
        self.save_checkpoint(self.global_step)

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/3600:.2f} hours")
        print(f"Final metrics: {self.logger.get_summary()}")

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            loss, _, _ = self.model.compute_loss(input_ids, labels)
            total_loss += loss.item()
            n_batches += 1

            if n_batches >= 50:  # Limit validation batches
                break

        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            "step": step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.model.config,
        }

        path = checkpoint_dir / f"checkpoint_{step}.pt"
        torch.save(checkpoint, path)
        print(f"\nSaved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["step"]
        print(f"Loaded checkpoint from {path}, step {self.global_step}")


class TNTTrainer(AtlasTrainer):
    """
    TNT-style two-stage trainer.

    Stage 1: Large chunks with hierarchical memory
    Stage 2: Small chunks for fine-grained accuracy

    Reference: TNT paper Section 4
    """

    def __init__(
        self,
        model: Atlas,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        # Stage 1 config
        stage1_chunk_size: int = 2048,
        stage1_steps: int = 45000,
        # Stage 2 config
        stage2_chunk_size: int = 256,
        stage2_steps: int = 5000,
        # Other
        **kwargs,
    ):
        super().__init__(model, train_loader, val_loader, **kwargs)

        self.stage1_chunk_size = stage1_chunk_size
        self.stage1_steps = stage1_steps
        self.stage2_chunk_size = stage2_chunk_size
        self.stage2_steps = stage2_steps

        self.total_steps = stage1_steps + stage2_steps
        self.current_stage = 1

    def train(self):
        """Run two-stage training."""
        print(f"=== Stage 1: Large chunks ({self.stage1_chunk_size}) ===")
        self.current_stage = 1
        self._train_stage(
            chunk_size=self.stage1_chunk_size,
            steps=self.stage1_steps,
            reset_local=True,  # Reset local memory per chunk
        )

        print(f"\n=== Stage 2: Small chunks ({self.stage2_chunk_size}) ===")
        self.current_stage = 2
        # Reduce LR for fine-tuning
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * 0.1

        self._train_stage(
            chunk_size=self.stage2_chunk_size,
            steps=self.stage2_steps,
            reset_local=False,  # Don't reset in stage 2
        )

        print("\nTNT training complete!")

    def _train_stage(
        self,
        chunk_size: int,
        steps: int,
        reset_local: bool = True,
    ):
        """Train for one stage."""
        self.model.train()

        data_iter = iter(self.train_loader)
        pbar = tqdm(range(steps), desc=f"Stage {self.current_stage}")

        accum_loss = 0.0

        for stage_step in pbar:
            step = self.global_step + stage_step

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                self.memory_states = None

            input_ids = batch["input_ids"].to(self.device)
            labels = batch.get("labels", input_ids).to(self.device)

            # Process in chunks (TNT style)
            seq_len = input_ids.shape[1]
            n_chunks = (seq_len + chunk_size - 1) // chunk_size

            chunk_losses = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, seq_len)

                chunk_input = input_ids[:, start:end]
                chunk_labels = labels[:, start:end]

                # Reset local memory per chunk in stage 1
                if reset_local and i > 0:
                    # Keep global state, reset within-chunk state
                    # (simplified: just detach to break gradient flow)
                    if self.memory_states:
                        self.memory_states = [
                            (W.detach(), m.detach())
                            for W, m in self.memory_states
                        ]

                loss, self.memory_states, metrics = self.model.compute_loss(
                    chunk_input,
                    chunk_labels,
                    memory_states=self.memory_states,
                    return_metrics=(stage_step % self.log_every == 0 and i == 0),
                )
                chunk_losses.append(loss)

            # Average loss over chunks
            total_loss = sum(chunk_losses) / len(chunk_losses)
            total_loss = total_loss / self.grad_accum_steps
            total_loss.backward()
            accum_loss += total_loss.item()

            # Optimizer step
            if (stage_step + 1) % self.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                if stage_step % self.log_every == 0:
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.logger.log_step(
                        step=step,
                        loss=accum_loss * self.grad_accum_steps,
                        metrics={"stage": self.current_stage, "chunk_size": chunk_size},
                        lr=current_lr,
                    )
                    pbar.set_postfix({
                        "loss": f"{accum_loss * self.grad_accum_steps:.4f}",
                        "chunks": n_chunks,
                    })

                accum_loss = 0.0

            # Validation
            if self.val_loader and stage_step > 0 and stage_step % self.val_every == 0:
                val_loss = self.validate()
                self.logger.log_validation(step, val_loss, torch.exp(torch.tensor(val_loss)).item())
                self.model.train()

            # Checkpoint
            if stage_step > 0 and stage_step % self.save_every == 0:
                self.save_checkpoint(step)

        self.global_step += steps

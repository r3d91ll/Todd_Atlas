"""
Episodic DDP Trainer for Atlas with Memory Verification.

Implements the episodic training loop:
1. Storage phase: Present content, allow memory to store
2. Retrieval phase: Test retrieval, heavy penalty for failure

Combined with phase-based gate floor scheduling to prevent collapse.
"""

import os
import time
import math
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..model.attention import GateMode
from .retrieval_verifier import RetrievalVerifier
from .masking import create_masked_batch, create_math_masked_batch, compute_masked_accuracy

# Import alert system (optional)
try:
    from training_framework.monitoring.alert_system import AlertSystem, TelegramConfig
    ALERTS_AVAILABLE = True
except ImportError:
    ALERTS_AVAILABLE = False

# Import grokking metrics (optional)
try:
    from training_framework.monitoring.grokking_metrics import (
        GrokkingDetector,
        GrokkingConfig,
        create_grokking_detector,
    )
    GROKKING_AVAILABLE = True
except ImportError:
    GROKKING_AVAILABLE = False

# Import numerical stability metrics (optional)
try:
    from training_framework.monitoring.numerical_stability import (
        NumericalStabilityMonitor,
        NumericalStabilityMetrics,
    )
    STABILITY_AVAILABLE = True
except ImportError:
    STABILITY_AVAILABLE = False

# Import StableMax loss (arXiv:2501.04697v2)
try:
    from .stablemax import StableCrossEntropyLoss
    STABLEMAX_AVAILABLE = True
except ImportError:
    STABLEMAX_AVAILABLE = False

# Import orthogonal gradient projection (arXiv:2501.04697v2)
try:
    from .orthogonal_grad import apply_orthogonal_projection
    ORTHOGONAL_GRAD_AVAILABLE = True
except ImportError:
    ORTHOGONAL_GRAD_AVAILABLE = False


class TrainingPhase(Enum):
    """Training phase for gate floor scheduling."""
    PHASE1 = "phase1"  # High gate floor (0.30)
    PHASE2 = "phase2"  # Medium gate floor (0.10)
    PHASE3 = "phase3"  # Low gate floor (0.05)


class EpisodePhase(Enum):
    """Current phase within an episodic cycle."""
    STORAGE = "storage"
    RETRIEVAL = "retrieval"


@dataclass
class EpisodicConfig:
    """Configuration for episodic training."""
    # Episode structure
    storage_samples: int = 10      # Number of storage samples per episode
    retrieval_samples: int = 10    # Number of retrieval samples per episode

    # Phase-based gate floor scheduling
    phase1_steps: int = 10000      # Steps in phase 1
    phase2_steps: int = 20000      # Steps in phase 2 (cumulative: 30000)
    phase1_gate_floor: float = 0.30
    phase2_gate_floor: float = 0.10
    phase3_gate_floor: float = 0.05

    # Gate targets for episodic modes
    storage_gate_target: float = 0.80   # Force high gates during storage
    retrieval_gate_floor: float = 0.30  # Minimum gate during retrieval

    # Loss weights
    retrieval_loss_weight: float = 5.0  # Heavy penalty for retrieval failure
    storage_loss_weight: float = 1.0    # Standard LM loss during storage

    # Memory resets
    reset_memory_between_episodes: bool = False

    # Multi-task masked word prediction (NEW)
    use_masked_retrieval: bool = True    # Enable masked word prediction
    mask_token_id: int = 3               # Token ID for [MASK]
    num_masks: int = 1                   # Number of masks per sequence
    pad_token_id: int = 0                # Padding token ID (skip when masking)


@dataclass
class TrainerConfig:
    """Configuration for the trainer."""
    # Training parameters
    max_steps: int = 57000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    grad_clip: float = 1.0

    # Batch settings
    batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Logging and checkpointing
    log_interval: int = 10
    checkpoint_interval: int = 5000
    metrics_path: str = "runs/experiment/metrics_stream.jsonl"
    checkpoint_dir: str = "runs/experiment/checkpoints"

    # Episodic training
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)

    # Device and precision
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile_model: bool = False

    # DDP settings
    use_ddp: bool = False
    local_rank: int = 0

    # Telegram alerts (optional)
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Grokking detection (optional)
    grokking_enabled: bool = False
    grokking_interval: int = 500

    # Numerical stability options (arXiv:2501.04697v2)
    # StableMax: Numerically stable softmax alternative
    use_stablemax: bool = True  # Use StableCrossEntropyLoss instead of F.cross_entropy
    # PerpGrad: Project out weight-aligned gradients for immediate generalization
    use_orthogonal_grad: bool = True
    orthogonal_grad_strength: float = 1.0  # 0.0 = disabled, 1.0 = full projection


class EpisodicDDPTrainer:
    """
    Episodic trainer with DDP support and memory verification.

    Training loop structure:
    ```
    for episode in episodes:
        # Storage phase
        model.set_gate_mode(STORAGE)
        for batch in storage_batches:
            loss = standard_lm_loss(batch)
            verifier.record_storage(batch_hash, targets, memory_state)

        # Retrieval phase
        model.set_gate_mode(RETRIEVAL)
        for batch in retrieval_batches:
            loss = lm_loss + 5x * retrieval_loss
            metrics = verifier.verify_retrieval(batch_hash, logits, memory)
    ```
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        config: TrainerConfig,
        metrics_adapter: Optional[Any] = None,
    ):
        self.config = config
        self.ep_config = config.episodic
        self.device = torch.device(config.device)

        # Set up dtype
        self.dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(config.dtype, torch.bfloat16)

        # Model setup
        self.model = model.to(self.device)
        if config.compile_model:
            self.model = torch.compile(self.model)

        if config.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
            )

        # Data
        self.train_dataloader = train_dataloader
        self.data_iter = iter(train_dataloader)

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        # Track orthogonal grad settings for use in _optimizer_step
        self.use_orthogonal_grad = config.use_orthogonal_grad and ORTHOGONAL_GRAD_AVAILABLE
        self.orthogonal_grad_strength = config.orthogonal_grad_strength
        if self.use_orthogonal_grad:
            print(f"PerpGrad projection enabled (strength: {config.orthogonal_grad_strength})")
        elif config.use_orthogonal_grad and not ORTHOGONAL_GRAD_AVAILABLE:
            print("Warning: PerpGrad requested but not available")

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps - config.warmup_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # StableMax loss criterion (arXiv:2501.04697v2)
        # Numerically stable alternative to softmax that prevents collapse
        self.criterion = None
        if config.use_stablemax and STABLEMAX_AVAILABLE:
            self.criterion = StableCrossEntropyLoss(ignore_index=-100)
            print("StableMax loss enabled")
        elif config.use_stablemax and not STABLEMAX_AVAILABLE:
            print("Warning: StableMax requested but not available, using F.cross_entropy")

        # Retrieval verification
        self.retrieval_verifier = RetrievalVerifier(
            max_buffer_size=100,
            retrieval_loss_weight=self.ep_config.retrieval_loss_weight,
            device=config.device,
        )

        # Metrics adapter (optional)
        self.metrics_adapter = metrics_adapter

        # State tracking
        self.global_step = 0
        self.training_phase = TrainingPhase.PHASE1
        self.episode_phase = EpisodePhase.STORAGE
        self.episode_step = 0
        self.total_episodes = 0

        # Memory states (passed through forward)
        self.memory_states: Optional[List[Tuple]] = None

        # Metrics
        self._metrics_buffer: List[Dict[str, Any]] = []
        self._start_time = time.time()

        # Alert system (optional)
        self.alert_system = None
        if ALERTS_AVAILABLE and config.telegram_enabled and config.telegram_bot_token and config.telegram_chat_id:
            tg_config = TelegramConfig(
                bot_token=config.telegram_bot_token,
                chat_id=config.telegram_chat_id,
            )
            self.alert_system = AlertSystem(tg_config)
            self.alert_system.send_alert(
                'INFO',
                f'🚀 Atlas training started: {config.max_steps} steps, {config.device}'
            )
            print("Telegram alerts enabled")

        # Grokking detection (optional)
        self.grokking_detector = None
        self._last_grokking_metrics = None
        if GROKKING_AVAILABLE and config.grokking_enabled:
            grok_config = GrokkingConfig(
                enabled=True,
                metrics_interval=config.grokking_interval,
            )
            self.grokking_detector = GrokkingDetector(grok_config)
            print(f"Grokking detection enabled (interval: {config.grokking_interval} steps)")

        # Numerical stability monitor (optional)
        # Based on arXiv:2501.04697v2 "Grokking at the Edge of Numerical Stability"
        self.stability_monitor = None
        self._last_stability_metrics = None
        self._last_logits = None  # Cache logits for stability analysis
        if STABILITY_AVAILABLE and config.grokking_enabled:  # Use same flag as grokking
            self.stability_monitor = NumericalStabilityMonitor()
            print("Numerical stability monitoring enabled")

        # Initialize gate floor
        self._update_gate_floor()

    def _get_raw_model(self) -> nn.Module:
        """Get unwrapped model (handles DDP)."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

    def _get_batch(self) -> Dict[str, torch.Tensor]:
        """Get next batch from dataloader."""
        try:
            batch = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.train_dataloader)
            batch = next(self.data_iter)

        # Move to device
        return {k: v.to(self.device) for k, v in batch.items()}

    def _update_training_phase(self) -> None:
        """Update training phase based on step count."""
        if self.global_step < self.ep_config.phase1_steps:
            new_phase = TrainingPhase.PHASE1
        elif self.global_step < self.ep_config.phase1_steps + self.ep_config.phase2_steps:
            new_phase = TrainingPhase.PHASE2
        else:
            new_phase = TrainingPhase.PHASE3

        if new_phase != self.training_phase:
            self.training_phase = new_phase
            self._update_gate_floor()
            print(f"[Step {self.global_step}] Entering {new_phase.value}")

    def _update_gate_floor(self) -> None:
        """Update gate floor based on current training phase."""
        model = self._get_raw_model()
        floor = {
            TrainingPhase.PHASE1: self.ep_config.phase1_gate_floor,
            TrainingPhase.PHASE2: self.ep_config.phase2_gate_floor,
            TrainingPhase.PHASE3: self.ep_config.phase3_gate_floor,
        }[self.training_phase]

        model.set_gate_floor(floor)

    def _set_episode_mode(self, phase: EpisodePhase) -> None:
        """Set gate mode for current episode phase."""
        model = self._get_raw_model()
        self.episode_phase = phase

        if phase == EpisodePhase.STORAGE:
            model.set_gate_mode(GateMode.STORAGE)
        else:
            model.set_gate_mode(GateMode.RETRIEVAL)

    def _compute_batch_hash(self, batch: Dict[str, torch.Tensor]) -> str:
        """Compute hash for batch matching."""
        return self.retrieval_verifier.compute_batch_hash(batch)

    def _get_memory_state_snapshot(self) -> torch.Tensor:
        """Get a snapshot of current memory state for verification."""
        model = self._get_raw_model()
        states = model.get_memory_state()

        # Concatenate all memory matrices
        memory_tensors = []
        for state in states:
            if state is not None:
                if isinstance(state, tuple) and len(state) > 0:
                    M = state[0]
                    if M is not None:
                        memory_tensors.append(M.detach().flatten())
                elif isinstance(state, torch.Tensor):
                    memory_tensors.append(state.detach().flatten())

        if memory_tensors:
            return torch.cat(memory_tensors)
        else:
            # Return dummy tensor if no memory available
            return torch.zeros(1, device=self.device)

    def _get_lr(self) -> float:
        """Get current learning rate with warmup."""
        if self.global_step < self.config.warmup_steps:
            return self.config.learning_rate * self.global_step / self.config.warmup_steps
        return self.scheduler.get_last_lr()[0]

    def _set_lr(self, lr: float) -> None:
        """Set learning rate in optimizer."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _storage_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute storage phase step.

        During storage:
        - Gate mode is STORAGE (high gate values)
        - Standard LM loss
        - Record what was stored for later verification
        """
        input_ids = batch.get('input_ids', batch.get('inputs'))
        labels = batch.get('labels', input_ids)

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            logits, self.memory_states, block_metrics = self.model(
                input_ids,
                memory_states=self.memory_states,
                return_metrics=True,
            )

            # Standard LM loss (use StableMax if enabled)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if self.criterion is not None:
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            loss = loss * self.ep_config.storage_loss_weight

        # Record storage for verification
        batch_hash = self._compute_batch_hash(batch)
        memory_snapshot = self._get_memory_state_snapshot()
        self.retrieval_verifier.record_storage(
            batch_hash=batch_hash,
            target_tokens=labels,
            memory_state=memory_snapshot,
        )

        # Backward
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # Cache logits for stability analysis (detach to avoid memory leak)
        self._last_logits = logits.detach()

        return {
            'loss': loss.item(),
            'storage_loss': loss.item(),
            'phase': 'storage',
            'batch_hash': batch_hash,
        }

    def _retrieval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute retrieval phase step.

        Routes to appropriate retrieval method based on task type:
        - 'language': Masked word prediction (pure random masking)
        - 'math': Modular arithmetic (answer prediction)

        During retrieval:
        - Gate mode is RETRIEVAL (minimum gate floor)
        - Heavy penalty for wrong answers
        - Track separate accuracies per task type
        """
        # Check task type from multi-task loader
        task_type = batch.get('task_type', 'language')

        if self.ep_config.use_masked_retrieval:
            if task_type == 'math':
                return self._math_retrieval_step(batch)
            else:
                return self._language_retrieval_step(batch)
        else:
            # Legacy retrieval (original behavior)
            return self._legacy_retrieval_step(batch)

    def _language_retrieval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute language retrieval with PURE RANDOM masked word prediction.

        Model sees: "Shall I [MASK] thee to a [MASK] day..."
        Must output: ["compare", "summer"] from memory

        No heuristics. Pure random masking. Model MUST use memory.
        """
        # Create masked version of the batch
        masked_batch, mask_positions, original_tokens = create_masked_batch(
            batch,
            mask_token_id=self.ep_config.mask_token_id,
            num_masks=self.ep_config.num_masks,
            pad_token_id=self.ep_config.pad_token_id,
        )

        input_ids = masked_batch['input_ids'].to(self.device)
        mask_positions = mask_positions.to(self.device)
        original_tokens = original_tokens.to(self.device)

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            logits, self.memory_states, _ = self.model(
                input_ids,
                memory_states=self.memory_states,
                return_metrics=True,
            )

            batch_size = logits.size(0)
            num_masks = self.ep_config.num_masks

            # Gather logits at ALL masked positions
            masked_logits_list = []
            for k in range(num_masks):
                pos = mask_positions[:, k]  # [batch]
                # Handle -1 positions (no mask) by clamping
                valid_pos = pos.clamp(min=0)
                ml = logits[torch.arange(batch_size, device=self.device), valid_pos, :]
                masked_logits_list.append(ml)

            # Stack: [batch, num_masks, vocab]
            masked_logits = torch.stack(masked_logits_list, dim=1)

            # Predictions
            predictions = masked_logits.argmax(dim=-1)  # [batch, num_masks]

            # Exact match accuracy
            accuracy = compute_masked_accuracy(predictions, original_tokens, mask_positions)

            # Loss on masked positions only
            if self.criterion is not None:
                loss = self.criterion(
                    masked_logits.view(-1, masked_logits.size(-1)),
                    original_tokens.view(-1),
                )
            else:
                loss = F.cross_entropy(
                    masked_logits.view(-1, masked_logits.size(-1)),
                    original_tokens.view(-1),
                    ignore_index=-1,  # Ignore -1 tokens (no mask)
                )
            loss = loss * self.ep_config.retrieval_loss_weight

        # Backward
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # Cache logits for stability analysis
        self._last_logits = logits.detach()

        return {
            'loss': loss.item(),
            'retrieval_loss': loss.item(),
            'masked_word_accuracy': accuracy,
            'task_type': 'language',
            'phase': 'retrieval',
            'num_masks': num_masks,
        }

    def _math_retrieval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute modular arithmetic retrieval.

        Input: "23 + 45 = [MASK]"
        Must output: "68" (mod 97)

        Fourier/Circular geometric metrics apply to this task.
        """
        # Create masked version (answer always at last position)
        masked_batch, _mask_positions, _original_tokens = create_math_masked_batch(
            batch,
            mask_token_id=self.ep_config.mask_token_id,
        )

        input_ids = masked_batch['input_ids'].to(self.device)
        answers = batch['answer'].to(self.device)

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            logits, self.memory_states, _ = self.model(
                input_ids,
                memory_states=self.memory_states,
                return_metrics=True,
            )

            # Answer is always at last position for math
            answer_logits = logits[:, -1, :]  # [batch, vocab]
            predictions = answer_logits.argmax(dim=-1)  # [batch]

            # Exact match accuracy
            accuracy = (predictions == answers.squeeze(-1)).float().mean().item()

            # Loss on answer position
            if self.criterion is not None:
                loss = self.criterion(answer_logits, answers.squeeze(-1))
            else:
                loss = F.cross_entropy(answer_logits, answers.squeeze(-1))
            loss = loss * self.ep_config.retrieval_loss_weight

        # Backward
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # Cache logits for stability analysis
        self._last_logits = logits.detach()

        return {
            'loss': loss.item(),
            'retrieval_loss': loss.item(),
            'math_accuracy': accuracy,
            'task_type': 'math',
            'phase': 'retrieval',
        }

    def _legacy_retrieval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Legacy retrieval step (original behavior without masking).

        Kept for backward compatibility with existing experiments.
        """
        input_ids = batch.get('input_ids', batch.get('inputs'))
        labels = batch.get('labels', input_ids)

        with torch.autocast(device_type='cuda', dtype=self.dtype):
            logits, self.memory_states, block_metrics = self.model(
                input_ids,
                memory_states=self.memory_states,
                return_metrics=True,
            )

            # Standard LM loss (use StableMax if enabled)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if self.criterion is not None:
                lm_loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
            else:
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

        # Compute retrieval loss and verification
        batch_hash = self._compute_batch_hash(batch)
        current_memory = self._get_memory_state_snapshot()

        retrieval_loss, verification_metrics = self.retrieval_verifier.compute_retrieval_loss_from_hash(
            batch_hash=batch_hash,
            model_logits=logits,
        )

        # Total loss
        total_loss = lm_loss + retrieval_loss

        # Backward
        scaled_loss = total_loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        # Cache logits for stability analysis (detach to avoid memory leak)
        # Must be set in both storage and retrieval steps for fresh metrics
        self._last_logits = logits.detach()

        metrics = {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'retrieval_loss': retrieval_loss.item(),
            'phase': 'retrieval',
            'batch_hash': batch_hash,
        }
        metrics.update(verification_metrics)

        return metrics

    def _optimizer_step(self) -> Dict[str, float]:
        """Execute optimizer step with gradient clipping and optional PerpGrad projection."""
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip,
        )

        # Apply orthogonal gradient projection (arXiv:2501.04697v2)
        # Removes weight-aligned gradient components to prevent NLM
        if self.use_orthogonal_grad:
            apply_orthogonal_projection(self.model, strength=self.orthogonal_grad_strength)

        # Update weights
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update scheduler (after warmup)
        if self.global_step >= self.config.warmup_steps:
            self.scheduler.step()

        return {'grad_norm': grad_norm.item()}

    def _run_episode(self) -> Dict[str, Any]:
        """
        Run one complete episode (storage + retrieval cycle).

        Returns aggregated metrics for the episode.
        """
        episode_metrics = {
            'storage_losses': [],
            'retrieval_losses': [],
            'retrieval_accuracies': [],
            # Multi-task separate tracking
            'math_accuracies': [],
            'masked_word_accuracies': [],
        }

        # Collect batches for this episode - we'll reuse them for retrieval
        episode_batches = []

        # Storage phase
        self._set_episode_mode(EpisodePhase.STORAGE)
        for _ in range(self.ep_config.storage_samples):
            batch = self._get_batch()
            episode_batches.append(batch)  # Save for retrieval phase
            metrics = self._storage_step(batch)
            episode_metrics['storage_losses'].append(metrics['loss'])

            self.episode_step += 1

            # Optimizer step every gradient_accumulation_steps
            if self.episode_step % self.config.gradient_accumulation_steps == 0:
                opt_metrics = self._optimizer_step()
                self.global_step += 1
                self._update_training_phase()
                self._set_lr(self._get_lr())

        # Retrieval phase - reuse the SAME batches from storage
        self._set_episode_mode(EpisodePhase.RETRIEVAL)
        for i in range(self.ep_config.retrieval_samples):
            # Reuse stored batches (cycle if retrieval_samples > storage_samples)
            batch = episode_batches[i % len(episode_batches)]
            metrics = self._retrieval_step(batch)
            episode_metrics['retrieval_losses'].append(metrics['loss'])

            # Track legacy accuracy
            if 'retrieval_token_accuracy' in metrics:
                episode_metrics['retrieval_accuracies'].append(
                    metrics['retrieval_token_accuracy']
                )

            # Track multi-task accuracies separately
            if 'math_accuracy' in metrics:
                episode_metrics['math_accuracies'].append(metrics['math_accuracy'])
            if 'masked_word_accuracy' in metrics:
                episode_metrics['masked_word_accuracies'].append(metrics['masked_word_accuracy'])

            self.episode_step += 1

            # Optimizer step every gradient_accumulation_steps
            if self.episode_step % self.config.gradient_accumulation_steps == 0:
                opt_metrics = self._optimizer_step()
                self.global_step += 1
                self._update_training_phase()
                self._set_lr(self._get_lr())

        # Aggregate episode metrics
        aggregated = {
            'episode': self.total_episodes,
            'step': self.global_step,
            'storage_loss_mean': sum(episode_metrics['storage_losses']) / len(episode_metrics['storage_losses']),
            'retrieval_loss_mean': sum(episode_metrics['retrieval_losses']) / len(episode_metrics['retrieval_losses']),
            'training_phase': self.training_phase.value,
        }

        # Legacy retrieval accuracy
        if episode_metrics['retrieval_accuracies']:
            aggregated['retrieval_accuracy_mean'] = (
                sum(episode_metrics['retrieval_accuracies']) /
                len(episode_metrics['retrieval_accuracies'])
            )

        # Multi-task accuracies (separate tracking)
        if episode_metrics['math_accuracies']:
            aggregated['math_accuracy'] = (
                sum(episode_metrics['math_accuracies']) /
                len(episode_metrics['math_accuracies'])
            )

        if episode_metrics['masked_word_accuracies']:
            aggregated['masked_word_accuracy'] = (
                sum(episode_metrics['masked_word_accuracies']) /
                len(episode_metrics['masked_word_accuracies'])
            )

        # Combined overall accuracy (50-50 weighting to match training split)
        if episode_metrics['math_accuracies'] and episode_metrics['masked_word_accuracies']:
            aggregated['overall_accuracy'] = (
                aggregated['math_accuracy'] * 0.5 +
                aggregated['masked_word_accuracy'] * 0.5
            )

        # Reset memory between episodes if configured
        if self.ep_config.reset_memory_between_episodes:
            self._get_raw_model().reset_all_memory()
            self.memory_states = None

        self.total_episodes += 1
        return aggregated

    def _collect_metrics(self, episode_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive metrics for logging."""
        model = self._get_raw_model()
        metrics = {
            'step': self.global_step,
            'episode': self.total_episodes,
            'timestamp': time.time(),
            'elapsed_hours': (time.time() - self._start_time) / 3600,
            'learning_rate': self._get_lr(),
            'training_phase': self.training_phase.value,
        }

        # Episode metrics
        metrics.update(episode_metrics)

        # Gate metrics
        gate_metrics = model.get_gate_metrics()
        metrics.update(gate_metrics)

        # Adapter metrics (if available)
        if self.metrics_adapter is not None:
            adapter_metrics = self.metrics_adapter.collect(model, episode_metrics)
            metrics.update(adapter_metrics)

        # Verifier statistics
        verifier_stats = self.retrieval_verifier.get_statistics()
        metrics['verifier_success_rate'] = verifier_stats['success_rate']
        metrics['verifier_buffer_size'] = verifier_stats['buffer_size']

        # Grokking metrics (computed at interval)
        if self.grokking_detector is not None:
            if self.global_step % self.config.grokking_interval == 0:
                val_metrics = {
                    'retrieval_accuracy': episode_metrics.get('retrieval_accuracy_mean', 0),
                }
                # Pass gate_mean for Atlas-specific gate health check
                current_gate_mean = metrics.get('gate_mean', 1.0)
                grok_metrics = self.grokking_detector.compute_metrics(
                    model=model,
                    step=self.global_step,
                    val_metrics=val_metrics,
                    gate_mean=current_gate_mean,
                )
                self._last_grokking_metrics = grok_metrics
                metrics.update(grok_metrics.to_dict())

                # Log grokking phase with gate health
                if self.global_step % self.config.log_interval == 0:
                    gate_status = "✓" if grok_metrics.gate_healthy else "⚠ COLLAPSED"
                    print(
                        f"  [Grokking] Phase: {grok_metrics.phase} | "
                        f"Gate: {current_gate_mean:.1%} {gate_status} | "
                        f"Fourier: {grok_metrics.embedding_fourier_concentration:.3f} | "
                        f"Circular: {grok_metrics.embedding_circular_fit:.3f}"
                    )
            elif self._last_grokking_metrics is not None:
                # Include last computed grokking phase in metrics
                metrics['grokking/phase'] = self._last_grokking_metrics.phase

        # Numerical stability metrics (computed at same interval as grokking)
        # Based on arXiv:2501.04697v2 "Grokking at the Edge of Numerical Stability"
        if self.stability_monitor is not None:
            if self.global_step % self.config.grokking_interval == 0:
                stability_metrics = self.stability_monitor.compute_metrics(
                    model=model,
                    logits=self._last_logits,
                    step=self.global_step,
                )
                self._last_stability_metrics = stability_metrics
                metrics.update(stability_metrics.to_dict())

                # Log stability warnings
                if stability_metrics.sc_risk in ('high', 'critical'):
                    print(
                        f"  [Stability] ⚠️ SC Risk: {stability_metrics.sc_risk} | "
                        f"SC Fraction: {stability_metrics.sc_fraction:.1%} | "
                        f"Max Logit: {stability_metrics.max_logit_mean:.1f}"
                    )
                if stability_metrics.nlm_active:
                    print(
                        f"  [Stability] ⚠️ NLM Active: Grad-Weight Cosine = {stability_metrics.grad_weight_cosine:.3f}"
                    )
            elif self._last_stability_metrics is not None:
                # Include last computed stability risk in metrics
                metrics['stability/sc_risk'] = self._last_stability_metrics.sc_risk

        return metrics

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to console and file."""
        # Console output
        if self.global_step % self.config.log_interval == 0:
            loss = metrics.get('storage_loss_mean', metrics.get('loss', 0))
            gate_mean = metrics.get('gate_mean', 0)
            collapse_risk = metrics.get('gate_collapse_risk', 0)
            ret_acc = metrics.get('retrieval_accuracy_mean', 0)

            print(
                f"[Step {self.global_step:>6}] "
                f"Loss: {loss:.4f} | "
                f"Gate: {gate_mean:.2%} | "
                f"Collapse Risk: {collapse_risk:.1%} | "
                f"Ret Acc: {ret_acc:.1%} | "
                f"Phase: {self.training_phase.value}"
            )

        # Check alerts
        self._check_alerts(metrics)

        # File output (JSONL for Streamlit)
        self._metrics_buffer.append(metrics)
        if len(self._metrics_buffer) >= 1:
            self._flush_metrics()

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and send alerts."""
        if self.alert_system is None:
            return

        # Only check every log_interval to avoid spam
        if self.global_step % self.config.log_interval != 0:
            return

        collapse_risk = metrics.get('gate_collapse_risk', 0)
        gate_mean = metrics.get('gate_mean', 1.0)
        ret_acc = metrics.get('retrieval_accuracy_mean', 1.0)
        loss = metrics.get('storage_loss_mean', metrics.get('loss', 0))

        # Critical: Gate collapse
        if collapse_risk > 0.80:
            self.alert_system.send_alert(
                'CRITICAL',
                f'🚨 Gate collapse risk {collapse_risk:.1%} at step {self.global_step}'
            )
        elif collapse_risk > 0.50:
            self.alert_system.send_alert(
                'WARNING',
                f'⚠️ Gate collapse risk {collapse_risk:.1%} at step {self.global_step}'
            )

        # Warning: Low gate mean
        if gate_mean < 0.10:
            self.alert_system.send_alert(
                'WARNING',
                f'⚠️ Gate mean very low: {gate_mean:.1%} at step {self.global_step}'
            )

        # Warning: Low retrieval accuracy
        if ret_acc < 0.30 and ret_acc > 0:
            self.alert_system.send_alert(
                'WARNING',
                f'⚠️ Retrieval accuracy low: {ret_acc:.1%} at step {self.global_step}'
            )

        # Critical: Loss spike
        if loss > 10.0:
            self.alert_system.send_alert(
                'CRITICAL',
                f'🚨 Loss spike: {loss:.2f} at step {self.global_step}'
            )

    def _flush_metrics(self) -> None:
        """Flush metrics buffer to file."""
        import json
        os.makedirs(os.path.dirname(self.config.metrics_path), exist_ok=True)

        with open(self.config.metrics_path, 'a') as f:
            for m in self._metrics_buffer:
                f.write(json.dumps(m) + '\n')
        self._metrics_buffer.clear()

    def _save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        model = self._get_raw_model()
        checkpoint = {
            'step': self.global_step,
            'episode': self.total_episodes,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_phase': self.training_phase.value,
            'config': {
                'trainer': self.config.__dict__,
                'episodic': self.ep_config.__dict__,
            },
        }

        path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.global_step}.pt"
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint and resume training."""
        # weights_only=False needed for PyTorch 2.6+ (changed default)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        model = self._get_raw_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.global_step = checkpoint['step']
        self.total_episodes = checkpoint['episode']
        self.training_phase = TrainingPhase(checkpoint['training_phase'])

        self._update_gate_floor()
        print(f"Resumed from step {self.global_step}, episode {self.total_episodes}")

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns final training statistics.
        """
        print(f"Starting episodic training: {self.config.max_steps} steps")
        print(f"  Storage samples/episode: {self.ep_config.storage_samples}")
        print(f"  Retrieval samples/episode: {self.ep_config.retrieval_samples}")
        print(f"  Phase 1 gate floor: {self.ep_config.phase1_gate_floor}")

        best_retrieval_acc = 0.0
        self.model.train()
        self.optimizer.zero_grad()

        while self.global_step < self.config.max_steps:
            # Run one episode
            episode_metrics = self._run_episode()

            # Collect and log metrics
            metrics = self._collect_metrics(episode_metrics)
            self._log_metrics(metrics)

            # Early stopping on grokking (Atlas-aware)
            if self._last_grokking_metrics is not None:
                phase = self._last_grokking_metrics.phase
                if phase == "grokked":
                    print(f"\n🎯 GROKKING ACHIEVED at step {self.global_step}!")
                    print(f"   Gate mean: {self._last_grokking_metrics.gate_mean:.1%} (healthy)")
                    print(f"   Fourier concentration: {self._last_grokking_metrics.embedding_fourier_concentration:.3f}")
                    print(f"   Circular fit: {self._last_grokking_metrics.embedding_circular_fit:.3f}")
                    print(f"   Effective dim ratio: {self._last_grokking_metrics.embedding_effective_dim_ratio:.1%}")
                    self._flush_metrics()
                    self._save_checkpoint(is_best=True)
                    break
                elif phase == "gate_collapse":
                    # Log warning but continue training - gates may recover
                    if self.global_step % (self.config.log_interval * 10) == 0:
                        print(f"\n⚠️  GATE COLLAPSE detected at step {self.global_step}")
                        print(f"   Gate mean: {self._last_grokking_metrics.gate_mean:.1%} (below threshold)")
                        print("   Memory may be bypassed - consider adjusting gate_floor or memory learning rate")

            # Track best retrieval accuracy
            ret_acc = metrics.get('retrieval_accuracy_mean', 0)
            is_best = ret_acc > best_retrieval_acc
            if is_best:
                best_retrieval_acc = ret_acc

            # Checkpoint
            if self.global_step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(is_best=is_best)

        # Final flush and checkpoint
        self._flush_metrics()
        self._save_checkpoint(is_best=False)

        # Final statistics
        verifier_stats = self.retrieval_verifier.get_statistics()
        training_hours = (time.time() - self._start_time) / 3600

        # Send completion alert
        if self.alert_system:
            self.alert_system.send_alert(
                'INFO',
                f'✅ Atlas training complete!\n'
                f'Steps: {self.global_step}\n'
                f'Time: {training_hours:.1f}h\n'
                f'Best ret acc: {best_retrieval_acc:.1%}'
            )

        return {
            'final_step': self.global_step,
            'total_episodes': self.total_episodes,
            'best_retrieval_accuracy': best_retrieval_acc,
            'verifier_success_rate': verifier_stats['success_rate'],
            'training_time_hours': training_hours,
        }

"""
Phase Detection Framework for Atlas Training.

Detects training phases and triggers appropriate actions:
- Phase 1: Memory Learning - accuracy trending up OR loss decreasing
- Phase 2: Converged - accuracy plateau (low variance) at any level
- Phase 3: Overfitting - train acc ↑, val acc ↓ (gap growing)
- Phase 4: Grokking - val acc recovers, eff_dim stabilizes

Key differences from GrokkingDetector:
- GrokkingDetector: Focuses on internal geometric structure (embeddings, memory matrices)
- PhaseDetector: Focuses on training dynamics (train/val metrics, convergence, overfitting)

Usage:
    config = PhaseDetectorConfig()
    detector = PhaseDetector(config)

    for step in training_loop:
        metrics = {"train_accuracy": ..., "val_accuracy": ..., "loss": ...}
        phase = detector.update(step, metrics)
        if detector.phase_changed:
            handle_phase_transition(detector.current_phase, detector.previous_phase)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import Enum
import numpy as np
import json
from pathlib import Path


class TrainingPhase(Enum):
    """Training phases for phase detection."""
    UNKNOWN = "unknown"
    MEMORY_LEARNING = "memory_learning"
    CONVERGED = "converged"
    OVERFITTING = "overfitting"
    GROKKING = "grokking"


@dataclass
class PhaseDetectorConfig:
    """Configuration for phase detection."""
    # Window sizes for statistical analysis
    convergence_window: int = 2000      # Steps for plateau detection
    stability_window: int = 1000        # Steps for stability check
    overfitting_window: int = 500       # Steps for overfitting detection

    # Thresholds
    variance_threshold: float = 0.001   # Plateau = low variance in this range
    improvement_threshold: float = 0.001  # Minimum improvement to count as "learning"
    overfitting_gap_threshold: float = 0.05  # Train-val gap threshold
    overfitting_gap_growth_rate: float = 0.001  # Gap must grow at this rate

    # Effective dimension thresholds
    eff_dim_healthy_min: float = 0.30   # Below this = collapse warning
    eff_dim_stable_threshold: float = 0.02  # Max variance for "stable" eff_dim

    # Grokking detection
    grokking_recovery_threshold: float = 0.05  # Val acc must improve by this after overfitting
    grokking_eff_dim_stability: float = 0.01  # Eff dim variance threshold for grokking

    # Phase transition confirmation
    confirmation_steps: int = 100       # Steps to confirm phase transition
    min_steps_per_phase: int = 500      # Minimum steps before phase can change


@dataclass
class PhaseHistory:
    """Stores metric history for phase detection."""
    step: int
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    train_loss: float = float('inf')
    val_loss: float = float('inf')
    effective_dim_ratio: float = 0.0
    grad_weight_cosine: float = 0.0
    gate_mean: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "effective_dim_ratio": self.effective_dim_ratio,
            "grad_weight_cosine": self.grad_weight_cosine,
            "gate_mean": self.gate_mean,
        }


@dataclass
class PhaseTransition:
    """Records a phase transition event."""
    step: int
    from_phase: TrainingPhase
    to_phase: TrainingPhase
    metrics_snapshot: Dict[str, float]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "from_phase": self.from_phase.value,
            "to_phase": self.to_phase.value,
            "metrics_snapshot": self.metrics_snapshot,
            "reason": self.reason,
        }


class PhaseDetector:
    """
    Detects training phases based on train/val metrics and triggers phase transitions.

    Phases:
    1. MEMORY_LEARNING: Model is actively learning (loss decreasing or accuracy increasing)
    2. CONVERGED: Metrics have plateaued (low variance over window)
    3. OVERFITTING: Train accuracy increasing while val accuracy decreases
    4. GROKKING: Val accuracy recovers after overfitting, eff_dim stabilizes

    Triggers checkpoints at phase transitions for later analysis.
    """

    def __init__(self, config: PhaseDetectorConfig):
        self.config = config

        # Metric history
        self.history: deque = deque(maxlen=max(
            config.convergence_window,
            config.stability_window,
            config.overfitting_window
        ) + 100)

        # Phase state
        self._current_phase = TrainingPhase.UNKNOWN
        self._previous_phase = TrainingPhase.UNKNOWN
        self._phase_changed = False
        self._phase_start_step = 0
        self._pending_phase: Optional[TrainingPhase] = None
        self._pending_phase_start: int = 0

        # Transition history
        self.transitions: List[PhaseTransition] = []

        # Callbacks
        self._on_phase_change: List[Callable] = []

    @property
    def current_phase(self) -> TrainingPhase:
        """Get current training phase."""
        return self._current_phase

    @property
    def previous_phase(self) -> TrainingPhase:
        """Get previous training phase."""
        return self._previous_phase

    @property
    def phase_changed(self) -> bool:
        """Check if phase changed in last update."""
        return self._phase_changed

    def register_callback(self, callback: Callable[[PhaseTransition], None]) -> None:
        """Register a callback for phase transitions."""
        self._on_phase_change.append(callback)

    def update(self, step: int, metrics: Dict[str, float]) -> TrainingPhase:
        """
        Update phase detector with new metrics.

        Args:
            step: Current training step
            metrics: Dictionary with keys:
                - train_accuracy (or masked_word_accuracy)
                - val_accuracy (optional, for overfitting detection)
                - train_loss / val_loss
                - effective_dim_ratio
                - grad_weight_cosine (diagnostic)
                - gate_mean (diagnostic)

        Returns:
            Current training phase
        """
        self._phase_changed = False

        # Create history entry
        entry = PhaseHistory(
            step=step,
            train_accuracy=metrics.get('train_accuracy', metrics.get('masked_word_accuracy', 0.0)),
            val_accuracy=metrics.get('val_accuracy', 0.0),
            train_loss=metrics.get('train_loss', metrics.get('loss', float('inf'))),
            val_loss=metrics.get('val_loss', float('inf')),
            effective_dim_ratio=metrics.get('effective_dim_ratio',
                                           metrics.get('grokking/embedding_effective_dim_ratio', 0.0)),
            grad_weight_cosine=metrics.get('grad_weight_cosine', 0.0),
            gate_mean=metrics.get('gate_mean', metrics.get('grokking/gate_mean', 0.0)),
        )
        self.history.append(entry)

        # Need minimum history before detection
        if len(self.history) < self.config.confirmation_steps:
            return self._current_phase

        # Detect phase
        detected_phase = self._detect_phase()

        # Handle phase transition with confirmation
        self._handle_phase_transition(step, detected_phase, entry)

        return self._current_phase

    def _detect_phase(self) -> TrainingPhase:
        """
        Detect current phase based on metric trends.

        Priority order:
        1. GROKKING (if recovering from overfitting)
        2. OVERFITTING (if train/val gap growing)
        3. CONVERGED (if metrics stable)
        4. MEMORY_LEARNING (default if improving)
        """
        recent = list(self.history)

        # Extract metric arrays for analysis
        train_acc = np.array([h.train_accuracy for h in recent])
        val_acc = np.array([h.val_accuracy for h in recent])
        train_loss = np.array([h.train_loss for h in recent])
        eff_dim = np.array([h.effective_dim_ratio for h in recent])

        # Use appropriate window for each check
        conv_window = min(len(recent), self.config.convergence_window)
        stability_window = min(len(recent), self.config.stability_window)
        overfit_window = min(len(recent), self.config.overfitting_window)

        # Check for grokking (recovery after overfitting)
        if self._current_phase == TrainingPhase.OVERFITTING:
            if self._is_grokking(val_acc, eff_dim, stability_window):
                return TrainingPhase.GROKKING

        # Check for overfitting
        if self._is_overfitting(train_acc, val_acc, overfit_window):
            return TrainingPhase.OVERFITTING

        # Check for convergence (plateau)
        if self._is_converged(train_acc, train_loss, conv_window):
            return TrainingPhase.CONVERGED

        # Check for active learning
        if self._is_learning(train_acc, train_loss, stability_window):
            return TrainingPhase.MEMORY_LEARNING

        # Default to current phase if no clear signal
        return self._current_phase if self._current_phase != TrainingPhase.UNKNOWN else TrainingPhase.MEMORY_LEARNING

    def _is_learning(self, train_acc: np.ndarray, train_loss: np.ndarray, window: int) -> bool:
        """Check if model is actively learning."""
        if len(train_acc) < window:
            return True  # Assume learning if not enough data

        recent_acc = train_acc[-window:]
        recent_loss = train_loss[-window:]

        # Check for improvement trend
        acc_trend = self._compute_trend(recent_acc)
        loss_trend = self._compute_trend(recent_loss)

        # Learning if accuracy improving OR loss decreasing
        return acc_trend > self.config.improvement_threshold or loss_trend < -self.config.improvement_threshold

    def _is_converged(self, train_acc: np.ndarray, train_loss: np.ndarray, window: int) -> bool:
        """Check if model has converged (metrics plateaued)."""
        if len(train_acc) < window:
            return False

        recent_acc = train_acc[-window:]
        recent_loss = train_loss[-window:]

        # Check variance is below threshold
        acc_variance = np.var(recent_acc)
        loss_variance = np.var(recent_loss)

        # Converged if both accuracy and loss are stable
        return (acc_variance < self.config.variance_threshold and
                loss_variance < self.config.variance_threshold)

    def _is_overfitting(self, train_acc: np.ndarray, val_acc: np.ndarray, window: int) -> bool:
        """Check if model is overfitting."""
        if len(train_acc) < window or len(val_acc) < window:
            return False

        # Need validation data
        if np.all(val_acc[-window:] == 0):
            return False  # No validation data

        recent_train = train_acc[-window:]
        recent_val = val_acc[-window:]

        # Compute trends
        train_trend = self._compute_trend(recent_train)
        val_trend = self._compute_trend(recent_val)

        # Compute gap
        current_gap = recent_train[-1] - recent_val[-1]
        initial_gap = recent_train[0] - recent_val[0]
        gap_growth = current_gap - initial_gap

        # Overfitting if:
        # 1. Train accuracy increasing (or stable)
        # 2. Val accuracy decreasing
        # 3. Gap is growing
        return (train_trend >= -self.config.improvement_threshold and
                val_trend < -self.config.improvement_threshold and
                gap_growth > self.config.overfitting_gap_growth_rate * window)

    def _is_grokking(self, val_acc: np.ndarray, eff_dim: np.ndarray, window: int) -> bool:
        """Check if model is grokking (recovering from overfitting)."""
        if len(val_acc) < window or len(eff_dim) < window:
            return False

        recent_val = val_acc[-window:]
        recent_eff_dim = eff_dim[-window:]

        # Check for val accuracy recovery
        val_trend = self._compute_trend(recent_val)
        val_improvement = recent_val[-1] - recent_val[0]

        # Check for stable effective dimension
        eff_dim_variance = np.var(recent_eff_dim)

        # Grokking if:
        # 1. Val accuracy is recovering (improving)
        # 2. Eff dim is stabilizing
        return (val_improvement > self.config.grokking_recovery_threshold and
                val_trend > self.config.improvement_threshold and
                eff_dim_variance < self.config.grokking_eff_dim_stability)

    def _compute_trend(self, values: np.ndarray) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        try:
            x = np.arange(len(values))
            # Normalize x to prevent numerical issues
            x_norm = x / len(values)
            slope, _ = np.polyfit(x_norm, values, 1)
            # Rescale slope back
            return float(slope / len(values))
        except Exception:
            return 0.0

    def _handle_phase_transition(self, step: int, detected_phase: TrainingPhase, entry: PhaseHistory) -> None:
        """Handle phase transition with confirmation."""
        if detected_phase == self._current_phase:
            # Reset pending transition if phase matches current
            self._pending_phase = None
            return

        if detected_phase == self._pending_phase:
            # Check if we've confirmed the transition
            steps_in_pending = step - self._pending_phase_start
            if steps_in_pending >= self.config.confirmation_steps:
                # Also check minimum time in current phase
                steps_in_current = step - self._phase_start_step
                if steps_in_current >= self.config.min_steps_per_phase:
                    self._execute_phase_transition(step, detected_phase, entry)
        else:
            # Start new pending transition
            self._pending_phase = detected_phase
            self._pending_phase_start = step

    def _execute_phase_transition(self, step: int, new_phase: TrainingPhase, entry: PhaseHistory) -> None:
        """Execute a confirmed phase transition."""
        self._previous_phase = self._current_phase
        self._current_phase = new_phase
        self._phase_changed = True
        self._phase_start_step = step
        self._pending_phase = None

        # Determine transition reason
        reason = self._get_transition_reason(self._previous_phase, new_phase)

        # Create transition record
        transition = PhaseTransition(
            step=step,
            from_phase=self._previous_phase,
            to_phase=new_phase,
            metrics_snapshot=entry.to_dict(),
            reason=reason,
        )
        self.transitions.append(transition)

        # Call callbacks
        for callback in self._on_phase_change:
            try:
                callback(transition)
            except Exception as e:
                print(f"PhaseDetector callback error: {e}")

    def _get_transition_reason(self, from_phase: TrainingPhase, to_phase: TrainingPhase) -> str:
        """Generate human-readable transition reason."""
        reasons = {
            (TrainingPhase.UNKNOWN, TrainingPhase.MEMORY_LEARNING): "Training started",
            (TrainingPhase.MEMORY_LEARNING, TrainingPhase.CONVERGED): "Metrics plateaued (low variance)",
            (TrainingPhase.MEMORY_LEARNING, TrainingPhase.OVERFITTING): "Train/val gap growing",
            (TrainingPhase.CONVERGED, TrainingPhase.OVERFITTING): "Train improving, val declining",
            (TrainingPhase.CONVERGED, TrainingPhase.MEMORY_LEARNING): "Learning resumed",
            (TrainingPhase.OVERFITTING, TrainingPhase.GROKKING): "Val accuracy recovering, eff_dim stable",
            (TrainingPhase.OVERFITTING, TrainingPhase.CONVERGED): "Metrics stabilized",
            (TrainingPhase.GROKKING, TrainingPhase.CONVERGED): "Grokking complete",
        }
        return reasons.get((from_phase, to_phase), f"Transition: {from_phase.value} -> {to_phase.value}")

    def get_checkpoint_name(self, step: int) -> str:
        """Generate checkpoint name for current phase."""
        phase = self._current_phase.value

        # Get relevant metrics
        if self.history:
            latest = self.history[-1]
            acc = latest.train_accuracy
            eff_dim = latest.effective_dim_ratio
            return f"{phase}_step_{step}_acc_{acc:.3f}_effdim_{eff_dim:.3f}.pt"

        return f"{phase}_step_{step}.pt"

    def should_checkpoint(self) -> bool:
        """Check if we should save a checkpoint (phase just changed)."""
        return self._phase_changed

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of phase detection state."""
        summary = {
            "current_phase": self._current_phase.value,
            "previous_phase": self._previous_phase.value,
            "phase_start_step": self._phase_start_step,
            "history_length": len(self.history),
            "transitions_count": len(self.transitions),
        }

        if self.history:
            latest = self.history[-1]
            summary["latest_metrics"] = latest.to_dict()

        if self.transitions:
            summary["last_transition"] = self.transitions[-1].to_dict()

        return summary

    def save_transitions(self, path: Path) -> None:
        """Save all phase transitions to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "current_phase": self._current_phase.value,
            "transitions": [t.to_dict() for t in self.transitions],
            "summary": self.get_summary(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def create_phase_detector(config_dict: Dict) -> PhaseDetector:
    """Create PhaseDetector from config dictionary."""
    phase_config = config_dict.get("monitoring", {}).get("phase_detection", {})

    config = PhaseDetectorConfig(
        convergence_window=phase_config.get("convergence_window", 2000),
        stability_window=phase_config.get("stability_window", 1000),
        overfitting_window=phase_config.get("overfitting_window", 500),
        variance_threshold=phase_config.get("variance_threshold", 0.001),
        improvement_threshold=phase_config.get("improvement_threshold", 0.001),
        overfitting_gap_threshold=phase_config.get("overfitting_gap_threshold", 0.05),
        eff_dim_healthy_min=phase_config.get("eff_dim_healthy_min", 0.30),
    )

    return PhaseDetector(config)

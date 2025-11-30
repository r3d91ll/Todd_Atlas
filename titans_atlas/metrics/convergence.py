"""
Convergence Metrics for Atlas Training.

Provides comprehensive tracking of training convergence including:
- Loss smoothing and trend analysis
- Perplexity tracking
- Gradient statistics
- Learning rate correlation
- Early stopping signals
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import numpy as np
from collections import deque


@dataclass
class ConvergenceState:
    """Current convergence state."""

    # Loss tracking
    current_loss: float = float("inf")
    best_loss: float = float("inf")
    smoothed_loss: float = float("inf")
    loss_trend: float = 0.0  # Negative = decreasing (good)

    # Perplexity
    perplexity: float = float("inf")
    smoothed_perplexity: float = float("inf")

    # Gradient health
    grad_norm: float = 0.0
    grad_norm_smoothed: float = 0.0
    grad_variance: float = 0.0

    # Progress indicators
    steps_since_improvement: int = 0
    convergence_score: float = 0.0  # 0-1, higher = more converged
    is_converging: bool = False
    is_plateaued: bool = False

    # Learning dynamics
    learning_rate: float = 0.0
    effective_lr: float = 0.0  # LR * grad_norm approximation


@dataclass
class LossWindow:
    """Sliding window for loss statistics."""

    window_size: int = 100
    values: deque = field(default_factory=lambda: deque(maxlen=100))

    def add(self, value: float):
        self.values.append(value)

    def mean(self) -> float:
        if not self.values:
            return float("inf")
        return np.mean(list(self.values))

    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return np.std(list(self.values))

    def trend(self) -> float:
        """Compute trend (slope of linear fit). Negative = decreasing."""
        if len(self.values) < 10:
            return 0.0

        y = np.array(list(self.values))
        x = np.arange(len(y))

        # Simple linear regression
        x_mean = x.mean()
        y_mean = y.mean()

        numerator = ((x - x_mean) * (y - y_mean)).sum()
        denominator = ((x - x_mean) ** 2).sum()

        if abs(denominator) < 1e-10:
            return 0.0

        return numerator / denominator

    def min(self) -> float:
        if not self.values:
            return float("inf")
        return min(self.values)

    def is_plateau(self, threshold: float = 0.01) -> bool:
        """Check if loss has plateaued (low variance, near-zero trend)."""
        if len(self.values) < 50:
            return False

        recent = list(self.values)[-50:]
        variance = np.var(recent)
        trend = abs(self.trend())

        return variance < threshold and trend < threshold


class ConvergenceMetrics:
    """
    Track and analyze training convergence.

    Usage:
        metrics = ConvergenceMetrics(output_dir="./runs/exp1")

        for step in range(max_steps):
            loss = train_step()
            grad_norm = get_grad_norm()

            state = metrics.update(
                step=step,
                loss=loss,
                grad_norm=grad_norm,
                learning_rate=lr
            )

            if state.is_converging:
                print("Model is converging!")

            if state.is_plateaued:
                print("Consider stopping or adjusting LR")

            # Periodic logging
            if step % 100 == 0:
                metrics.log_to_jsonl()
    """

    def __init__(
        self,
        output_dir: str = "./convergence_metrics",
        ema_alpha: float = 0.1,  # EMA smoothing factor
        window_size: int = 100,  # Rolling window size
        patience: int = 500,  # Steps without improvement before plateau
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ema_alpha = ema_alpha
        self.patience = patience

        # State
        self.state = ConvergenceState()

        # Sliding windows
        self.loss_window = LossWindow(window_size=window_size)
        self.grad_window = LossWindow(window_size=window_size)
        self.ppl_window = LossWindow(window_size=window_size)

        # History for plotting
        self.history: List[Dict[str, Any]] = []
        self.step_count = 0

        # Best checkpointing
        self.best_step = 0

    def update(
        self,
        step: int,
        loss: float,
        grad_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
        tokens_seen: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> ConvergenceState:
        """
        Update convergence metrics with new training step data.

        Args:
            step: Current training step
            loss: Training loss (cross-entropy)
            grad_norm: Global gradient norm (optional)
            learning_rate: Current learning rate (optional)
            tokens_seen: Total tokens processed (optional)
            batch_size: Current batch size (optional)

        Returns:
            Updated ConvergenceState
        """
        self.step_count = step

        # Update loss tracking
        self.state.current_loss = loss
        self.loss_window.add(loss)

        # EMA smoothing
        if self.state.smoothed_loss == float("inf"):
            self.state.smoothed_loss = loss
        else:
            self.state.smoothed_loss = (
                self.ema_alpha * loss +
                (1 - self.ema_alpha) * self.state.smoothed_loss
            )

        # Best loss tracking
        if loss < self.state.best_loss:
            self.state.best_loss = loss
            self.state.steps_since_improvement = 0
            self.best_step = step
        else:
            self.state.steps_since_improvement += 1

        # Trend analysis
        self.state.loss_trend = self.loss_window.trend()

        # Perplexity (exp of cross-entropy loss)
        self.state.perplexity = min(np.exp(loss), 1e10)  # Cap at 10B
        self.ppl_window.add(self.state.perplexity)

        if self.state.smoothed_perplexity == float("inf"):
            self.state.smoothed_perplexity = self.state.perplexity
        else:
            self.state.smoothed_perplexity = (
                self.ema_alpha * self.state.perplexity +
                (1 - self.ema_alpha) * self.state.smoothed_perplexity
            )

        # Gradient statistics
        if grad_norm is not None:
            self.state.grad_norm = grad_norm
            self.grad_window.add(grad_norm)

            if self.state.grad_norm_smoothed == 0:
                self.state.grad_norm_smoothed = grad_norm
            else:
                self.state.grad_norm_smoothed = (
                    self.ema_alpha * grad_norm +
                    (1 - self.ema_alpha) * self.state.grad_norm_smoothed
                )

            self.state.grad_variance = self.grad_window.std()

        # Learning rate
        if learning_rate is not None:
            self.state.learning_rate = learning_rate
            if grad_norm is not None:
                self.state.effective_lr = learning_rate * grad_norm

        # Convergence scoring (0-1 scale)
        self._compute_convergence_score()

        # Plateau detection
        self.state.is_plateaued = (
            self.state.steps_since_improvement > self.patience or
            self.loss_window.is_plateau()
        )

        # Convergence detection (improving and stable)
        self.state.is_converging = (
            self.state.loss_trend < 0 and
            abs(self.state.loss_trend) > 1e-4 and
            not self.state.is_plateaued
        )

        # Record history
        record = {
            "step": step,
            "timestamp": time.time(),
            "loss": loss,
            "smoothed_loss": self.state.smoothed_loss,
            "best_loss": self.state.best_loss,
            "loss_trend": self.state.loss_trend,
            "perplexity": self.state.perplexity,
            "smoothed_perplexity": self.state.smoothed_perplexity,
            "grad_norm": self.state.grad_norm,
            "grad_norm_smoothed": self.state.grad_norm_smoothed,
            "grad_variance": self.state.grad_variance,
            "learning_rate": self.state.learning_rate,
            "convergence_score": self.state.convergence_score,
            "is_converging": self.state.is_converging,
            "is_plateaued": self.state.is_plateaued,
            "steps_since_improvement": self.state.steps_since_improvement,
        }

        if tokens_seen is not None:
            record["tokens_seen"] = tokens_seen
        if batch_size is not None:
            record["batch_size"] = batch_size

        self.history.append(record)

        return self.state

    def _compute_convergence_score(self):
        """
        Compute a 0-1 convergence score based on multiple factors.

        Higher score = more converged. Factors:
        - Loss stability (low variance)
        - Trend magnitude (small = stable)
        - Distance from theoretical minimum (estimated)
        - Gradient stability
        """
        scores = []

        # Loss stability score (based on coefficient of variation)
        if len(self.loss_window.values) > 10:
            mean_loss = self.loss_window.mean()
            std_loss = self.loss_window.std()
            if mean_loss > 0:
                cv = std_loss / mean_loss
                stability_score = max(0, 1 - cv)
                scores.append(stability_score)

        # Trend magnitude score (smaller trend = better)
        trend_magnitude = abs(self.state.loss_trend)
        trend_score = max(0, 1 - trend_magnitude * 10)  # Scale factor
        scores.append(trend_score)

        # Gradient stability score
        if self.state.grad_variance > 0:
            grad_cv = self.state.grad_variance / max(self.state.grad_norm_smoothed, 1e-8)
            grad_score = max(0, 1 - grad_cv)
            scores.append(grad_score)

        # Improvement recency score
        if self.patience > 0:
            recency_score = max(0, 1 - self.state.steps_since_improvement / self.patience)
            scores.append(recency_score)

        # Aggregate
        if scores:
            self.state.convergence_score = np.mean(scores)
        else:
            self.state.convergence_score = 0.0

    def log_to_jsonl(self, filepath: str = None) -> str:
        """Write metrics to JSONL file for dashboard consumption."""
        if filepath is None:
            filepath = self.output_dir / "metrics.jsonl"

        if not self.history:
            return str(filepath)

        # Write only recent entries (not the entire history each time)
        recent = self.history[-1] if self.history else {}

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(filepath, "a") as f:
            f.write(json.dumps(convert_types(recent)) + "\n")

        return str(filepath)

    def save_full_history(self, filepath: str = None):
        """Save complete history to JSON file."""
        if filepath is None:
            filepath = self.output_dir / "full_history.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(filepath, "w") as f:
            json.dump(convert_types(self.history), f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.history:
            return {}

        losses = [h["loss"] for h in self.history]
        ppls = [h["perplexity"] for h in self.history if h["perplexity"] < 1e9]

        return {
            "total_steps": len(self.history),
            "current_loss": self.state.current_loss,
            "best_loss": self.state.best_loss,
            "best_step": self.best_step,
            "final_smoothed_loss": self.state.smoothed_loss,
            "loss_reduction": losses[0] - losses[-1] if len(losses) > 1 else 0,
            "loss_reduction_pct": (1 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0,
            "current_perplexity": self.state.perplexity,
            "best_perplexity": min(ppls) if ppls else float("inf"),
            "convergence_score": self.state.convergence_score,
            "is_converging": self.state.is_converging,
            "is_plateaued": self.state.is_plateaued,
            "steps_since_improvement": self.state.steps_since_improvement,
        }

    def should_stop_early(
        self,
        loss_threshold: Optional[float] = None,
        perplexity_threshold: Optional[float] = None,
        convergence_threshold: float = 0.95,
    ) -> Tuple[bool, str]:
        """
        Check if training should stop early.

        Args:
            loss_threshold: Stop if loss below this
            perplexity_threshold: Stop if perplexity below this
            convergence_threshold: Stop if convergence score above this

        Returns:
            (should_stop, reason)
        """
        if loss_threshold is not None and self.state.smoothed_loss < loss_threshold:
            return True, f"Loss target reached: {self.state.smoothed_loss:.4f} < {loss_threshold}"

        if perplexity_threshold is not None and self.state.smoothed_perplexity < perplexity_threshold:
            return True, f"Perplexity target reached: {self.state.smoothed_perplexity:.2f} < {perplexity_threshold}"

        if self.state.convergence_score > convergence_threshold:
            return True, f"Convergence threshold reached: {self.state.convergence_score:.3f} > {convergence_threshold}"

        if self.state.is_plateaued and self.state.steps_since_improvement > self.patience * 2:
            return True, f"Severe plateau: {self.state.steps_since_improvement} steps without improvement"

        return False, ""

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        return {
            "step": self.step_count,
            "loss": {
                "current": self.state.current_loss,
                "smoothed": self.state.smoothed_loss,
                "best": self.state.best_loss,
                "trend": self.state.loss_trend,
            },
            "perplexity": {
                "current": self.state.perplexity,
                "smoothed": self.state.smoothed_perplexity,
            },
            "gradients": {
                "norm": self.state.grad_norm,
                "smoothed": self.state.grad_norm_smoothed,
                "variance": self.state.grad_variance,
            },
            "convergence": {
                "score": self.state.convergence_score,
                "is_converging": self.state.is_converging,
                "is_plateaued": self.state.is_plateaued,
                "steps_since_improvement": self.state.steps_since_improvement,
            },
            "learning_rate": self.state.learning_rate,
        }


class GradientMonitor:
    """
    Detailed gradient monitoring for debugging training issues.

    Tracks per-layer gradient statistics, detects vanishing/exploding
    gradients, and provides early warning for training instabilities.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layer_norms: Dict[str, List[float]] = {}
        self.layer_stats: Dict[str, Dict[str, float]] = {}

    def compute_gradient_stats(self) -> Dict[str, Any]:
        """Compute gradient statistics for all parameters."""
        stats = {
            "global_norm": 0.0,
            "per_layer": {},
            "warnings": [],
        }

        total_norm = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach()
                grad_norm = grad.norm().item()
                total_norm += grad_norm ** 2

                layer_stats = {
                    "norm": grad_norm,
                    "mean": grad.mean().item(),
                    "std": grad.std().item(),
                    "max": grad.abs().max().item(),
                    "min": grad.abs().min().item(),
                    "zero_fraction": (grad.abs() < 1e-8).float().mean().item(),
                }

                # Track history
                if name not in self.layer_norms:
                    self.layer_norms[name] = []
                self.layer_norms[name].append(grad_norm)
                if len(self.layer_norms[name]) > 100:
                    self.layer_norms[name].pop(0)

                stats["per_layer"][name] = layer_stats

                # Warnings
                if grad_norm < 1e-7:
                    stats["warnings"].append(f"Vanishing gradient in {name}: {grad_norm:.2e}")
                elif grad_norm > 100:
                    stats["warnings"].append(f"Exploding gradient in {name}: {grad_norm:.2e}")

                if layer_stats["zero_fraction"] > 0.9:
                    stats["warnings"].append(f"Sparse gradient in {name}: {layer_stats['zero_fraction']:.1%} zeros")

        stats["global_norm"] = np.sqrt(total_norm)
        return stats

    def get_layer_norm_trends(self) -> Dict[str, Dict[str, float]]:
        """Get gradient norm trends per layer."""
        trends = {}
        for name, norms in self.layer_norms.items():
            if len(norms) > 10:
                recent = norms[-10:]
                trends[name] = {
                    "mean": np.mean(recent),
                    "std": np.std(recent),
                    "trend": (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0,
                }
        return trends

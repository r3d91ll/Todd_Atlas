"""
Dynamic Weight Decay Scheduler for Grokking.

Automatically adjusts weight decay based on excluded_loss trajectory.
When excluded_loss plateaus (not rising), it indicates the model is stuck
in memorization and needs more regularization to push toward grokking.

Key concepts:
- excluded_loss rising = circuits forming (grokking imminent) = weight decay is working
- excluded_loss flat = stuck in memorization = increase weight decay
- Weight decay is critical for grokking (recommended default: 1.0, not 0.1)

References:
- Power et al. (2022): Weight decay is critical hyperparameter for grokking
- Nanda et al. (2023): Circuit formation correlates with regularization strength
"""

import logging
from dataclasses import dataclass
from typing import Optional, List
from collections import deque

import torch

logger = logging.getLogger(__name__)


@dataclass
class WeightDecayConfig:
    """Configuration for dynamic weight decay scheduling."""
    enabled: bool = True
    initial: float = 1.0  # Starting weight decay (1.0 recommended for grokking)
    min_value: float = 0.1  # Minimum allowed weight decay
    max_value: float = 10.0  # Maximum allowed weight decay
    adjustment_factor: float = 1.5  # Multiply by this when adjusting
    patience_steps: int = 1000  # Steps without progress before adjustment
    excluded_loss_threshold: float = 0.01  # Minimum change to consider "progress"
    cooldown_steps: int = 500  # Steps to wait after adjustment before next check


class DynamicWeightDecayScheduler:
    """
    Dynamically adjusts weight decay based on excluded_loss trajectory.

    The scheduler monitors excluded_loss and increases weight decay when
    the model appears stuck in memorization (flat excluded_loss).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: WeightDecayConfig,
    ):
        """
        Initialize the scheduler.

        Args:
            optimizer: PyTorch optimizer with weight_decay parameter
            config: Configuration for the scheduler
        """
        self.optimizer = optimizer
        self.config = config

        self.current_weight_decay = config.initial
        self.excluded_loss_history: deque = deque(maxlen=config.patience_steps * 2)
        self.steps_without_progress = 0
        self.cooldown_remaining = 0
        self.adjustment_count = 0

        # Apply initial weight decay
        self._set_weight_decay(config.initial)

        logger.info(
            f"DynamicWeightDecayScheduler initialized: "
            f"initial={config.initial}, max={config.max_value}, "
            f"patience={config.patience_steps}"
        )

    def step(self, excluded_loss: float) -> bool:
        """
        Update weight decay based on excluded_loss trajectory.

        Args:
            excluded_loss: Current excluded_loss value from grokking detector

        Returns:
            True if weight decay was adjusted, False otherwise
        """
        if not self.config.enabled:
            return False

        self.excluded_loss_history.append(excluded_loss)

        # Handle cooldown
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False

        # Need enough history to compute progress
        if len(self.excluded_loss_history) < 2:
            return False

        # Check if excluded_loss is making progress (rising = good for grokking)
        recent_change = self.excluded_loss_history[-1] - self.excluded_loss_history[-2]

        if recent_change > self.config.excluded_loss_threshold:
            # Progress detected - excluded_loss is rising (circuits forming)
            self.steps_without_progress = 0
            return False
        else:
            # No progress - excluded_loss is flat or falling
            self.steps_without_progress += 1

        # Check if we've exceeded patience
        if self.steps_without_progress >= self.config.patience_steps:
            # Attempt to increase weight decay
            new_wd = min(
                self.current_weight_decay * self.config.adjustment_factor,
                self.config.max_value
            )

            if new_wd != self.current_weight_decay:
                old_wd = self.current_weight_decay
                self._set_weight_decay(new_wd)
                self.steps_without_progress = 0
                self.cooldown_remaining = self.config.cooldown_steps
                self.adjustment_count += 1

                logger.info(
                    f"Weight decay adjusted: {old_wd:.4f} -> {new_wd:.4f} "
                    f"(adjustment #{self.adjustment_count})"
                )
                return True
            else:
                # Already at max - reset counter to avoid spam
                self.steps_without_progress = 0
                logger.debug(f"Weight decay already at max ({self.config.max_value})")

        return False

    def _set_weight_decay(self, weight_decay: float) -> None:
        """Set weight decay on optimizer parameter groups."""
        self.current_weight_decay = weight_decay

        for param_group in self.optimizer.param_groups:
            # Only update groups that already have weight decay (respect no_decay patterns)
            if param_group.get("weight_decay", 0) > 0:
                param_group["weight_decay"] = weight_decay

    def get_current_weight_decay(self) -> float:
        """Return current weight decay value."""
        return self.current_weight_decay

    def get_stats(self) -> dict:
        """Get scheduler statistics for logging."""
        return {
            "weight_decay/current": self.current_weight_decay,
            "weight_decay/steps_without_progress": self.steps_without_progress,
            "weight_decay/adjustment_count": self.adjustment_count,
            "weight_decay/cooldown_remaining": self.cooldown_remaining,
        }

    def state_dict(self) -> dict:
        """Get state for checkpointing."""
        return {
            "current_weight_decay": self.current_weight_decay,
            "excluded_loss_history": list(self.excluded_loss_history),
            "steps_without_progress": self.steps_without_progress,
            "cooldown_remaining": self.cooldown_remaining,
            "adjustment_count": self.adjustment_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.current_weight_decay = state["current_weight_decay"]
        self.excluded_loss_history = deque(
            state["excluded_loss_history"],
            maxlen=self.config.patience_steps * 2
        )
        self.steps_without_progress = state["steps_without_progress"]
        self.cooldown_remaining = state["cooldown_remaining"]
        self.adjustment_count = state["adjustment_count"]

        # Apply the weight decay
        self._set_weight_decay(self.current_weight_decay)


def create_weight_decay_scheduler(
    optimizer: torch.optim.Optimizer,
    config_dict: dict,
) -> Optional[DynamicWeightDecayScheduler]:
    """
    Create weight decay scheduler from config dictionary.

    Args:
        optimizer: PyTorch optimizer
        config_dict: Full config dictionary

    Returns:
        DynamicWeightDecayScheduler if enabled, None otherwise
    """
    wd_config = config_dict.get("dynamic_weight_decay", {})

    if not wd_config.get("enabled", False):
        return None

    config = WeightDecayConfig(
        enabled=True,
        initial=wd_config.get("initial", 1.0),
        min_value=wd_config.get("min_value", 0.1),
        max_value=wd_config.get("max_value", 10.0),
        adjustment_factor=wd_config.get("adjustment_factor", 1.5),
        patience_steps=wd_config.get("patience_steps", 1000),
        excluded_loss_threshold=wd_config.get("excluded_loss_threshold", 0.01),
        cooldown_steps=wd_config.get("cooldown_steps", 500),
    )

    return DynamicWeightDecayScheduler(optimizer, config)

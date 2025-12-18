"""
Base Metrics Adapter - Abstract interface for experiment-specific metrics.

Each experiment implements its own adapter by subclassing MetricsAdapter.
This is the ONLY file that needs to change when adding a new experiment type.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import torch.nn as nn


class MetricsAdapter(ABC):
    """
    Abstract base class for experiment-specific metrics collection.

    Subclasses must implement:
    - collect(model, outputs) -> Dict[str, Any]
    - get_alert_thresholds() -> Dict[str, Tuple[float, str]]

    Example implementation for a new experiment:

    ```python
    class MyExperimentAdapter(MetricsAdapter):
        def collect(self, model, outputs):
            return {
                'custom_metric': compute_custom_metric(model),
                'another_metric': outputs.get('some_value', 0.0),
            }

        def get_alert_thresholds(self):
            return {
                'custom_metric': (0.5, 'warning'),  # Alert if > 0.5
                'another_metric': (0.1, 'warning_below'),  # Alert if < 0.1
            }
    ```
    """

    @abstractmethod
    def collect(self, model: nn.Module, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect experiment-specific metrics.

        Args:
            model: The model being trained
            outputs: Outputs from the current training step

        Returns:
            Dictionary of metric_name -> value
        """
        pass

    @abstractmethod
    def get_alert_thresholds(self) -> Dict[str, Tuple[float, str]]:
        """
        Return alert thresholds for metrics.

        Returns:
            Dictionary of metric_name -> (threshold_value, severity_string)

            Severity options:
            - 'info': Informational
            - 'warning': Something may be wrong
            - 'critical': Immediate attention needed
            - Add '_below' suffix for below-threshold alerts
              (e.g., 'warning_below' alerts when value < threshold)
        """
        pass

    def on_training_start(self, model: nn.Module, config: Any) -> None:
        """
        Hook called when training starts.

        Override for setup that requires model/config access.
        """
        pass

    def on_training_end(self, model: nn.Module, stats: Dict[str, Any]) -> None:
        """
        Hook called when training ends.

        Override for cleanup or final analysis.
        """
        pass

    def on_checkpoint(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """
        Hook called when checkpoint is saved.

        Override to add checkpoint-specific analysis.

        Returns:
            Additional metrics to log with checkpoint
        """
        return {}


class DefaultAdapter(MetricsAdapter):
    """
    Default adapter that collects basic metrics.

    Use this as a starting point or for experiments without custom metrics.
    """

    def collect(self, model: nn.Module, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collect basic metrics available from any model."""
        metrics = {}

        # Extract loss if available
        if isinstance(outputs, dict):
            if 'loss' in outputs:
                metrics['loss'] = outputs['loss'].item() if hasattr(outputs['loss'], 'item') else outputs['loss']

        # Count parameters
        if not hasattr(self, '_param_count'):
            self._param_count = sum(p.numel() for p in model.parameters())
        metrics['param_count'] = self._param_count

        # Gradient norm (if gradients exist)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        metrics['grad_norm'] = total_norm ** 0.5

        return metrics

    def get_alert_thresholds(self) -> Dict[str, Tuple[float, str]]:
        """Default thresholds."""
        return {
            'loss': (10.0, 'critical'),  # Very high loss
            'grad_norm': (100.0, 'warning'),  # Gradient explosion
        }

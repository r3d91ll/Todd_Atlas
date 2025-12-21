"""
Base Trainer - Abstract trainer class with pluggable metrics adapter.

Provides the foundation for experiment-specific trainers with:
- Automatic metrics collection
- Alert system integration
- Checkpoint management
- Configurable training loops
"""

import time
import signal
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metric_collector import MetricCollector, MetricCollectorConfig

if TYPE_CHECKING:
    from ..monitoring.alert_system import AlertSystem
    from ..adapters.base_adapter import MetricsAdapter


@dataclass
class TrainerConfig:
    """Configuration for base trainer."""
    # Training parameters
    max_steps: int = 100000
    gradient_accumulation: int = 1
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000

    # Paths
    output_dir: str = "runs/experiment"
    metrics_stream: str = "metrics_stream.jsonl"

    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    dtype: str = "bfloat16"

    # Checkpointing
    keep_last_n: int = 5
    save_optimizer: bool = True

    # Monitoring
    enable_alerts: bool = True
    enable_dashboard: bool = True


class BaseTrainer(ABC):
    """
    Abstract base trainer with modular metrics collection.

    Subclasses must implement:
    - train_step(batch) -> Dict[str, Any]
    - optionally: evaluate() -> Dict[str, Any]
    """

    def __init__(
        self,
        model: nn.Module,
        metrics_adapter: 'MetricsAdapter',
        config: TrainerConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        alert_system: Optional['AlertSystem'] = None,
    ):
        self.model = model
        self.metrics_adapter = metrics_adapter
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.alert_system = alert_system

        # Setup paths
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Setup metrics collector
        metrics_config = MetricCollectorConfig(
            stream_path=str(self.output_dir / config.metrics_stream)
        )
        self.metric_collector = MetricCollector(metrics_config)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self._start_time = None
        self._stop_requested = False

        # Setup logging
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup device
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup handlers for graceful shutdown."""
        def handler(signum, frame):
            self.logger.warning(f"Received signal {signum}, stopping training...")
            self._stop_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    @abstractmethod
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Execute a single training step.

        Args:
            batch: Input batch

        Returns:
            Dictionary of metrics from this step
        """
        pass

    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on validation set.

        Override for custom evaluation logic.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(batch['input_ids'])
                loss = outputs.get('loss', outputs) if isinstance(outputs, dict) else outputs
                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 100:  # Limit eval batches
                    break

        self.model.train()
        return {'val_loss': total_loss / max(num_batches, 1)}

    def collect_metrics(self, step_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and log metrics from training step.

        Combines step metrics with adapter-specific metrics.
        """
        # Base metrics
        metrics = {
            'step': self.global_step,
            'epoch': self.epoch,
            'timestamp': time.time(),
            'elapsed_hours': self._get_elapsed_hours(),
        }

        # Add step metrics
        metrics.update(step_metrics)

        # Add adapter-specific metrics (experiment-specific)
        try:
            adapter_metrics = self.metrics_adapter.collect(self.model, step_metrics)
            metrics.update(adapter_metrics)
        except Exception as e:
            self.logger.warning(f"Adapter metrics collection failed: {e}")

        # Log to collector
        self.metric_collector.log(metrics)

        # Check alert thresholds
        if self.alert_system and self.config.enable_alerts:
            self._check_alerts(metrics)

        return metrics

    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against alert thresholds."""
        try:
            thresholds = self.metrics_adapter.get_alert_thresholds()
            for metric_name, (threshold, severity) in thresholds.items():
                value = metrics.get(metric_name)
                if value is not None:
                    # Handle both "above" and "below" thresholds
                    if severity.endswith('_below'):
                        severity = severity[:-6]
                        if value < threshold:
                            self.alert_system.send_alert(
                                f"{metric_name} = {value:.4f} (below {threshold})",
                                severity=severity
                            )
                    else:
                        if value > threshold:
                            self.alert_system.send_alert(
                                f"{metric_name} = {value:.4f} (above {threshold})",
                                severity=severity
                            )
        except Exception as e:
            self.logger.warning(f"Alert check failed: {e}")

    def _get_elapsed_hours(self) -> float:
        """Get elapsed training time in hours."""
        if self._start_time is None:
            return 0.0
        return (time.time() - self._start_time) / 3600

    def save_checkpoint(self, suffix: str = "") -> Path:
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_{self.global_step}{suffix}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
        }

        if self.config.save_optimizer and self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_checkpoints()

        return checkpoint_path

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the latest N."""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for old_checkpoint in checkpoints[self.config.keep_last_n:]:
            old_checkpoint.unlink()
            self.logger.debug(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint."""
        # weights_only=False needed for PyTorch 2.6+ (changed default)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint.get('epoch', 0)

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Loaded checkpoint from step {self.global_step}")

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        Returns:
            Final training statistics
        """
        self.logger.info(f"Starting training for {self.config.max_steps} steps")
        self._start_time = time.time()

        # Notify training start
        if self.alert_system:
            self.alert_system.send_training_start(
                experiment_name=self.output_dir.name,
                config_summary=f"Steps: {self.config.max_steps}"
            )

        self.model.train()
        data_iter = iter(self.train_loader)

        while self.global_step < self.config.max_steps and not self._stop_requested:
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_loader)
                batch = next(data_iter)

            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Training step
            step_metrics = self.train_step(batch)
            self.global_step += 1

            # Collect and log metrics
            if self.global_step % self.config.log_interval == 0:
                metrics = self.collect_metrics(step_metrics)
                self._log_progress(metrics)

            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                self.metric_collector.log({
                    'step': self.global_step,
                    'eval': True,
                    **eval_metrics
                })

            # Checkpoint
            if self.global_step % self.config.save_interval == 0:
                checkpoint_path = self.save_checkpoint()
                if self.alert_system:
                    gate_mean = step_metrics.get('gate_mean', 0.0)
                    self.alert_system.send_checkpoint(
                        self.global_step,
                        step_metrics.get('loss', 0.0),
                        gate_mean
                    )

        # Final save and notification
        self.save_checkpoint(suffix="_final")
        self.metric_collector.flush()

        final_stats = {
            'total_steps': self.global_step,
            'total_epochs': self.epoch,
            'elapsed_hours': self._get_elapsed_hours(),
            'final_loss': step_metrics.get('loss', 0.0),
        }

        if self.alert_system:
            self.alert_system.send_training_complete(
                self.global_step,
                final_stats['final_loss'],
                step_metrics.get('gate_mean', 0.0)
            )

        return final_stats

    def _log_progress(self, metrics: Dict[str, Any]) -> None:
        """Log training progress."""
        loss = metrics.get('loss', 0.0)
        elapsed = metrics.get('elapsed_hours', 0.0)
        steps_remaining = self.config.max_steps - self.global_step
        steps_per_hour = self.global_step / max(elapsed, 0.001)
        eta_hours = steps_remaining / max(steps_per_hour, 1)

        self.logger.info(
            f"Step {self.global_step:,}/{self.config.max_steps:,} | "
            f"Loss: {loss:.4f} | "
            f"Elapsed: {elapsed:.2f}h | "
            f"ETA: {eta_hours:.2f}h"
        )

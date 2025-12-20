"""Core training framework components."""

from .metric_collector import MetricCollector
from .base_trainer import BaseTrainer, TrainerConfig

__all__ = ['MetricCollector', 'BaseTrainer', 'TrainerConfig']

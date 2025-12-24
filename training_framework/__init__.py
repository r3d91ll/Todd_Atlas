"""
Training Framework - Modular monitoring and training infrastructure.

This framework provides reusable components for:
- Metrics collection and streaming
- Real-time dashboard monitoring (Streamlit)
- Telegram alerting with configurable thresholds
- Pluggable experiment-specific adapters

Usage:
    from training_framework import MetricCollector, AlertSystem
    from training_framework.adapters import AtlasMetricsAdapter
"""

from .core.metric_collector import MetricCollector
from .core.base_trainer import BaseTrainer, TrainerConfig
from .monitoring.alert_system import AlertSystem, TelegramNotifier, TelegramConfig
from .phase_detector import PhaseDetector, PhaseDetectorConfig, TrainingPhase, create_phase_detector

__all__ = [
    'MetricCollector',
    'BaseTrainer',
    'TrainerConfig',
    'AlertSystem',
    'TelegramNotifier',
    'TelegramConfig',
    'PhaseDetector',
    'PhaseDetectorConfig',
    'TrainingPhase',
    'create_phase_detector',
]

__version__ = '0.1.0'

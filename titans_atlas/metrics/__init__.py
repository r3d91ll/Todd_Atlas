"""
Metrics and measurement modules for Atlas training.

Includes:
- Weaver space measurements (geometric dynamics in compute space)
- Convergence metrics
- Training progress tracking
"""

from titans_atlas.metrics.weaver_space import (
    WeaverSpaceMetrics,
    MemoryTrajectoryCapture,
    AttentionGeometryTracker,
    ManifoldEvolutionMonitor,
    OmegaQualityMeasure,
)
from titans_atlas.metrics.convergence import ConvergenceMetrics

__all__ = [
    "WeaverSpaceMetrics",
    "MemoryTrajectoryCapture",
    "AttentionGeometryTracker",
    "ManifoldEvolutionMonitor",
    "OmegaQualityMeasure",
    "ConvergenceMetrics",
]

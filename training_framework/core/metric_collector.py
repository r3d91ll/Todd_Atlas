"""
Metric Collector - Streams training metrics to JSONL for dashboard consumption.

Provides buffered writing with configurable flush intervals for performance.
Supports both real-time streaming and batch analysis.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque


@dataclass
class MetricCollectorConfig:
    """Configuration for metric collection."""
    stream_path: str = "metrics_stream.jsonl"
    flush_interval: int = 10  # Flush every N metrics
    flush_timeout: float = 5.0  # Flush after N seconds regardless
    buffer_size: int = 1000  # Max buffer before forced flush
    rolling_window: int = 1000  # Window size for rolling statistics


class MetricCollector:
    """
    Collects and streams training metrics to JSONL file.

    Features:
    - Buffered writes for performance
    - Thread-safe operation
    - Rolling statistics computation
    - Automatic flush on timeout
    """

    def __init__(self, config: MetricCollectorConfig):
        self.config = config
        self.stream_path = Path(config.stream_path)
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.time()

        # Rolling window for statistics
        self._rolling_metrics: Dict[str, deque] = {}

        # Track total metrics logged
        self._total_logged = 0

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log a metrics dictionary.

        Args:
            metrics: Dictionary of metric name -> value
        """
        # Add timestamp if not present
        if 'timestamp' not in metrics:
            metrics['timestamp'] = time.time()

        with self._lock:
            self._buffer.append(metrics)
            self._total_logged += 1

            # Update rolling windows
            self._update_rolling(metrics)

            # Check flush conditions
            should_flush = (
                len(self._buffer) >= self.config.flush_interval or
                len(self._buffer) >= self.config.buffer_size or
                (time.time() - self._last_flush) >= self.config.flush_timeout
            )

            if should_flush:
                self._flush_locked()

    def _update_rolling(self, metrics: Dict[str, Any]) -> None:
        """Update rolling statistics windows."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in self._rolling_metrics:
                    self._rolling_metrics[key] = deque(maxlen=self.config.rolling_window)
                self._rolling_metrics[key].append(value)

    def get_rolling_stats(self, metric_name: str) -> Optional[Dict[str, float]]:
        """
        Get rolling statistics for a metric.

        Returns:
            Dict with mean, std, min, max, or None if metric not found
        """
        with self._lock:
            if metric_name not in self._rolling_metrics:
                return None

            values = list(self._rolling_metrics[metric_name])
            if not values:
                return None

            import statistics
            return {
                'mean': statistics.mean(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                'min': min(values),
                'max': max(values),
                'count': len(values),
            }

    def get_latest(self, n: int = 1) -> List[Dict[str, Any]]:
        """Get the latest N metrics from buffer or file."""
        with self._lock:
            if len(self._buffer) >= n:
                return self._buffer[-n:]

        # Read from file if needed
        try:
            with open(self.stream_path, 'r') as f:
                lines = f.readlines()
                recent = lines[-n:] if len(lines) >= n else lines
                return [json.loads(line) for line in recent]
        except FileNotFoundError:
            return []

    def flush(self) -> None:
        """Force flush the buffer to disk."""
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        """Internal flush (must hold lock)."""
        if not self._buffer:
            return

        with open(self.stream_path, 'a') as f:
            for metrics in self._buffer:
                f.write(json.dumps(metrics) + '\n')

        self._buffer.clear()
        self._last_flush = time.time()

    def get_total_logged(self) -> int:
        """Get total number of metrics logged."""
        return self._total_logged

    def read_all(self) -> List[Dict[str, Any]]:
        """Read all metrics from stream file."""
        self.flush()  # Ensure buffer is written

        metrics = []
        try:
            with open(self.stream_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
        except FileNotFoundError:
            pass

        return metrics

    def clear(self) -> None:
        """Clear all metrics (buffer and file)."""
        with self._lock:
            self._buffer.clear()
            self._rolling_metrics.clear()
            self._total_logged = 0

        if self.stream_path.exists():
            self.stream_path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()


class MetricAggregator:
    """
    Aggregates metrics across multiple sources (e.g., distributed training).
    """

    def __init__(self):
        self._aggregated: Dict[str, List[float]] = {}

    def add(self, metrics: Dict[str, Any], weight: float = 1.0) -> None:
        """Add metrics with optional weight."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in self._aggregated:
                    self._aggregated[key] = []
                self._aggregated[key].append(value * weight)

    def aggregate(self, method: str = 'mean') -> Dict[str, float]:
        """
        Aggregate collected metrics.

        Args:
            method: 'mean', 'sum', 'max', or 'min'
        """
        import statistics

        result = {}
        for key, values in self._aggregated.items():
            if not values:
                continue

            if method == 'mean':
                result[key] = statistics.mean(values)
            elif method == 'sum':
                result[key] = sum(values)
            elif method == 'max':
                result[key] = max(values)
            elif method == 'min':
                result[key] = min(values)

        return result

    def clear(self) -> None:
        """Clear aggregated metrics."""
        self._aggregated.clear()

"""
Observability metrics for Atlas training.

Logs:
- Training metrics (loss, perplexity, gradient norms)
- Memory metrics (norm, rank, update magnitude)
- Retention metrics (lambda values, penalties)
- Gating metrics (memory vs attention balance)
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsLogger:
    """
    Structured metrics logger with JSONL output.

    Writes metrics to JSONL file for dashboard consumption.
    Supports real-time streaming and aggregation.
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str = "atlas",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.memory_file = self.output_dir / "memory_stats.jsonl"

        # Running averages
        self.running_loss = []
        self.running_ppl = []

        # Memorization detection (from Todd_Atlas lessons)
        self.ppl_history = []  # For decline rate calculation
        self.val_ppl_history = []
        self.train_val_gaps = []

        # Step counter
        self.global_step = 0

    def log_step(
        self,
        step: int,
        loss: float,
        metrics: Optional[Dict[str, Any]] = None,
        lr: Optional[float] = None,
    ):
        """
        Log metrics for a single training step.

        Args:
            step: Global step number
            loss: Training loss
            metrics: Optional dict of additional metrics
            lr: Optional learning rate
        """
        self.global_step = step

        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "loss": loss,
            "perplexity": np.exp(loss) if loss < 100 else float("inf"),
        }

        if lr is not None:
            record["learning_rate"] = lr

        # Add running averages
        self.running_loss.append(loss)
        if len(self.running_loss) > 100:
            self.running_loss.pop(0)
        record["loss_avg_100"] = np.mean(self.running_loss)

        # PPL decline rate (memorization detector)
        ppl = record["perplexity"]
        self.ppl_history.append({"step": step, "ppl": ppl})
        if len(self.ppl_history) > 200:
            self.ppl_history.pop(0)

        # Calculate decline rate over last 100 steps
        if len(self.ppl_history) >= 100:
            recent = self.ppl_history[-100:]
            old_ppl = recent[0]["ppl"]
            new_ppl = recent[-1]["ppl"]
            if old_ppl > 0 and old_ppl != float("inf"):
                # Percentage decline per 100 steps
                decline_rate = ((old_ppl - new_ppl) / old_ppl) * 100
                record["ppl_decline_rate_100"] = decline_rate

                # WARNING: Constant high decline rate = memorization
                # Healthy: declining rate (e.g., 50% -> 30% -> 15%)
                # Overfit: constant rate (e.g., 50% -> 50% -> 50%)

        # Process nested metrics
        if metrics:
            flat_metrics = self._flatten_metrics(metrics)
            record.update(flat_metrics)

        # Write to JSONL
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_memory_state(
        self,
        step: int,
        layer_idx: int,
        W: torch.Tensor,
    ):
        """
        Log detailed memory state for a layer.

        Args:
            step: Global step
            layer_idx: Layer index
            W: Memory matrix [batch, d_key, d_value]
        """
        # Compute statistics (on first batch item)
        W_sample = W[0].detach().cpu().float()

        # SVD for rank analysis
        try:
            U, S, V = torch.linalg.svd(W_sample)
            singular_values = S.numpy().tolist()[:20]  # Top 20
            effective_rank = (S.sum() / S.max()).item() if S.max() > 0 else 0
        except Exception:
            singular_values = []
            effective_rank = 0

        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "layer": layer_idx,
            "frobenius_norm": W_sample.norm().item(),
            "spectral_norm": S[0].item() if len(S) > 0 else 0,
            "effective_rank": effective_rank,
            "singular_values": singular_values,
            "mean": W_sample.mean().item(),
            "std": W_sample.std().item(),
            "min": W_sample.min().item(),
            "max": W_sample.max().item(),
        }

        with open(self.memory_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_validation(
        self,
        step: int,
        val_loss: float,
        val_ppl: float,
    ):
        """Log validation metrics with generalization gap tracking."""
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "type": "validation",
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
        }

        # Track val PPL history
        self.val_ppl_history.append({"step": step, "val_ppl": val_ppl})

        # Generalization gap: train_ppl - val_ppl
        # Negative = overfitting (train doing better than val)
        # Should be close to 0 for healthy training
        if self.ppl_history:
            # Get train PPL at closest step
            train_ppl = self.ppl_history[-1]["ppl"]
            if train_ppl != float("inf") and val_ppl != float("inf"):
                gap = train_ppl - val_ppl
                record["generalization_gap"] = gap
                self.train_val_gaps.append({"step": step, "gap": gap})

                # WARNING thresholds
                if gap < -50:
                    record["warning"] = "SEVERE_OVERFIT"
                elif gap < -20:
                    record["warning"] = "MODERATE_OVERFIT"

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _flatten_metrics(
        self,
        metrics: Dict[str, Any],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Flatten nested metrics dict."""
        flat = {}

        for key, value in metrics.items():
            full_key = f"{prefix}{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_metrics(value, f"{full_key}/"))
            elif isinstance(value, (int, float)):
                flat[full_key] = value
            elif isinstance(value, torch.Tensor):
                flat[full_key] = value.item()

        return flat

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "total_steps": self.global_step,
            "final_loss": self.running_loss[-1] if self.running_loss else 0,
            "avg_loss_last_100": np.mean(self.running_loss) if self.running_loss else 0,
        }


class MemoryProfiler:
    """
    Profile memory usage during training.

    Tracks:
    - GPU memory allocation
    - Peak memory usage
    - Memory per component
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots = []

    def snapshot(self, label: str):
        """Take a memory snapshot."""
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            self.snapshots.append({
                "label": label,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
            })

    def report(self) -> Dict[str, Any]:
        """Generate memory report."""
        if not self.snapshots:
            return {}

        return {
            "snapshots": self.snapshots,
            "peak_allocated_gb": max(s["allocated_gb"] for s in self.snapshots),
            "peak_reserved_gb": max(s["reserved_gb"] for s in self.snapshots),
        }

    def reset(self):
        """Reset snapshots."""
        self.snapshots = []
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

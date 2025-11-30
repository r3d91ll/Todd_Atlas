"""
Weaver Space Measurements for Atlas Model Training.

This module captures the geometric dynamics in compute space (where geometry
gets constructed dynamically during inference), as opposed to static weight
patterns in storage space.

Key measurements:
1. Memory State Trajectories - How memory matrices evolve during Newton-Schulz iterations
2. Attention Geometry Evolution - How attention patterns construct semantic relationships
3. Semantic Manifold Construction - Effective dimensionality preservation through layers
4. Omega Rule Window Quality - Key-value mapping fidelity during sliding window optimization

These measurements test the Conveyance Hypothesis: information transfer effectiveness
depends on geometric construction in compute space, not storage patterns.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import numpy as np


@dataclass
class GeometricMetrics:
    """Container for geometric metrics computed from tensors."""

    # SVD-based metrics
    singular_values: Optional[Tensor] = None
    effective_rank: float = 0.0
    spectral_norm: float = 0.0
    nuclear_norm: float = 0.0
    condition_number: float = 0.0

    # Distribution metrics
    entropy: float = 0.0
    sparsity: float = 0.0

    # Effective dimensionality (PCA-based)
    d_eff: float = 0.0
    variance_explained_90: int = 0

    # Collapse indicator β
    collapse_beta: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to JSON-serializable dict (no tensors)."""
        return {
            "effective_rank": self.effective_rank,
            "spectral_norm": self.spectral_norm,
            "nuclear_norm": self.nuclear_norm,
            "condition_number": self.condition_number,
            "entropy": self.entropy,
            "sparsity": self.sparsity,
            "d_eff": self.d_eff,
            "variance_explained_90": self.variance_explained_90,
            "collapse_beta": self.collapse_beta,
        }


def compute_geometric_metrics(tensor: Tensor, compute_svd: bool = True) -> GeometricMetrics:
    """
    Compute geometric metrics for a tensor.

    Args:
        tensor: Input tensor (will be reshaped to 2D if needed)
        compute_svd: Whether to compute SVD-based metrics (expensive)

    Returns:
        GeometricMetrics with computed values
    """
    metrics = GeometricMetrics()

    # Flatten to 2D if needed
    if tensor.ndim > 2:
        tensor = tensor.reshape(tensor.shape[0], -1)
    elif tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)

    # Move to CPU and convert to float32 for stability
    tensor = tensor.detach().float().cpu()

    # Basic metrics
    metrics.sparsity = (tensor.abs() < 1e-6).float().mean().item()

    if compute_svd and tensor.numel() > 0:
        try:
            # SVD decomposition
            U, S, Vh = torch.linalg.svd(tensor, full_matrices=False)
            metrics.singular_values = S

            # Spectral norm (largest singular value)
            metrics.spectral_norm = S[0].item() if len(S) > 0 else 0.0

            # Nuclear norm (sum of singular values)
            metrics.nuclear_norm = S.sum().item()

            # Condition number (ratio of largest to smallest)
            if len(S) > 1 and S[-1] > 1e-10:
                metrics.condition_number = (S[0] / S[-1]).item()

            # Effective rank (using entropy of normalized singular values)
            if S.sum() > 1e-10:
                S_normalized = S / S.sum()
                S_normalized = S_normalized[S_normalized > 1e-10]  # Remove zeros
                entropy = -(S_normalized * S_normalized.log()).sum().item()
                metrics.entropy = entropy
                metrics.effective_rank = np.exp(entropy)

            # Effective dimensionality (PCA-based: dims for 90% variance)
            if S.sum() > 1e-10:
                variance_explained = (S ** 2).cumsum(0) / (S ** 2).sum()
                dims_90 = (variance_explained < 0.9).sum().item() + 1
                metrics.variance_explained_90 = dims_90
                metrics.d_eff = dims_90

            # Collapse indicator β (ratio of top-k variance to total)
            # Lower β means more distributed representation (less collapse)
            if len(S) > 1:
                top_k = min(5, len(S))
                top_variance = (S[:top_k] ** 2).sum()
                total_variance = (S ** 2).sum()
                if total_variance > 1e-10:
                    metrics.collapse_beta = (top_variance / total_variance).item()

        except Exception as e:
            # SVD can fail for degenerate matrices
            pass

    return metrics


class MemoryTrajectoryCapture:
    """
    Capture how Atlas memory states evolve during optimization.

    Tracks:
    - Memory matrix M at each state
    - Surprise metrics S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M; x_t)
    - Geometric properties (SVD spectrum, effective rank) of memory states
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 100,  # Capture every N batches
        max_history: int = 10,   # Keep last N trajectory snapshots
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.max_history = max_history

        # Storage
        self.trajectories: List[Dict[str, Any]] = []
        self.step_count = 0

    def should_capture(self) -> bool:
        """Check if we should capture this step."""
        return self.step_count % self.sample_rate == 0

    def capture(
        self,
        memory_state: Dict[str, Tensor],
        gradients: Optional[Dict[str, Tensor]] = None,
        loss: Optional[float] = None,
        step: int = 0,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Capture a memory state snapshot.

        Args:
            memory_state: Current memory parameters
            gradients: Gradients from omega rule (if available)
            loss: Current reconstruction loss
            step: Training step
            layer_idx: Which layer's memory

        Returns:
            Captured metrics dict
        """
        self.step_count += 1

        if not self.should_capture():
            return {}

        snapshot = {
            "step": step,
            "layer_idx": layer_idx,
            "timestamp": time.time(),
            "loss": loss,
            "memory_metrics": {},
            "gradient_metrics": {},
        }

        # Compute metrics for each memory parameter
        for name, param in memory_state.items():
            if param is not None and param.numel() > 0:
                # Sample a subset for large tensors
                if param.ndim == 3:  # Batched: take first batch element
                    param_sample = param[0]
                else:
                    param_sample = param

                metrics = compute_geometric_metrics(param_sample, compute_svd=True)
                snapshot["memory_metrics"][name] = metrics.to_dict()

        # Compute gradient metrics if available
        if gradients:
            for name, grad in gradients.items():
                if grad is not None and grad.numel() > 0:
                    if grad.ndim == 3:
                        grad_sample = grad[0]
                    else:
                        grad_sample = grad

                    metrics = compute_geometric_metrics(grad_sample, compute_svd=True)
                    snapshot["gradient_metrics"][name] = metrics.to_dict()

        # Store and manage history
        self.trajectories.append(snapshot)
        if len(self.trajectories) > self.max_history:
            self.trajectories.pop(0)

        return snapshot

    def save(self, epoch: int, batch: int):
        """Save trajectories to file."""
        if not self.trajectories:
            return

        output_file = self.output_dir / f"memory_trajectories_{epoch:04d}_{batch:06d}.json"
        with open(output_file, "w") as f:
            json.dump(self.trajectories, f, indent=2)

        # Clear after save
        self.trajectories = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of captured trajectories."""
        if not self.trajectories:
            return {}

        summary = {
            "num_snapshots": len(self.trajectories),
            "steps": [t["step"] for t in self.trajectories],
        }

        # Aggregate effective rank trends
        eff_ranks = []
        for t in self.trajectories:
            for name, metrics in t["memory_metrics"].items():
                if "effective_rank" in metrics:
                    eff_ranks.append(metrics["effective_rank"])

        if eff_ranks:
            summary["mean_effective_rank"] = np.mean(eff_ranks)
            summary["std_effective_rank"] = np.std(eff_ranks)

        return summary


class AttentionGeometryTracker:
    """
    Track how attention patterns construct semantic relationships.

    Measures:
    - SVD spectrum evolution across sequence positions
    - Attention entropy and sparsity patterns
    - Geometric structure changes with sequence length
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 100,
        num_heads_to_track: int = 4,  # Track subset of heads
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.num_heads_to_track = num_heads_to_track

        self.captures: List[Dict[str, Any]] = []
        self.step_count = 0

    def should_capture(self) -> bool:
        return self.step_count % self.sample_rate == 0

    def capture(
        self,
        attention_weights: Tensor,  # (batch, heads, seq, seq)
        layer_idx: int = 0,
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Capture attention geometry snapshot.

        Args:
            attention_weights: Attention matrix (batch, heads, q_len, kv_len)
            layer_idx: Which layer
            step: Training step

        Returns:
            Captured metrics
        """
        self.step_count += 1

        if not self.should_capture():
            return {}

        batch, num_heads, q_len, kv_len = attention_weights.shape

        # Take first batch element and subset of heads
        attn = attention_weights[0, :self.num_heads_to_track].detach().cpu().float()

        snapshot = {
            "step": step,
            "layer_idx": layer_idx,
            "timestamp": time.time(),
            "seq_len": q_len,
            "per_head_metrics": [],
            "aggregate_metrics": {},
        }

        # Per-head analysis
        for head_idx in range(min(num_heads, self.num_heads_to_track)):
            attn_head = attn[head_idx]  # (q_len, kv_len)

            head_metrics = {
                "head_idx": head_idx,
            }

            # Attention entropy (average across queries)
            # H = -sum(p * log(p)) where p = attention weights
            attn_probs = attn_head.clamp(min=1e-10)
            entropy = -(attn_probs * attn_probs.log()).sum(dim=-1).mean().item()
            head_metrics["entropy"] = entropy

            # Sparsity (% of weights below threshold)
            sparsity = (attn_head < 0.01).float().mean().item()
            head_metrics["sparsity"] = sparsity

            # Peak attention (max weight per query, averaged)
            peak_attn = attn_head.max(dim=-1)[0].mean().item()
            head_metrics["peak_attention"] = peak_attn

            # Attention spread (how many positions have >5% weight, averaged)
            spread = (attn_head > 0.05).float().sum(dim=-1).mean().item()
            head_metrics["attention_spread"] = spread

            # SVD of attention matrix
            geo_metrics = compute_geometric_metrics(attn_head, compute_svd=True)
            head_metrics["effective_rank"] = geo_metrics.effective_rank
            head_metrics["spectral_norm"] = geo_metrics.spectral_norm
            head_metrics["collapse_beta"] = geo_metrics.collapse_beta

            snapshot["per_head_metrics"].append(head_metrics)

        # Aggregate metrics across heads
        entropies = [m["entropy"] for m in snapshot["per_head_metrics"]]
        snapshot["aggregate_metrics"]["mean_entropy"] = np.mean(entropies)
        snapshot["aggregate_metrics"]["std_entropy"] = np.std(entropies)

        eff_ranks = [m["effective_rank"] for m in snapshot["per_head_metrics"]]
        snapshot["aggregate_metrics"]["mean_effective_rank"] = np.mean(eff_ranks)

        self.captures.append(snapshot)
        return snapshot

    def save(self, epoch: int, batch: int):
        """Save captures to file."""
        if not self.captures:
            return

        output_file = self.output_dir / f"attention_geometry_{epoch:04d}_{batch:06d}.json"
        with open(output_file, "w") as f:
            json.dump(self.captures, f, indent=2)

        self.captures = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of attention geometry."""
        if not self.captures:
            return {}

        all_entropies = []
        all_eff_ranks = []

        for c in self.captures:
            for m in c["per_head_metrics"]:
                all_entropies.append(m["entropy"])
                all_eff_ranks.append(m["effective_rank"])

        return {
            "num_captures": len(self.captures),
            "mean_entropy": np.mean(all_entropies) if all_entropies else 0,
            "mean_effective_rank": np.mean(all_eff_ranks) if all_eff_ranks else 0,
        }


class ManifoldEvolutionMonitor:
    """
    Monitor effective dimensionality preservation through layers.

    Measures:
    - D_eff using PCA with 90% variance threshold
    - Dimensional collapse/preservation across layers
    - Semantic path efficiency through manifold
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 100,
        variance_threshold: float = 0.9,  # For D_eff computation
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.variance_threshold = variance_threshold

        self.layer_metrics: List[Dict[str, Any]] = []
        self.step_count = 0

    def should_capture(self) -> bool:
        return self.step_count % self.sample_rate == 0

    def capture(
        self,
        hidden_states: List[Tensor],  # One per layer
        step: int = 0,
    ) -> Dict[str, Any]:
        """
        Capture manifold evolution across layers.

        Args:
            hidden_states: List of (batch, seq, d_model) tensors, one per layer
            step: Training step

        Returns:
            Captured metrics
        """
        self.step_count += 1

        if not self.should_capture():
            return {}

        snapshot = {
            "step": step,
            "timestamp": time.time(),
            "num_layers": len(hidden_states),
            "per_layer_metrics": [],
        }

        prev_d_eff = None

        for layer_idx, hidden in enumerate(hidden_states):
            # Take first batch element, reshape to (seq, d_model)
            h = hidden[0].detach().cpu().float()

            layer_metrics = {
                "layer_idx": layer_idx,
            }

            # Compute D_eff via PCA
            geo_metrics = compute_geometric_metrics(h, compute_svd=True)

            layer_metrics["d_eff"] = geo_metrics.d_eff
            layer_metrics["effective_rank"] = geo_metrics.effective_rank
            layer_metrics["collapse_beta"] = geo_metrics.collapse_beta
            layer_metrics["spectral_norm"] = geo_metrics.spectral_norm

            # Dimensional change from previous layer
            if prev_d_eff is not None:
                layer_metrics["d_eff_delta"] = geo_metrics.d_eff - prev_d_eff
                layer_metrics["d_eff_ratio"] = geo_metrics.d_eff / max(prev_d_eff, 1)

            prev_d_eff = geo_metrics.d_eff

            # Hidden state statistics
            layer_metrics["mean_activation"] = h.mean().item()
            layer_metrics["std_activation"] = h.std().item()
            layer_metrics["max_activation"] = h.abs().max().item()

            snapshot["per_layer_metrics"].append(layer_metrics)

        # Compute overall manifold health metrics
        d_effs = [m["d_eff"] for m in snapshot["per_layer_metrics"]]
        snapshot["d_eff_trajectory"] = d_effs
        snapshot["d_eff_preservation"] = d_effs[-1] / max(d_effs[0], 1) if d_effs else 0
        snapshot["min_d_eff"] = min(d_effs) if d_effs else 0
        snapshot["d_eff_variance"] = np.var(d_effs) if d_effs else 0

        self.layer_metrics.append(snapshot)
        return snapshot

    def save(self, epoch: int, batch: int):
        """Save manifold evolution to file."""
        if not self.layer_metrics:
            return

        output_file = self.output_dir / f"manifold_evolution_{epoch:04d}_{batch:06d}.json"
        with open(output_file, "w") as f:
            json.dump(self.layer_metrics, f, indent=2)

        self.layer_metrics = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of manifold evolution."""
        if not self.layer_metrics:
            return {}

        preservations = [m["d_eff_preservation"] for m in self.layer_metrics]
        min_d_effs = [m["min_d_eff"] for m in self.layer_metrics]

        return {
            "num_captures": len(self.layer_metrics),
            "mean_d_eff_preservation": np.mean(preservations),
            "mean_min_d_eff": np.mean(min_d_effs),
        }


class OmegaQualityMeasure:
    """
    Measure key-value mapping fidelity during sliding window optimization.

    Tracks:
    - Reconstruction error: ||M(k_i) - v_i||²
    - Mapping quality changes with window position
    - Correlation with surprise metrics and performance
    """

    def __init__(
        self,
        output_dir: Path,
        sample_rate: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        self.quality_metrics: List[Dict[str, Any]] = []
        self.step_count = 0

    def should_capture(self) -> bool:
        return self.step_count % self.sample_rate == 0

    def capture(
        self,
        keys: Tensor,           # (batch, window, d_key) or expanded
        values: Tensor,         # (batch, window, d_value)
        predictions: Tensor,    # (batch, window, d_value) - M(k)
        decay_weights: Tensor,  # (batch, window) - gamma_i
        step: int = 0,
        layer_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Capture omega rule quality metrics.

        Args:
            keys: Input keys to memory
            values: Target values
            predictions: Memory predictions M(k)
            decay_weights: Importance weights gamma_i
            step: Training step
            layer_idx: Layer index

        Returns:
            Captured metrics
        """
        self.step_count += 1

        if not self.should_capture():
            return {}

        # Move to CPU
        keys = keys.detach().cpu().float()
        values = values.detach().cpu().float()
        predictions = predictions.detach().cpu().float()
        decay_weights = decay_weights.detach().cpu().float()

        batch, window_size, d_value = values.shape

        snapshot = {
            "step": step,
            "layer_idx": layer_idx,
            "timestamp": time.time(),
            "window_size": window_size,
        }

        # Per-position reconstruction error
        errors = ((predictions - values) ** 2).sum(dim=-1)  # (batch, window)
        weighted_errors = (errors * decay_weights)

        # Take first batch
        errors_b0 = errors[0].numpy()
        decay_b0 = decay_weights[0].numpy()

        # Position-wise metrics
        snapshot["per_position_mse"] = errors_b0.tolist()
        snapshot["decay_weights"] = decay_b0.tolist()
        snapshot["weighted_error"] = (errors_b0 * decay_b0).tolist()

        # Aggregate metrics
        snapshot["mean_mse"] = errors.mean().item()
        snapshot["weighted_mse"] = weighted_errors.sum().item() / (decay_weights.sum().item() + 1e-8)
        snapshot["max_error"] = errors.max().item()
        snapshot["min_error"] = errors.min().item()

        # Error distribution by window position (early vs late)
        mid = window_size // 2
        early_error = errors_b0[:mid].mean() if mid > 0 else 0
        late_error = errors_b0[mid:].mean() if mid < window_size else 0
        snapshot["early_window_error"] = float(early_error)
        snapshot["late_window_error"] = float(late_error)
        snapshot["position_bias"] = float(late_error - early_error)

        # Key-value alignment (how well keys predict values)
        # Compute cosine similarity between predictions and values
        pred_norm = predictions / (predictions.norm(dim=-1, keepdim=True) + 1e-8)
        val_norm = values / (values.norm(dim=-1, keepdim=True) + 1e-8)
        cosine_sim = (pred_norm * val_norm).sum(dim=-1)  # (batch, window)

        snapshot["mean_cosine_similarity"] = cosine_sim.mean().item()
        snapshot["per_position_cosine"] = cosine_sim[0].numpy().tolist()

        self.quality_metrics.append(snapshot)
        return snapshot

    def save(self, epoch: int, batch: int):
        """Save quality metrics to file."""
        if not self.quality_metrics:
            return

        output_file = self.output_dir / f"omega_quality_{epoch:04d}_{batch:06d}.json"
        with open(output_file, "w") as f:
            json.dump(self.quality_metrics, f, indent=2)

        self.quality_metrics = []

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of omega quality."""
        if not self.quality_metrics:
            return {}

        mean_mses = [m["mean_mse"] for m in self.quality_metrics]
        cosine_sims = [m["mean_cosine_similarity"] for m in self.quality_metrics]

        return {
            "num_captures": len(self.quality_metrics),
            "mean_reconstruction_mse": np.mean(mean_mses),
            "mean_cosine_similarity": np.mean(cosine_sims),
        }


class WeaverSpaceMetrics:
    """
    Main interface for weaver space measurements.

    Orchestrates all measurement modules and provides unified API
    for integration with training loop.
    """

    def __init__(
        self,
        output_dir: str = "./weaver_metrics",
        sample_rate: int = 100,
        enabled: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.enabled = enabled

        # Initialize measurement modules
        self.memory_tracker = MemoryTrajectoryCapture(
            output_dir=self.output_dir / "memory",
            sample_rate=sample_rate,
        )
        self.attention_tracker = AttentionGeometryTracker(
            output_dir=self.output_dir / "attention",
            sample_rate=sample_rate,
        )
        self.manifold_monitor = ManifoldEvolutionMonitor(
            output_dir=self.output_dir / "manifold",
            sample_rate=sample_rate,
        )
        self.omega_quality = OmegaQualityMeasure(
            output_dir=self.output_dir / "omega",
            sample_rate=sample_rate,
        )

        # Hooks for model integration
        self._hooks = []
        self._hidden_states = []
        self._attention_weights = []

        # Current step tracking
        self.current_step = 0
        self.current_epoch = 0

    def register_hooks(self, model: nn.Module):
        """
        Register forward hooks on model to capture intermediate states.

        Args:
            model: Atlas model to instrument
        """
        if not self.enabled:
            return

        def capture_hidden_states(module, input, output):
            """Capture hidden states from transformer blocks."""
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            self._hidden_states.append(hidden.detach())

        def capture_attention(module, input, output):
            """Capture attention weights from attention modules."""
            # Output from SlidingWindowAttention is (output, kv_cache)
            # We need to hook deeper to get attention weights
            pass

        # Register hooks on DeepTransformerBlocks
        for name, module in model.named_modules():
            if "DeepTransformerBlock" in type(module).__name__:
                hook = module.register_forward_hook(capture_hidden_states)
                self._hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def step(
        self,
        step: int,
        epoch: int = 0,
        loss: Optional[float] = None,
        memory_state: Optional[Dict[str, Tensor]] = None,
        gradients: Optional[Dict[str, Tensor]] = None,
        omega_data: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Main step function called during training.

        Args:
            step: Current training step
            epoch: Current epoch
            loss: Training loss
            memory_state: Memory state dict from OmegaRule
            gradients: Gradients from omega optimization
            omega_data: Dict with keys, values, predictions, decay_weights

        Returns:
            Dict of captured metrics (empty if not sampling this step)
        """
        if not self.enabled:
            return {}

        self.current_step = step
        self.current_epoch = epoch

        metrics = {}

        # Capture memory trajectory
        if memory_state is not None:
            mem_metrics = self.memory_tracker.capture(
                memory_state=memory_state,
                gradients=gradients,
                loss=loss,
                step=step,
            )
            if mem_metrics:
                metrics["memory"] = mem_metrics

        # Capture manifold evolution from hooks
        if self._hidden_states:
            manifold_metrics = self.manifold_monitor.capture(
                hidden_states=self._hidden_states,
                step=step,
            )
            if manifold_metrics:
                metrics["manifold"] = manifold_metrics
            self._hidden_states = []

        # Capture omega quality
        if omega_data is not None:
            omega_metrics = self.omega_quality.capture(
                keys=omega_data.get("keys"),
                values=omega_data.get("values"),
                predictions=omega_data.get("predictions"),
                decay_weights=omega_data.get("decay_weights"),
                step=step,
            )
            if omega_metrics:
                metrics["omega"] = omega_metrics

        return metrics

    def save_checkpoint(self, batch: int):
        """Save all accumulated metrics."""
        if not self.enabled:
            return

        self.memory_tracker.save(self.current_epoch, batch)
        self.attention_tracker.save(self.current_epoch, batch)
        self.manifold_monitor.save(self.current_epoch, batch)
        self.omega_quality.save(self.current_epoch, batch)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            "memory": self.memory_tracker.get_summary(),
            "attention": self.attention_tracker.get_summary(),
            "manifold": self.manifold_monitor.get_summary(),
            "omega": self.omega_quality.get_summary(),
        }

    def log_to_jsonl(self, filepath: str = None) -> str:
        """Write summary to JSONL file for dashboard."""
        if filepath is None:
            filepath = self.output_dir / "weaver_summary.jsonl"

        summary = self.get_summary()
        summary["step"] = self.current_step
        summary["epoch"] = self.current_epoch
        summary["timestamp"] = time.time()

        with open(filepath, "a") as f:
            f.write(json.dumps(summary) + "\n")

        return str(filepath)

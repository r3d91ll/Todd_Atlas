"""
Grokking Detection Metrics for Atlas Models.

Implements geometric structure metrics from the grokking literature,
adapted for Atlas's unique architecture with Omega memory.

Key metrics:
- Fourier concentration: Detect periodic structure in embeddings/memory
- Circular fit: Detect circular organization in 2D PCA projection
- Effective dimensionality: Track compression of representations
- Embedding entropy: Measure organization of embedding space

These metrics help detect when the model has developed stable internal
geometry suitable for Kakeya set extraction.

References:
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting"
- Nanda et al. (2023): "Progress Measures for Grokking via Mechanistic Interpretability"
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque


@dataclass
class GrokkingConfig:
    """Configuration for grokking detection."""
    enabled: bool = True
    metrics_interval: int = 500
    track_embeddings: bool = True
    track_memory_matrices: bool = True
    track_hidden_states: bool = True

    # Thresholds for phase detection
    fourier_concentration_target: float = 0.5
    circular_fit_target: float = 0.8
    effective_dim_ratio_target: float = 0.3

    # Atlas-specific thresholds
    min_gate_mean: float = 0.1  # Gate collapse threshold - below this = memory bypassed
    min_retrieval_accuracy: float = 0.5  # Lower for language models (not classification)

    # Trend detection window
    trend_window: int = 10  # Number of measurements for trend


@dataclass
class GrokkingMetrics:
    """Container for grokking-related metrics."""
    step: int = 0

    # Multi-task accuracies (NEW: separate tracking for domain universality test)
    math_accuracy: float = 0.0           # Modular arithmetic accuracy
    masked_word_accuracy: float = 0.0    # Language masked word accuracy
    overall_accuracy: float = 0.0        # Combined (50-50 weighted)

    # Embedding metrics (computed on MATH embeddings when available)
    # NOTE: Fourier/Circular are meaningless for natural language embeddings
    # They detect circular structure in mod p arithmetic
    embedding_fourier_concentration: float = 0.0
    embedding_circular_fit: float = 0.0
    embedding_effective_dim: int = 0
    embedding_effective_dim_ratio: float = 0.0
    embedding_entropy: float = 0.0

    # Memory matrix metrics (averaged across layers)
    memory_fourier_concentration: float = 0.0
    memory_effective_dim: int = 0
    memory_effective_dim_ratio: float = 0.0
    memory_rank: float = 0.0

    # Per-layer memory metrics
    memory_metrics_by_layer: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Hidden state metrics
    hidden_effective_dim: float = 0.0
    hidden_entropy: float = 0.0

    # Atlas-specific metrics
    gate_mean: float = 0.0  # Average gate value - low = memory bypassed
    gate_healthy: bool = True  # True if gate_mean > threshold

    # Grokking phase detection
    phase: str = "unknown"  # memorization, circuit_formation, cleanup, grokked

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "step": self.step,
            # Multi-task accuracies
            "grokking/math_accuracy": self.math_accuracy,
            "grokking/masked_word_accuracy": self.masked_word_accuracy,
            "grokking/overall_accuracy": self.overall_accuracy,
            # Geometric metrics (meaningful for MATH task only)
            "grokking/embedding_fourier_concentration": self.embedding_fourier_concentration,
            "grokking/embedding_circular_fit": self.embedding_circular_fit,
            "grokking/embedding_effective_dim": self.embedding_effective_dim,
            "grokking/embedding_effective_dim_ratio": self.embedding_effective_dim_ratio,
            "grokking/embedding_entropy": self.embedding_entropy,
            "grokking/memory_fourier_concentration": self.memory_fourier_concentration,
            "grokking/memory_effective_dim": self.memory_effective_dim,
            "grokking/memory_effective_dim_ratio": self.memory_effective_dim_ratio,
            "grokking/memory_rank": self.memory_rank,
            "grokking/hidden_effective_dim": self.hidden_effective_dim,
            "grokking/hidden_entropy": self.hidden_entropy,
            "grokking/gate_mean": self.gate_mean,
            "grokking/gate_healthy": self.gate_healthy,
            "grokking/phase": self.phase,
        }


class GrokkingDetector:
    """
    Detects grokking phases by analyzing geometric structure of representations.

    Adapted for Atlas architecture:
    - Analyzes input embeddings (like standard transformers)
    - Analyzes memory matrices W (Atlas-specific)
    - Uses retrieval accuracy as primary grokking signal

    Grokking Phases:
    1. memorization: Train accuracy high, val accuracy low, no structure
    2. circuit_formation: Geometric metrics improving, val flat
    3. cleanup: Val accuracy jumping, structure solidifying
    4. grokked: Stable high performance + stable geometry
    """

    def __init__(self, config: GrokkingConfig):
        self.config = config
        self.history: deque = deque(maxlen=100)  # Keep last 100 measurements
        self.last_metrics: Optional[GrokkingMetrics] = None

    def compute_metrics(
        self,
        model: torch.nn.Module,
        step: int,
        train_metrics: Optional[Dict] = None,
        val_metrics: Optional[Dict] = None,
        task_metrics: Optional[Dict] = None,  # NEW: multi-task accuracies
        gate_mean: float = 1.0,
    ) -> GrokkingMetrics:
        """
        Compute all grokking-related metrics.

        Args:
            model: Atlas model to analyze
            step: Current training step
            train_metrics: Optional dict with train_loss, train_accuracy
            val_metrics: Optional dict with val_loss, val_accuracy, retrieval_accuracy
            task_metrics: Optional dict with math_accuracy, masked_word_accuracy (multi-task)
            gate_mean: Average gate value from training (Atlas-specific)

        Returns:
            GrokkingMetrics with all computed values

        Note:
            Fourier concentration and circular fit are ONLY meaningful for
            modular arithmetic (math task). For natural language, these
            metrics are noise - language has no reason to form circles.
        """
        metrics = GrokkingMetrics(step=step)

        # Atlas-specific: Gate health check
        metrics.gate_mean = gate_mean
        metrics.gate_healthy = gate_mean >= self.config.min_gate_mean

        # Multi-task accuracies (NEW: track separately for domain universality test)
        if task_metrics:
            metrics.math_accuracy = task_metrics.get('math_accuracy', 0.0)
            metrics.masked_word_accuracy = task_metrics.get('masked_word_accuracy', 0.0)
            # Combined with 50-50 weighting (matches training split)
            if metrics.math_accuracy > 0 or metrics.masked_word_accuracy > 0:
                metrics.overall_accuracy = (
                    metrics.math_accuracy * 0.5 +
                    metrics.masked_word_accuracy * 0.5
                )

        with torch.no_grad():
            # 1. Embedding metrics
            # NOTE: Fourier/Circular only meaningful for MATH embeddings
            # For language, these will be noise, but we compute them anyway
            # and use math_accuracy for phase detection
            if self.config.track_embeddings:
                embeddings = self._get_embeddings(model)
                if embeddings is not None:
                    metrics.embedding_fourier_concentration = self._fourier_concentration(embeddings)
                    metrics.embedding_circular_fit = self._circular_fit(embeddings)
                    metrics.embedding_effective_dim = self._effective_dimensionality(embeddings)
                    metrics.embedding_effective_dim_ratio = metrics.embedding_effective_dim / embeddings.shape[1]
                    metrics.embedding_entropy = self._embedding_entropy(embeddings)

            # 2. Memory matrix metrics
            if self.config.track_memory_matrices:
                memory_matrices = self._get_memory_matrices(model)
                if memory_matrices:
                    layer_metrics = []
                    for layer_idx, W in memory_matrices.items():
                        layer_m = self._analyze_memory_matrix(W)
                        metrics.memory_metrics_by_layer[layer_idx] = layer_m
                        layer_metrics.append(layer_m)

                    # Average across layers
                    if layer_metrics:
                        metrics.memory_fourier_concentration = np.mean([m["fourier_concentration"] for m in layer_metrics])
                        metrics.memory_effective_dim = np.mean([m["effective_dim"] for m in layer_metrics])
                        metrics.memory_effective_dim_ratio = np.mean([m["effective_dim_ratio"] for m in layer_metrics])
                        metrics.memory_rank = np.mean([m["effective_rank"] for m in layer_metrics])

            # 3. Hidden state metrics (if available from forward pass)
            # These would need to be passed in from the training loop

        # 4. Phase detection (uses MATH metrics where meaningful)
        metrics.phase = self._detect_phase(metrics, train_metrics, val_metrics, task_metrics)

        # Store for history
        self.history.append(metrics)
        self.last_metrics = metrics

        return metrics

    def _get_embeddings(self, model: torch.nn.Module) -> Optional[np.ndarray]:
        """Extract token embeddings from model."""
        try:
            # Try common attribute names (including Atlas's token_emb)
            for attr in ["token_emb", "embed_tokens", "token_embedding", "wte", "embedding"]:
                if hasattr(model, attr):
                    emb = getattr(model, attr)
                    if hasattr(emb, "weight"):
                        return emb.weight.detach().cpu().numpy()

            # Try nested in transformer
            if hasattr(model, "transformer"):
                return self._get_embeddings(model.transformer)

            return None
        except Exception:
            return None

    def _get_memory_matrices(self, model: torch.nn.Module) -> Dict[int, np.ndarray]:
        """Extract memory matrices W from all Atlas blocks."""
        memory_matrices = {}

        try:
            # Navigate to blocks
            blocks = None
            if hasattr(model, "blocks"):
                blocks = model.blocks
            elif hasattr(model, "layers"):
                blocks = model.layers
            elif hasattr(model, "transformer") and hasattr(model.transformer, "blocks"):
                blocks = model.transformer.blocks

            if blocks is None:
                return {}

            for layer_idx, block in enumerate(blocks):
                # Try to get memory matrix from block
                W = None

                # Atlas Omega: memory state cached in _last_memory_state as (M, S, context_buffer)
                if hasattr(block, "_last_memory_state") and block._last_memory_state is not None:
                    state = block._last_memory_state
                    if isinstance(state, tuple) and len(state) > 0:
                        W = state[0]  # M matrix is first element

                # Fallback: try other common patterns
                if W is None and hasattr(block, "memory"):
                    memory = block.memory
                    if hasattr(memory, "M"):
                        W = memory.M
                    elif hasattr(memory, "W"):
                        W = memory.W
                    elif hasattr(memory, "get_current_state"):
                        state = memory.get_current_state()
                        if isinstance(state, tuple) and len(state) > 0:
                            W = state[0]

                if W is not None:
                    # Handle batched memory (take first batch)
                    if W.dim() == 3:
                        W = W[0]
                    memory_matrices[layer_idx] = W.detach().cpu().numpy()

        except Exception:
            pass

        return memory_matrices

    def _fourier_concentration(self, embeddings: np.ndarray, k_max: int = 10) -> float:
        """
        Compute Fourier concentration of embeddings.

        Higher values indicate more periodic/structured representations,
        which are associated with grokking.

        Args:
            embeddings: [vocab_size, embed_dim] array
            k_max: Number of low-frequency components to consider

        Returns:
            Ratio of energy in low frequencies (0-1)
        """
        try:
            # FFT along vocab dimension
            fft_coeffs = np.fft.fft(embeddings, axis=0)

            # Energy in first k_max frequencies (excluding DC)
            low_freq_energy = np.sum(np.abs(fft_coeffs[1:k_max+1])**2)
            total_energy = np.sum(np.abs(fft_coeffs[1:])**2)  # Exclude DC

            if total_energy < 1e-10:
                return 0.0

            return float(low_freq_energy / total_energy)
        except Exception:
            return 0.0

    def _circular_fit(self, embeddings: np.ndarray) -> float:
        """
        Compute how well embeddings fit a circle in 2D PCA projection.

        Grokking models often arrange embeddings in circular patterns.

        Args:
            embeddings: [vocab_size, embed_dim] array

        Returns:
            Circular fit score (0-1, higher = more circular)
        """
        try:
            from sklearn.decomposition import PCA

            # Project to 2D
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)

            # Fit circle using least squares
            center, radius = self._fit_circle(embeddings_2d)

            # Compute residuals
            distances = np.sqrt(np.sum((embeddings_2d - center)**2, axis=1))
            residuals = np.abs(distances - radius)

            # Normalized residual (lower = better fit)
            if radius < 1e-10:
                return 0.0
            normalized_residual = np.std(residuals) / radius

            # Convert to score (1 = perfect circle)
            return float(max(0, 1.0 - normalized_residual))
        except Exception:
            return 0.0

    def _fit_circle(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Fit a circle to 2D points using algebraic least squares."""
        x, y = points[:, 0], points[:, 1]

        # Construct design matrix
        A = np.column_stack([x, y, np.ones(len(x))])
        b = x**2 + y**2

        # Solve least squares
        try:
            c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

            center_x = c[0] / 2
            center_y = c[1] / 2
            radius = np.sqrt(c[2] + center_x**2 + center_y**2)

            return np.array([center_x, center_y]), radius
        except Exception:
            return np.array([0, 0]), 1.0

    def _effective_dimensionality(self, embeddings: np.ndarray, threshold: float = 0.95) -> int:
        """
        Compute effective dimensionality via PCA.

        Lower effective dimensionality indicates more structured/compressed
        representations, which is associated with generalization.

        Args:
            embeddings: [vocab_size, embed_dim] array
            threshold: Cumulative variance threshold (default 95%)

        Returns:
            Number of dimensions to explain threshold% of variance
        """
        try:
            from sklearn.decomposition import PCA

            pca = PCA().fit(embeddings)
            cumsum = np.cumsum(pca.explained_variance_ratio_)

            return int(np.searchsorted(cumsum, threshold) + 1)
        except Exception:
            return embeddings.shape[1]  # Fall back to full dimension

    def _embedding_entropy(self, embeddings: np.ndarray) -> float:
        """
        Compute entropy of embedding norms.

        Lower entropy indicates more organized embedding space.
        """
        try:
            norms = np.linalg.norm(embeddings, axis=1)

            # Normalize to distribution
            norms = norms / np.sum(norms)

            # Compute entropy
            entropy = -np.sum(norms * np.log(norms + 1e-10))

            return float(entropy)
        except Exception:
            return 0.0

    def _analyze_memory_matrix(self, W: np.ndarray) -> Dict[str, float]:
        """
        Analyze a single memory matrix for geometric structure.

        Args:
            W: Memory matrix [d_key, d_value]

        Returns:
            Dict with fourier_concentration, effective_dim, effective_rank
        """
        metrics = {
            "fourier_concentration": 0.0,
            "effective_dim": 0,
            "effective_dim_ratio": 0.0,
            "effective_rank": 0.0,
        }

        try:
            # Fourier concentration
            metrics["fourier_concentration"] = self._fourier_concentration(W)

            # SVD for rank and effective dimensionality
            U, S, Vh = np.linalg.svd(W, full_matrices=False)

            # Effective rank via entropy of normalized singular values
            # Formula: exp(-sum(p_i * log(p_i))) where p_i = s_i / sum(s)
            # Range: 1 (single dominant SV) to n (uniform spectrum)
            S_norm = S / (np.sum(S) + 1e-10)
            entropy = -np.sum(S_norm * np.log(S_norm + 1e-10))
            metrics["effective_rank"] = float(np.exp(entropy))

            # Effective dimensionality (cumulative energy)
            energy = S**2
            cumsum = np.cumsum(energy) / np.sum(energy)
            metrics["effective_dim"] = float(np.searchsorted(cumsum, 0.95) + 1)
            metrics["effective_dim_ratio"] = metrics["effective_dim"] / min(W.shape)

        except Exception:
            pass

        return metrics

    def _detect_phase(
        self,
        metrics: GrokkingMetrics,
        train_metrics: Optional[Dict],
        val_metrics: Optional[Dict],
        task_metrics: Optional[Dict] = None,
    ) -> str:
        """
        Detect current grokking phase based on metrics and trends.

        Multi-Task Phase Detection:
        - Uses MATH accuracy for phase detection (Fourier/Circular are meaningful here)
        - Language accuracy tracked separately but not used for phase
        - Requires healthy gate values (gate_mean > min_gate_mean)

        Phases:
        - memorization: High train acc, low val/retrieval acc, low structure
        - circuit_formation: Structure metrics (Fourier/Circular) increasing
        - cleanup: Math accuracy jumping up
        - grokked: Stable high math performance + stable structure + healthy gates
        """
        if len(self.history) < self.config.trend_window:
            return "insufficient_data"

        recent = list(self.history)[-self.config.trend_window:]

        # Compute trends (Fourier/Circular meaningful for MATH, noise for language)
        fourier_trend = self._compute_trend([m.embedding_fourier_concentration for m in recent])
        circular_trend = self._compute_trend([m.embedding_circular_fit for m in recent])
        # dim_trend reserved for future use
        _dim_trend = self._compute_trend([m.embedding_effective_dim_ratio for m in recent])

        # Multi-task: Use MATH accuracy as primary grokking signal
        # Fourier/Circular are meaningful for mod arithmetic, not language
        # masked_word_accuracy tracked separately but not used for phase detection
        math_acc = 0.0
        if task_metrics:
            math_acc = task_metrics.get("math_accuracy", 0)

        # Fallback to legacy retrieval accuracy
        retrieval_acc = val_metrics.get("retrieval_accuracy", 0) if val_metrics else 0
        train_acc = train_metrics.get("accuracy", 0) if train_metrics else 0
        val_acc = val_metrics.get("accuracy", 0) if val_metrics else 0

        # Phase detection logic
        structure_improving = (fourier_trend > 0.001 or circular_trend > 0.001)
        structure_stable = abs(fourier_trend) < 0.0001 and abs(circular_trend) < 0.0001

        # Atlas-specific: Check gate health
        gate_healthy = metrics.gate_healthy

        # Use MATH accuracy for grokking detection (where Fourier/Circular are meaningful)
        # High math accuracy = geometric structure learned
        high_math = math_acc > 0.9  # 90% math accuracy = clear grokking
        high_retrieval = retrieval_acc > self.config.min_retrieval_accuracy

        # Geometric structure thresholds
        high_structure = (
            metrics.embedding_fourier_concentration > self.config.fourier_concentration_target and
            metrics.embedding_circular_fit > self.config.circular_fit_target
        )

        # ATLAS REQUIREMENT: Must have healthy gates to be considered "grokked"
        # If gates collapsed, memory is bypassed and model hasn't truly grokked
        if (high_math or high_retrieval) and high_structure and structure_stable and gate_healthy:
            return "grokked"
        elif not gate_healthy:
            # Gate collapse - model is bypassing memory, stuck in attention-only mode
            return "gate_collapse"
        elif high_math or high_retrieval:
            return "cleanup"
        elif structure_improving:
            return "circuit_formation"
        else:
            return "memorization"

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        try:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            return float(slope)
        except Exception:
            return 0.0

    def has_grokked(self) -> bool:
        """Check if model has completed grokking."""
        if self.last_metrics is None:
            return False
        return self.last_metrics.phase == "grokked"

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of grokking detection state."""
        if not self.history:
            return {"status": "no_data"}

        latest = self.last_metrics
        return {
            "step": latest.step,
            "phase": latest.phase,
            "embedding_fourier": latest.embedding_fourier_concentration,
            "embedding_circular": latest.embedding_circular_fit,
            "embedding_dim_ratio": latest.embedding_effective_dim_ratio,
            "memory_rank": latest.memory_rank,
            "has_grokked": self.has_grokked(),
            "measurements": len(self.history),
        }


def create_grokking_detector(config_dict: Dict) -> GrokkingDetector:
    """Create detector from config dictionary."""
    grok_config = config_dict.get("monitoring", {}).get("grokking", {})

    config = GrokkingConfig(
        enabled=grok_config.get("enabled", True),
        metrics_interval=grok_config.get("metrics_interval", 500),
        track_embeddings=grok_config.get("track_embeddings", True),
        track_memory_matrices=grok_config.get("track_memory_matrices", True),
        track_hidden_states=grok_config.get("track_hidden_states", True),
        fourier_concentration_target=grok_config.get("fourier_concentration_target", 0.5),
        circular_fit_target=grok_config.get("circular_fit_target", 0.8),
        effective_dim_ratio_target=grok_config.get("effective_dim_ratio_target", 0.3),
    )

    return GrokkingDetector(config)

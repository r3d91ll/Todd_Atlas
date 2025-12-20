"""
Atlas Metrics Adapter - Memory-specific metrics for Atlas/Titans experiments.

This is the ONLY file that changes between experiments.
Implements comprehensive memory observability for gate collapse detection.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
from collections import deque

from .base_adapter import MetricsAdapter


class AtlasMetricsAdapter(MetricsAdapter):
    """
    Metrics adapter for Atlas models with Titans/Miras memory architecture.

    Collects:
    - Memory matrix health (rank, entropy, sparsity)
    - Gate health (collapse risk, dead/saturated neurons)
    - Storage/retrieval effectiveness
    - Memory access patterns
    """

    def __init__(
        self,
        track_per_layer: bool = True,
        rank_computation_interval: int = 100,  # Compute rank every N steps
        entropy_bins: int = 100,
    ):
        self.track_per_layer = track_per_layer
        self.rank_computation_interval = rank_computation_interval
        self.entropy_bins = entropy_bins

        # Step counter for expensive computations
        self._step = 0

        # Cache for expensive computations
        self._cached_ranks: Dict[int, float] = {}
        self._cached_entropy: Dict[int, float] = {}

        # History for storage/retrieval tracking
        self._storage_magnitudes: deque = deque(maxlen=100)
        self._retrieval_accuracies: deque = deque(maxlen=100)

    def collect(self, model: nn.Module, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect Atlas-specific metrics.

        Args:
            model: AtlasOmega model
            outputs: Training step outputs
        """
        self._step += 1
        metrics = {}

        # Unwrap DDP if needed
        if hasattr(model, 'module'):
            model = model.module

        # Get blocks (layers)
        blocks = getattr(model, 'blocks', [])
        if not blocks:
            blocks = getattr(model, 'layers', [])

        # === Memory Matrix Health ===
        all_memory_magnitudes = []
        all_memory_sparsities = []
        memory_ranks = []
        memory_entropies = []

        for i, block in enumerate(blocks):
            memory_state = self._get_memory_state(block)
            if memory_state is None:
                continue

            M = memory_state
            if isinstance(memory_state, tuple):
                M = memory_state[0]  # Memory matrix is first element

            # Basic stats
            magnitude = M.norm().item()
            sparsity = (M.abs() < 1e-6).float().mean().item()

            all_memory_magnitudes.append(magnitude)
            all_memory_sparsities.append(sparsity)

            if self.track_per_layer:
                metrics[f'layer_{i}/memory_magnitude'] = magnitude
                metrics[f'layer_{i}/memory_sparsity'] = sparsity
                metrics[f'layer_{i}/memory_mean'] = M.mean().item()
                metrics[f'layer_{i}/memory_std'] = M.std().item()

            # Expensive computations (periodic)
            if self._step % self.rank_computation_interval == 0:
                rank = self._compute_effective_rank(M)
                entropy = self._compute_matrix_entropy(M)
                memory_ranks.append(rank)
                memory_entropies.append(entropy)
                self._cached_ranks[i] = rank
                self._cached_entropy[i] = entropy

                if self.track_per_layer:
                    metrics[f'layer_{i}/memory_rank'] = rank
                    metrics[f'layer_{i}/memory_entropy'] = entropy
            else:
                # Use cached values
                if i in self._cached_ranks:
                    memory_ranks.append(self._cached_ranks[i])
                if i in self._cached_entropy:
                    memory_entropies.append(self._cached_entropy[i])

        # Aggregate memory metrics
        if all_memory_magnitudes:
            metrics['memory_magnitude_mean'] = sum(all_memory_magnitudes) / len(all_memory_magnitudes)
            metrics['memory_magnitude_std'] = self._std(all_memory_magnitudes)
            metrics['memory_sparsity'] = sum(all_memory_sparsities) / len(all_memory_sparsities)

        if memory_ranks:
            metrics['memory_rank_mean'] = sum(memory_ranks) / len(memory_ranks)

        if memory_entropies:
            metrics['memory_entropy_mean'] = sum(memory_entropies) / len(memory_entropies)

        # === Gate Health ===
        all_gates = self._collect_all_gates(model, blocks)
        if all_gates is not None and all_gates.numel() > 0:
            metrics['gate_mean'] = all_gates.mean().item()
            metrics['gate_std'] = all_gates.std().item()
            metrics['gate_min'] = all_gates.min().item()
            metrics['gate_max'] = all_gates.max().item()

            # Dead and saturated neurons
            metrics['gate_dead_neurons'] = (all_gates < 0.01).sum().item()
            metrics['gate_saturated_neurons'] = (all_gates > 0.99).sum().item()
            metrics['gate_dead_ratio'] = (all_gates < 0.01).float().mean().item()

            # Gate collapse risk: fraction of gates below 10%
            metrics['gate_collapse_risk'] = 1.0 - (all_gates > 0.10).float().mean().item()

            # Per-layer gate variance
            gate_variances = []
            for i, block in enumerate(blocks):
                block_gates = self._get_block_gates(block)
                if block_gates is not None:
                    var = block_gates.var().item()
                    gate_variances.append(var)
                    if self.track_per_layer:
                        metrics[f'layer_{i}/gate_mean'] = block_gates.mean().item()
                        metrics[f'layer_{i}/gate_var'] = var

            if gate_variances:
                metrics['gate_variance_mean'] = sum(gate_variances) / len(gate_variances)

        # === Storage/Retrieval Effectiveness ===
        if 'storage_success' in outputs:
            self._storage_magnitudes.append(outputs['storage_success'])
            metrics['storage_success_rate'] = sum(self._storage_magnitudes) / len(self._storage_magnitudes)

        if 'retrieval_accuracy' in outputs:
            self._retrieval_accuracies.append(outputs['retrieval_accuracy'])
            metrics['retrieval_hit_rate'] = sum(self._retrieval_accuracies) / len(self._retrieval_accuracies)

        # From retrieval verifier outputs
        if 'retrieval_token_accuracy' in outputs:
            metrics['retrieval_token_accuracy'] = outputs['retrieval_token_accuracy']
        if 'retrieval_exact_match' in outputs:
            metrics['retrieval_exact_match'] = outputs['retrieval_exact_match']
        if 'memory_retention_similarity' in outputs:
            metrics['memory_retention_similarity'] = outputs['memory_retention_similarity']

        # === Memory Access Patterns ===
        # These require tracking across steps - simplified version
        if hasattr(model, '_last_memory_keys'):
            keys = model._last_memory_keys
            if keys is not None:
                unique_ratio = keys.unique().numel() / max(keys.numel(), 1)
                metrics['memory_key_diversity'] = unique_ratio

        # === Surprise Accumulator (if available) ===
        all_surprise_norms = []
        for i, block in enumerate(blocks):
            S = self._get_surprise_state(block)
            if S is not None:
                s_norm = S.norm().item()
                all_surprise_norms.append(s_norm)
                if self.track_per_layer:
                    metrics[f'layer_{i}/surprise_norm'] = s_norm

        if all_surprise_norms:
            metrics['surprise_norm_mean'] = sum(all_surprise_norms) / len(all_surprise_norms)

        # === Training Metrics from outputs ===
        if 'loss' in outputs:
            loss_val = outputs['loss']
            metrics['loss'] = loss_val.item() if hasattr(loss_val, 'item') else loss_val
        if 'perplexity' in outputs:
            metrics['perplexity'] = outputs['perplexity']
        if 'storage_loss' in outputs:
            metrics['storage_loss'] = outputs['storage_loss']
        if 'retrieval_loss' in outputs:
            metrics['retrieval_loss'] = outputs['retrieval_loss']

        return metrics

    def get_alert_thresholds(self) -> Dict[str, Tuple[float, str]]:
        """Return alert thresholds for Atlas metrics."""
        return {
            # Gate health - CRITICAL
            'gate_collapse_risk': (0.80, 'critical'),  # 80% of gates below 10%
            'gate_dead_ratio': (0.30, 'warning'),  # 30% of gates dead

            # Retrieval effectiveness - WARNING
            'retrieval_token_accuracy': (0.50, 'warning_below'),  # Below 50%
            'retrieval_exact_match': (0.30, 'warning_below'),  # Below 30%

            # Memory health
            'memory_sparsity': (0.50, 'warning'),  # 50% of memory dead
            'memory_rank_mean': (5.0, 'warning_below'),  # Rank too low

            # Training
            'loss': (10.0, 'critical'),  # Very high loss
        }

    def _get_memory_state(self, block) -> Optional[torch.Tensor]:
        """Extract memory state from a block."""
        # First try block-level cached state (AtlasOmegaBlock caches this)
        if hasattr(block, '_last_memory_state') and block._last_memory_state is not None:
            state = block._last_memory_state
            # State is tuple (M, S, context_buffer) - return M
            if isinstance(state, tuple) and len(state) > 0:
                return state[0]
            return state

        # Try different attribute names on memory modules
        for attr in ['memory', 'memory_module', 'titans_memory', 'omega_memory']:
            mem_module = getattr(block, attr, None)
            if mem_module is not None:
                # Try to get state
                for state_attr in ['M', 'memory_matrix', '_M', 'state', '_last_state']:
                    state = getattr(mem_module, state_attr, None)
                    if state is not None:
                        return state

                # Try get_memory_state or get_state methods
                for method in ['get_memory_state', 'get_state']:
                    if hasattr(mem_module, method):
                        state = getattr(mem_module, method)()
                        if state is not None:
                            return state

        return None

    def _get_surprise_state(self, block) -> Optional[torch.Tensor]:
        """Extract surprise accumulator from a block."""
        # First try block-level cached state (AtlasOmegaBlock caches this)
        # State is tuple (M, S, context_buffer) - S is second element
        if hasattr(block, '_last_memory_state') and block._last_memory_state is not None:
            state = block._last_memory_state
            if isinstance(state, tuple) and len(state) > 1:
                return state[1]  # S is the second element

        for attr in ['memory', 'memory_module', 'titans_memory', 'omega_memory']:
            mem_module = getattr(block, attr, None)
            if mem_module is not None:
                for s_attr in ['S', 'surprise', '_S', 'momentum']:
                    S = getattr(mem_module, s_attr, None)
                    if S is not None:
                        return S
        return None

    def _get_block_gates(self, block) -> Optional[torch.Tensor]:
        """Extract gate values from a block."""
        # Try gating module
        gating = getattr(block, 'gating', None) or getattr(block, 'gate', None)
        if gating is not None:
            if hasattr(gating, '_last_gate'):
                return gating._last_gate
            if hasattr(gating, 'last_gate_values'):
                return gating.last_gate_values

        return None

    def _collect_all_gates(self, model, blocks) -> Optional[torch.Tensor]:
        """Collect all gate values across the model."""
        all_gates = []

        for block in blocks:
            gates = self._get_block_gates(block)
            if gates is not None:
                all_gates.append(gates.flatten())

        if all_gates:
            return torch.cat(all_gates)

        # Try model-level method
        if hasattr(model, 'get_all_gates'):
            return model.get_all_gates()

        return None

    def _compute_effective_rank(self, M: torch.Tensor) -> float:
        """
        Compute effective rank via singular values.

        Effective rank = exp(entropy of normalized singular values)
        """
        try:
            # Flatten if needed
            if M.dim() > 2:
                M = M.view(M.size(0), -1)

            # SVD
            U, S, V = torch.svd(M.float())

            # Normalize singular values
            S_norm = S / (S.sum() + 1e-10)

            # Entropy
            entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()

            # Effective rank
            return math.exp(entropy)
        except Exception:
            return 0.0

    def _compute_matrix_entropy(self, M: torch.Tensor) -> float:
        """
        Compute entropy of matrix values (information content).
        """
        try:
            # Flatten and normalize
            values = M.flatten().float()
            values = values - values.min()
            values = values / (values.max() + 1e-10)

            # Histogram
            hist = torch.histc(values, bins=self.entropy_bins, min=0, max=1)
            hist = hist / hist.sum()

            # Entropy
            entropy = -(hist * torch.log(hist + 1e-10)).sum().item()

            return entropy
        except Exception:
            return 0.0

    def _std(self, values: List[float]) -> float:
        """Compute standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def on_checkpoint(self, model: nn.Module, step: int) -> Dict[str, Any]:
        """Perform checkpoint-specific analysis."""
        # Force expensive computations
        self._step = step  # Ensure we compute ranks
        return {
            'checkpoint_step': step,
            'cached_ranks': dict(self._cached_ranks),
            'cached_entropy': dict(self._cached_entropy),
        }

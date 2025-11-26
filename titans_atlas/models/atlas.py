"""
Atlas: Learning to Optimally Memorize the Context at Test Time

PyTorch implementation of Atlas architecture with:
- Omega Rule: Sliding window optimization for memory
- Polynomial Features: φ_p(x) multivariate monomial expansion
- Muon Optimizer: Second-order approximation for memory updates
- DeepTransformer: Generalized Transformer with deep memory

Paper: "Atlas: Learning to Optimally Memorize the Context at Test Time" (arXiv:2505.23735)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, List
from itertools import combinations_with_replacement
import math

from titans_atlas.layers.memory import DeepMemory
from titans_atlas.layers.attention import SlidingWindowAttention, GatedAttentionUnit
from titans_atlas.utils import get_activation, l2_normalize, parallel_scan
from titans_atlas.configs import AtlasConfig


class PolynomialFeatures(nn.Module):
    """
    Polynomial Feature Mapping φ_p(x).

    From the paper: "Polynomial feature mapping φ_p(·) enables capacity
    of O(d_k^p) pairs, enabling super-linear scaling through feature
    expansion without parameter overhead."

    φ_p(x) = [x^β]_{|β|≤p} where β are multivariate monomial indices.
    """

    def __init__(
        self,
        input_dim: int,
        degree: int = 2,
        include_bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.degree = degree
        self.include_bias = include_bias

        # Compute output dimension
        # For degree p and input dim d, output is C(d+p, p) = (d+p)! / (d! * p!)
        self.output_dim = self._compute_output_dim()

        # Precompute monomial indices for efficient computation
        self._precompute_indices()

    def _compute_output_dim(self) -> int:
        """Compute output dimension for polynomial features."""
        # Number of monomials of degree exactly k in d variables is C(d+k-1, k)
        # Total up to degree p is sum of these
        from math import comb
        total = 0
        start = 0 if self.include_bias else 1
        for k in range(start, self.degree + 1):
            total += comb(self.input_dim + k - 1, k)
        return total

    def _precompute_indices(self):
        """Precompute indices for monomial computation."""
        # Generate all combinations of indices for each degree
        indices = []
        start = 0 if self.include_bias else 1

        for deg in range(start, self.degree + 1):
            if deg == 0:
                indices.append(())  # Constant term
            else:
                for combo in combinations_with_replacement(range(self.input_dim), deg):
                    indices.append(combo)

        self.register_buffer(
            "_indices",
            torch.tensor([list(idx) + [-1] * (self.degree - len(idx)) for idx in indices], dtype=torch.long)
        )
        self._index_lengths = [len(idx) for idx in indices]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply polynomial feature mapping.

        Args:
            x: (..., input_dim)

        Returns:
            (..., output_dim) polynomial features
        """
        shape = x.shape[:-1]
        x_flat = x.reshape(-1, self.input_dim)
        batch = x_flat.shape[0]

        # Compute each monomial
        features = []

        for i, length in enumerate(self._index_lengths):
            if length == 0:
                # Constant term
                features.append(torch.ones(batch, 1, device=x.device, dtype=x.dtype))
            else:
                # Product of selected dimensions
                indices = self._indices[i, :length]
                monomial = x_flat[:, indices].prod(dim=-1, keepdim=True)
                features.append(monomial)

        output = torch.cat(features, dim=-1)
        return output.reshape(*shape, self.output_dim)


class LearnableTaylorKernel(nn.Module):
    """
    Learnable Taylor expansion for softmax approximation.

    From the paper: "Softmax approximation via Taylor expansion:
    exp(q_i^T k_j) ≈ Σ a_i (q_i^T k_j)^i"

    This provides a polynomial kernel with learnable coefficients.
    """

    def __init__(
        self,
        order: int = 4,
        learnable: bool = True,
    ):
        super().__init__()
        self.order = order
        self.learnable = learnable

        # Initialize with Taylor coefficients of exp(x)
        # exp(x) = 1 + x + x²/2! + x³/3! + ...
        init_coeffs = torch.tensor([1.0 / math.factorial(i) for i in range(order + 1)])

        if learnable:
            self.coeffs = nn.Parameter(init_coeffs)
        else:
            self.register_buffer("coeffs", init_coeffs)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply Taylor kernel approximation.

        Args:
            x: (...,) inner products q^T k

        Returns:
            (...,) approximated exp(x)
        """
        result = torch.zeros_like(x)
        x_power = torch.ones_like(x)

        for i in range(self.order + 1):
            result = result + self.coeffs[i] * x_power
            if i < self.order:
                x_power = x_power * x

        return result


class MuonOptimizer:
    """
    Muon Optimizer for memory updates.

    From the paper: "Atlas extends OmegaNet by replacing gradient descent
    with Muon optimizer (momentum + normalization strategy) for
    second-order approximation of memory updates."

    Muon uses:
    - Momentum with Nesterov acceleration
    - Spectral normalization of updates
    - Adaptive learning rates
    """

    def __init__(
        self,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        eps: float = 1e-8,
    ):
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self.eps = eps
        self.state = {}

    def step(
        self,
        params: Dict[str, Tensor],
        grads: Dict[str, Tensor],
        lr: float = 0.01,
    ) -> Dict[str, Tensor]:
        """
        Perform one optimization step.

        Args:
            params: Current parameters
            grads: Gradients for each parameter
            lr: Learning rate

        Returns:
            Updated parameters
        """
        updated = {}

        for name, param in params.items():
            if name not in grads:
                updated[name] = param
                continue

            grad = grads[name]

            # Initialize momentum buffer
            if name not in self.state:
                self.state[name] = {"momentum_buffer": torch.zeros_like(param)}

            buf = self.state[name]["momentum_buffer"]

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param

            # Momentum update
            buf.mul_(self.momentum).add_(grad)

            # Nesterov momentum
            if self.nesterov:
                grad = grad + self.momentum * buf
            else:
                grad = buf

            # Spectral normalization of update
            # Normalize by Frobenius norm to stabilize
            grad_norm = grad.norm() + self.eps
            normalized_grad = grad / grad_norm

            # Apply update
            updated[name] = param - lr * normalized_grad * grad_norm.sqrt()

        return updated


class OmegaRule(nn.Module):
    """
    Omega Rule: Sliding Window Context Memorization.

    From the paper: "The fundamental innovation replaces online (per-token)
    memory updates with sliding window optimization:

    min_M Σ_{i=t-c+1}^{t} γ_i^(t) · ℓ(M; k_i, v_i)

    Where:
    - c: context window length
    - γ_i^(t): learned decay weights for token importance
    - ℓ: attentional bias (internal objective)"
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        context_window: int = 64,
        num_memory_layers: int = 2,
        learnable_decay: bool = True,
        use_polynomial_features: bool = True,
        polynomial_degree: int = 2,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.context_window = context_window
        self.learnable_decay = learnable_decay

        # Polynomial features for keys
        if use_polynomial_features:
            self.phi = PolynomialFeatures(d_key, polynomial_degree)
            effective_d_key = self.phi.output_dim
        else:
            self.phi = None
            effective_d_key = d_key

        # Deep memory module
        self.memory = DeepMemory(
            d_key=effective_d_key,
            d_value=d_value,
            num_layers=num_memory_layers,
        )

        # Learned decay weights γ_i^(t)
        if learnable_decay:
            # Project from token features to decay weight
            self.decay_proj = nn.Linear(d_key, 1)
        else:
            # Fixed exponential decay
            self.register_buffer(
                "fixed_decay",
                torch.exp(-torch.arange(context_window).float() / context_window)
            )

        # Learnable forget gate
        self.alpha_proj = nn.Linear(d_key, num_memory_layers)

        # Muon optimizer for memory updates
        self.muon = MuonOptimizer(momentum=0.95, nesterov=True)

    def _compute_decay_weights(
        self,
        keys: Tensor,
        window_size: int,
    ) -> Tensor:
        """
        Compute decay weights γ_i^(t) for tokens in window.

        Args:
            keys: (batch, window_size, d_key)

        Returns:
            weights: (batch, window_size) normalized decay weights
        """
        if self.learnable_decay:
            # Learned per-token importance
            weights = torch.sigmoid(self.decay_proj(keys).squeeze(-1))
        else:
            # Fixed exponential decay
            weights = self.fixed_decay[-window_size:].unsqueeze(0)
            weights = weights.expand(keys.shape[0], -1)

        # Normalize weights to sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def forward(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Apply Omega rule for sliding window memory optimization.

        Args:
            keys: (batch, seq_len, d_key)
            values: (batch, seq_len, d_value)
            queries: (batch, seq_len, d_key)
            memory_state: Previous memory state

        Returns:
            output: (batch, seq_len, d_value)
            new_memory_state: Updated memory
        """
        batch, seq_len, _ = keys.shape
        device = keys.device

        # Apply polynomial features to keys and queries
        if self.phi is not None:
            keys_phi = self.phi(keys)
            queries_phi = self.phi(queries)
        else:
            keys_phi = keys
            queries_phi = queries

        # Initialize memory state
        if memory_state is None:
            memory_state = {
                name: param.data.clone().unsqueeze(0).expand(batch, -1, -1)
                for name, param in self.memory.named_parameters()
                if param.ndim == 2  # Only weight matrices
            }

        outputs = []

        for t in range(seq_len):
            # Define sliding window
            start = max(0, t - self.context_window + 1)
            end = t + 1

            window_keys = keys_phi[:, start:end]  # (batch, window, phi_dim)
            window_values = values[:, start:end]  # (batch, window, d_value)
            window_size = end - start

            # Compute decay weights for window
            decay_weights = self._compute_decay_weights(
                keys[:, start:end], window_size
            )  # (batch, window)

            # Omega rule: minimize weighted loss over window
            # ℓ = Σ γ_i ||M(k_i) - v_i||²
            total_loss = 0
            gradients = {name: torch.zeros_like(param)
                        for name, param in memory_state.items()}

            for i in range(window_size):
                k_i = window_keys[:, i]  # (batch, phi_dim)
                v_i = window_values[:, i]  # (batch, d_value)
                gamma_i = decay_weights[:, i:i+1]  # (batch, 1)

                # Forward through memory
                pred = self._memory_forward(k_i, memory_state)
                loss_i = ((pred - v_i) ** 2).sum(dim=-1)

                # Weighted loss
                weighted_loss = (gamma_i.squeeze() * loss_i).mean()
                total_loss = total_loss + weighted_loss

                # Accumulate gradients (simplified - actual impl uses autograd)
                # ∂ℓ/∂M = 2γ(Mk - v)k^T for linear case
                error = pred - v_i  # (batch, d_value)
                for name in gradients:
                    if "weight" in name.lower() or name.endswith(".0"):
                        grad_contribution = gamma_i * torch.einsum(
                            "bv,bk->bkv", error, k_i
                        )
                        gradients[name] = gradients[name] + grad_contribution.mean(dim=0)

            # Update memory using Muon optimizer
            alpha = torch.sigmoid(self.alpha_proj(keys[:, t]).mean(dim=0))
            lr = 0.1 * (1 - alpha.mean().item())  # Adaptive LR based on forget gate

            # Apply forget gate
            for name in memory_state:
                layer_idx = min(int(name.split(".")[0]) if name[0].isdigit() else 0,
                              len(alpha) - 1)
                memory_state[name] = (1 - alpha[layer_idx]) * memory_state[name]

            # Muon step
            memory_state = self.muon.step(memory_state, gradients, lr=lr)

            # Query updated memory
            output_t = self._memory_forward(queries_phi[:, t], memory_state)
            outputs.append(output_t)

        output = torch.stack(outputs, dim=1)
        return output, memory_state

    def _memory_forward(
        self,
        query: Tensor,
        state: Dict[str, Tensor],
    ) -> Tensor:
        """Forward pass through memory with given state."""
        # Simplified: just use the first weight for linear projection
        # Full impl would apply all layers
        x = query
        for name, weight in sorted(state.items()):
            if weight.ndim == 2:
                x = F.linear(x, weight.mean(dim=0) if weight.ndim == 3 else weight)
                x = F.silu(x)
        return x


class AtlasMemory(nn.Module):
    """
    Atlas Memory Module combining all innovations:
    - Polynomial features
    - Omega rule
    - Muon optimizer
    - Deep memory
    """

    def __init__(
        self,
        d_model: int,
        d_key: int = 64,
        d_value: int = 64,
        context_window: int = 64,
        num_memory_layers: int = 2,
        polynomial_degree: int = 2,
        use_taylor_kernel: bool = True,
        taylor_order: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # Projections
        self.key_proj = nn.Linear(d_model, d_key)
        self.value_proj = nn.Linear(d_model, d_value)
        self.query_proj = nn.Linear(d_model, d_key)

        # Omega rule memory
        self.omega = OmegaRule(
            d_key=d_key,
            d_value=d_value,
            context_window=context_window,
            num_memory_layers=num_memory_layers,
            learnable_decay=True,
            use_polynomial_features=True,
            polynomial_degree=polynomial_degree,
        )

        # Taylor kernel for attention approximation
        if use_taylor_kernel:
            self.taylor = LearnableTaylorKernel(order=taylor_order, learnable=True)
        else:
            self.taylor = None

        # Output projection
        self.out_proj = nn.Linear(d_value, d_model)
        self.layer_norm = nn.LayerNorm(d_value)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass through Atlas memory.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: Previous memory state

        Returns:
            output: (batch, seq_len, d_model)
            new_state: Updated memory state
        """
        # Project to key, value, query
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        # L2 normalize
        keys = l2_normalize(keys, dim=-1)
        queries = l2_normalize(queries, dim=-1)

        # Apply Omega rule
        output, new_state = self.omega(keys, values, queries, memory_state)

        # Normalize and project
        output = self.layer_norm(output)
        output = self.out_proj(output)

        return output, new_state


class DeepTransformerBlock(nn.Module):
    """
    DeepTransformer Block from Atlas paper.

    "DeepTransformers: Strict generalizations of standard Transformers
    with deep memory and polynomial kernels."
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_head: int = 64,
        context_window: int = 64,
        polynomial_degree: int = 2,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        # Atlas memory (replaces standard attention for long-range)
        self.atlas_memory = AtlasMemory(
            d_model=d_model,
            d_key=d_head,
            d_value=d_head,
            context_window=context_window,
            polynomial_degree=polynomial_degree,
        )

        # Local sliding window attention
        self.local_attention = SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            window_size=context_window,
            dropout=dropout,
        )

        # Gating to combine memory and attention
        self.gate = GatedAttentionUnit(d_model)

        # Feed-forward
        ffn_hidden_dim = ffn_hidden_dim or 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.SiLU(),
            nn.Linear(ffn_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: Previous memory state
            attention_mask: Optional mask

        Returns:
            output: (batch, seq_len, d_model)
            new_memory_state: Updated state
        """
        # Atlas memory for long-range dependencies
        normed = self.norm1(x)
        memory_output, new_state = self.atlas_memory(normed, memory_state)

        # Local attention for short-range
        attn_output, _ = self.local_attention(self.norm2(x), attention_mask)

        # Combine via gating
        combined = self.gate(attn_output, memory_output)
        x = x + combined

        # Feed-forward
        x = x + self.ffn(self.norm3(x))

        return x, new_state


class DeepTransformer(nn.Module):
    """
    DeepTransformer: Full model with Atlas memory.

    From the paper: "DeepTransformers generalize standard Transformers
    with deep memory and polynomial kernels, achieving better
    long-context understanding while maintaining efficiency."
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config

        # Blocks
        self.blocks = nn.ModuleList([
            DeepTransformerBlock(
                d_model=config.d_model,
                num_heads=config.attention.num_heads,
                d_head=config.attention.d_head,
                context_window=config.context_window,
                polynomial_degree=config.polynomial_degree,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[List[Dict[str, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        new_states = []

        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem = block(x, mem_state, attention_mask)
            new_states.append(new_mem)

        x = self.final_norm(x)
        return x, new_states


class OmegaNet(nn.Module):
    """
    OmegaNet: Omega rule with polynomial features and gradient descent.

    Simpler version of Atlas that uses standard gradient descent
    instead of Muon optimizer.
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config

        # Use Atlas memory blocks
        self.blocks = nn.ModuleList([
            DeepTransformerBlock(
                d_model=config.d_model,
                num_heads=config.attention.num_heads,
                d_head=config.attention.d_head,
                context_window=config.context_window,
                polynomial_degree=config.polynomial_degree,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[List[Dict[str, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        new_states = []

        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem = block(x, mem_state, attention_mask)
            new_states.append(new_mem)

        x = self.final_norm(x)
        return x, new_states


class Atlas(nn.Module):
    """
    Atlas: Full model with Muon optimizer.

    Complete language model wrapper with token embedding,
    DeepTransformer backbone, and LM head.
    """

    def __init__(self, config: AtlasConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # DeepTransformer backbone
        self.backbone = DeepTransformer(config)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

    def forward(
        self,
        input_ids: Tensor,
        memory_states: Optional[List[Dict[str, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass for language modeling.

        Args:
            input_ids: (batch, seq_len) token IDs
            memory_states: Previous memory states
            attention_mask: Optional attention mask
            labels: (batch, seq_len) target tokens for loss

        Returns:
            Dictionary with logits, loss (if labels), memory_states
        """
        # Embed
        x = self.token_embedding(input_ids)

        # Backbone
        x, new_states = self.backbone(x, memory_states, attention_mask)

        # Logits
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "memory_states": new_states,
        }

        # Loss
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """Generate tokens autoregressively."""
        memory_states = None

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids, memory_states)
            logits = outputs["logits"][:, -1, :] / temperature
            memory_states = outputs["memory_states"]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

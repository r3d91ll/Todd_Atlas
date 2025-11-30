"""
Neural Memory Module for Atlas.

Implements the long-term memory that learns to memorize at test time
through gradient-based optimization with surprise metric.

Paper: "Atlas: Learning to Optimally Memorize the Context at Test Time" (arXiv:2505.23735)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from titans_atlas.utils import (
    get_activation,
    l2_normalize,
    parallel_scan,
    DepthwiseConv1d,
)


@dataclass
class MemoryState:
    """State container for neural memory."""
    weights: Dict[str, Tensor]  # Memory parameters
    surprise: Tensor  # Accumulated surprise S_t
    step: int  # Current timestep


class DeepMemory(nn.Module):
    """
    Deep Memory MLP for key-value storage.

    From the paper: "We use deep memory modules (L_M >= 2) rather than
    linear mappings. MLPs with multiple layers provide non-linear
    compression capabilities superior to matrix-valued compression."

    Theorem 1 shows capacity of O(d_k * d_v) for deep memory vs O(d_k)
    for linear memory.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        num_layers: int = 2,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        use_residual: bool = True,
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or 4 * d_value
        self.use_residual = use_residual

        # Build MLP layers
        layers = []
        in_dim = d_key
        for i in range(num_layers):
            out_dim = d_value if i == num_layers - 1 else self.hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(get_activation(activation))
            in_dim = out_dim

        self.mlp = nn.ModuleList(layers)

        # Residual projection if dimensions differ
        if use_residual and d_key != d_value:
            self.residual_proj = nn.Linear(d_key, d_value)
        else:
            self.residual_proj = None

    def forward(self, keys: Tensor) -> Tensor:
        """
        Query the memory with keys.

        Args:
            keys: (batch, seq_len, d_key) or (batch, d_key)

        Returns:
            values: (batch, seq_len, d_value) or (batch, d_value)
        """
        x = keys
        for layer in self.mlp:
            x = layer(x)

        if self.use_residual:
            if self.residual_proj is not None:
                x = x + self.residual_proj(keys)
            elif self.d_key == self.d_value:
                x = x + keys

        return x

    def get_parameters(self) -> Dict[str, Tensor]:
        """Get all memory parameters as a dictionary."""
        return {name: param for name, param in self.named_parameters()}


class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory Module.

    Implements the memory update rule from Titans:
        S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)  (Surprise)
        M_t = (1 - α_t) * M_{t-1} + S_t                (Memory Update)

    Where:
        - S_t: Surprise signal (momentum + gradient)
        - η_t: Momentum coefficient (learned or fixed)
        - θ_t: Learning rate for memory (learned or fixed)
        - α_t: Forget gate (learned or fixed)
        - ℓ: Associative memory loss ||M(k_t) - v_t||²

    The memory learns to store key-value associations at test time.
    """

    def __init__(
        self,
        d_model: int,
        d_key: int = 64,
        d_value: int = 64,
        num_memory_layers: int = 2,
        memory_hidden_dim: Optional[int] = None,
        activation: str = "silu",
        use_momentum: bool = True,
        use_forget_gate: bool = True,
        learnable_lr: bool = True,
        learnable_momentum: bool = True,
        learnable_forget: bool = True,
        use_l2_norm_keys: bool = True,
        use_layer_norm: bool = True,
        use_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.use_momentum = use_momentum
        self.use_forget_gate = use_forget_gate
        self.use_l2_norm_keys = use_l2_norm_keys

        # Projections to key, value, query spaces
        self.key_proj = nn.Linear(d_model, d_key)
        self.value_proj = nn.Linear(d_model, d_value)
        self.query_proj = nn.Linear(d_model, d_key)

        # Optional 1D convolutions after projections (as in paper)
        if use_conv:
            self.key_conv = DepthwiseConv1d(d_key, conv_kernel_size)
            self.value_conv = DepthwiseConv1d(d_value, conv_kernel_size)
            self.query_conv = DepthwiseConv1d(d_key, conv_kernel_size)
        else:
            self.key_conv = self.value_conv = self.query_conv = None

        # Deep memory module
        self.memory = DeepMemory(
            d_key=d_key,
            d_value=d_value,
            num_layers=num_memory_layers,
            hidden_dim=memory_hidden_dim,
            activation=activation,
            use_residual=True,
        )

        # Learned memory update parameters
        # θ_t: learning rate
        if learnable_lr:
            self.lr_proj = nn.Linear(d_model, num_memory_layers)
        else:
            self.register_buffer("fixed_lr", torch.tensor(0.1))
        self.learnable_lr = learnable_lr

        # η_t: momentum coefficient
        if use_momentum and learnable_momentum:
            self.momentum_proj = nn.Linear(d_model, num_memory_layers)
        elif use_momentum:
            self.register_buffer("fixed_momentum", torch.tensor(0.9))
        self.learnable_momentum = learnable_momentum

        # α_t: forget gate
        if use_forget_gate and learnable_forget:
            self.forget_proj = nn.Linear(d_model, num_memory_layers)
        elif use_forget_gate:
            self.register_buffer("fixed_forget", torch.tensor(0.01))
        self.learnable_forget = learnable_forget

        # Layer norm
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(d_value)
        else:
            self.layer_norm = None

        # Output projection
        self.out_proj = nn.Linear(d_value, d_model)

    def _compute_params(
        self,
        x: Tensor
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Compute input-dependent memory update parameters.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            theta: (batch, seq_len, num_layers) - learning rates
            eta: (batch, seq_len, num_layers) - momentum coefficients
            alpha: (batch, seq_len, num_layers) - forget gates
        """
        # Learning rate θ_t
        if self.learnable_lr:
            theta = torch.sigmoid(self.lr_proj(x))  # Bound to [0, 1]
        else:
            theta = self.fixed_lr.expand(x.shape[0], x.shape[1], -1)

        # Momentum η_t
        if self.use_momentum:
            if self.learnable_momentum:
                eta = torch.sigmoid(self.momentum_proj(x))
            else:
                eta = self.fixed_momentum.expand(x.shape[0], x.shape[1], -1)
        else:
            eta = None

        # Forget gate α_t
        if self.use_forget_gate:
            if self.learnable_forget:
                alpha = torch.sigmoid(self.forget_proj(x))
            else:
                alpha = self.fixed_forget.expand(x.shape[0], x.shape[1], -1)
        else:
            alpha = None

        return theta, eta, alpha

    def _compute_surprise(
        self,
        memory_state: Dict[str, Tensor],
        keys: Tensor,
        values: Tensor,
        theta: Tensor,
        eta: Optional[Tensor],
        prev_surprise: Optional[Dict[str, Tensor]],
    ) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Compute surprise signal S_t for memory update.

        S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)

        Where ℓ(M; x_t) = ||M(k_t) - v_t||²

        Args:
            memory_state: Current memory parameters
            keys: (batch, d_key) - current keys
            values: (batch, d_value) - target values
            theta: (batch, num_layers) - learning rates
            eta: (batch, num_layers) - momentum coefficients
            prev_surprise: Previous surprise for each parameter

        Returns:
            surprise: Dictionary of surprise tensors per parameter
            loss: Associative memory loss
        """
        # Temporarily set memory parameters
        original_params = {}
        for name, param in self.memory.named_parameters():
            original_params[name] = param.data.clone()
            if name in memory_state:
                param.data = memory_state[name]

        # Compute associative memory loss
        # ℓ(M; k_t, v_t) = ||M(k_t) - v_t||²
        predicted = self.memory(keys)
        loss = F.mse_loss(predicted, values, reduction="none").sum(dim=-1)

        # Compute gradients w.r.t. memory parameters
        gradients = {}
        for name, param in self.memory.named_parameters():
            if param.requires_grad:
                grad = torch.autograd.grad(
                    loss.sum(),
                    param,
                    create_graph=self.training,
                    retain_graph=True,
                )[0]
                gradients[name] = grad

        # Restore original parameters
        for name, param in self.memory.named_parameters():
            param.data = original_params[name]

        # Compute surprise: S_t = η_t * S_{t-1} - θ_t * ∇ℓ
        surprise = {}
        for i, (name, grad) in enumerate(gradients.items()):
            # Get layer-specific theta (learning rate)
            layer_idx = min(i, theta.shape[-1] - 1)
            theta_i = theta[..., layer_idx].unsqueeze(-1).unsqueeze(-1)

            # Momentary surprise: -θ_t * ∇ℓ
            momentary = -theta_i * grad

            # Add momentum from previous surprise
            if eta is not None and prev_surprise is not None and name in prev_surprise:
                eta_i = eta[..., layer_idx].unsqueeze(-1).unsqueeze(-1)
                surprise[name] = eta_i * prev_surprise[name] + momentary
            else:
                surprise[name] = momentary

        return surprise, loss.mean()

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Any]] = None,
        return_memory_state: bool = True,
    ) -> Tuple[Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass through neural memory.

        Args:
            x: (batch, seq_len, d_model) - input sequence
            memory_state: Previous memory state (for recurrent processing)
            return_memory_state: Whether to return updated memory state

        Returns:
            output: (batch, seq_len, d_model) - memory-enhanced output
            new_state: Updated memory state (if return_memory_state=True)
        """
        batch, seq_len, _ = x.shape

        # Project to key, value, query spaces
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        # Apply convolutions if present
        if self.key_conv is not None:
            keys = self.key_conv(keys)
            values = self.value_conv(values)
            queries = self.query_conv(queries)

        # L2 normalize keys and queries
        if self.use_l2_norm_keys:
            keys = l2_normalize(keys, dim=-1)
            queries = l2_normalize(queries, dim=-1)

        # Compute input-dependent parameters
        theta, eta, alpha = self._compute_params(x)

        # Initialize memory state if not provided
        if memory_state is None:
            memory_state = {
                "weights": {name: param.data.clone()
                           for name, param in self.memory.named_parameters()},
                "surprise": None,
            }

        # Process sequence
        outputs = []
        current_weights = memory_state["weights"]
        current_surprise = memory_state.get("surprise")

        for t in range(seq_len):
            k_t = keys[:, t]  # (batch, d_key)
            v_t = values[:, t]  # (batch, d_value)
            q_t = queries[:, t]  # (batch, d_key)
            theta_t = theta[:, t]  # (batch, num_layers)
            eta_t = eta[:, t] if eta is not None else None
            alpha_t = alpha[:, t] if alpha is not None else None

            # Query memory with current query
            # Temporarily load weights
            for name, param in self.memory.named_parameters():
                param.data = current_weights[name].mean(dim=0)  # Average across batch

            memory_output = self.memory(q_t)  # (batch, d_value)

            # Compute surprise signal
            surprise, _ = self._compute_surprise(
                current_weights,
                k_t,
                v_t,
                theta_t,
                eta_t,
                current_surprise,
            )

            # Update memory: M_t = (1 - α_t) * M_{t-1} + S_t
            new_weights = {}
            for name in current_weights:
                if alpha_t is not None:
                    # Get layer-specific alpha
                    layer_idx = list(current_weights.keys()).index(name)
                    layer_idx = min(layer_idx, alpha_t.shape[-1] - 1)
                    alpha_i = alpha_t[..., layer_idx].unsqueeze(-1).unsqueeze(-1)
                    decay = 1 - alpha_i
                else:
                    decay = 1.0

                new_weights[name] = decay * current_weights[name] + surprise[name]

            current_weights = new_weights
            current_surprise = surprise
            outputs.append(memory_output)

        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, seq_len, d_value)

        # Apply layer norm and output projection
        if self.layer_norm is not None:
            output = self.layer_norm(output)
        output = self.out_proj(output)

        # Prepare return state
        if return_memory_state:
            new_state = {
                "weights": current_weights,
                "surprise": current_surprise,
            }
            return output, new_state
        return output, None


class NeuralMemoryParallel(nn.Module):
    """
    Parallelized Neural Memory using tensorized mini-batch gradient descent.

    From the paper: "We tensorize mini-batch gradient descent to use
    matrix multiplications efficiently."

    This version processes chunks in parallel for efficient training.
    """

    def __init__(
        self,
        d_model: int,
        d_key: int = 64,
        d_value: int = 64,
        chunk_size: int = 64,
        num_memory_layers: int = 2,
        memory_hidden_dim: Optional[int] = None,
        activation: str = "silu",
        use_momentum: bool = True,
        use_forget_gate: bool = True,
        use_l2_norm_keys: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value
        self.chunk_size = chunk_size
        self.use_momentum = use_momentum
        self.use_forget_gate = use_forget_gate
        self.use_l2_norm_keys = use_l2_norm_keys

        # Projections
        self.key_proj = nn.Linear(d_model, d_key)
        self.value_proj = nn.Linear(d_model, d_value)
        self.query_proj = nn.Linear(d_model, d_key)

        # For linear memory, we can express updates as matrix operations
        # M_t = β_t * M_0 - Σ θ_i * (β_t/β_i) * ∇ℓ(M_{t'}; x_i)
        # where β_i = Π_{j=1}^{i} (1 - α_j)

        # Learnable initial memory state
        self.M_init = nn.Parameter(torch.zeros(d_key, d_value))

        # Chunk-level parameters (more efficient than token-level)
        self.theta = nn.Parameter(torch.ones(1) * 0.1)  # Learning rate
        if use_momentum:
            self.eta = nn.Parameter(torch.ones(1) * 0.9)  # Momentum
        if use_forget_gate:
            self.alpha = nn.Parameter(torch.ones(1) * 0.01)  # Forget

        # Output projection
        self.out_proj = nn.Linear(d_value, d_model)
        self.layer_norm = nn.LayerNorm(d_value)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Parallel forward pass.

        For linear memory M (matrix), the gradient of ||Mk - v||² w.r.t. M is:
            ∇_M ℓ = 2(Mk - v)kᵀ

        This allows us to express memory updates as matrix products.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: (batch, d_key, d_value) previous memory

        Returns:
            output: (batch, seq_len, d_model)
            new_memory: (batch, d_key, d_value)
        """
        batch, seq_len, _ = x.shape

        # Project to key, value, query
        keys = self.key_proj(x)  # (batch, seq_len, d_key)
        values = self.value_proj(x)  # (batch, seq_len, d_value)
        queries = self.query_proj(x)  # (batch, seq_len, d_key)

        # L2 normalize
        if self.use_l2_norm_keys:
            keys = l2_normalize(keys, dim=-1)
            queries = l2_normalize(queries, dim=-1)

        # Initialize memory
        if memory_state is None:
            M = self.M_init.unsqueeze(0).expand(batch, -1, -1)
        else:
            M = memory_state

        # Compute decay factors
        if self.use_forget_gate:
            alpha = torch.sigmoid(self.alpha)
            # β_t = (1-α)^t cumulative decay
            t = torch.arange(seq_len, device=x.device, dtype=x.dtype)
            beta = (1 - alpha) ** t  # (seq_len,)
        else:
            beta = torch.ones(seq_len, device=x.device, dtype=x.dtype)

        # Compute all predictions: M @ k_i for each timestep
        # This is a simplification - full implementation would track M evolution
        predictions = torch.einsum("bkv,bsk->bsv", M, keys)  # (batch, seq_len, d_value)

        # Compute errors
        errors = predictions - values  # (batch, seq_len, d_value)

        # Compute gradients: ∇_M = error @ key^T
        # For parallel processing, we accumulate weighted gradients
        theta = torch.sigmoid(self.theta)

        # Weighted gradient accumulation
        weights = theta * beta.unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        weighted_errors = errors * weights

        # Gradient update: ΔM = Σ weighted_error_i @ key_i^T
        delta_M = torch.einsum("bsv,bsk->bkv", weighted_errors, keys)

        # Apply momentum if enabled
        if self.use_momentum:
            eta = torch.sigmoid(self.eta)
            # For simplicity, apply momentum as scaling
            delta_M = eta * delta_M

        # Update memory
        if self.use_forget_gate:
            M_new = beta[-1] * M - delta_M
        else:
            M_new = M - delta_M

        # Query final memory for output
        output = torch.einsum("bkv,bsk->bsv", M_new, queries)

        # Normalize and project
        output = self.layer_norm(output)
        output = self.out_proj(output)

        return output, M_new

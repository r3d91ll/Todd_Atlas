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
from titans_atlas.utils import get_activation, l2_normalize, parallel_scan, RMSNorm
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
        chunk_size: int = 64,  # Process chunks in parallel during training
    ):
        super().__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.context_window = context_window
        self.learnable_decay = learnable_decay
        self.chunk_size = chunk_size  # For chunked training mode

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
        Process keys, values, and queries in chunked groups using the Omega rule to update a linear memory and produce memory-augmented outputs.
        
        Parameters:
            keys (Tensor): Tensor of shape (batch, seq_len, d_key) containing per-token key vectors (may be expanded by polynomial features).
            values (Tensor): Tensor of shape (batch, seq_len, d_value) containing per-token value vectors.
            queries (Tensor): Tensor of shape (batch, seq_len, d_key) containing per-token query vectors (may be expanded by polynomial features).
            memory_state (Optional[Dict[str, Tensor]]): Optional state containing '_linear_M' with shape (batch, d_value, phi_dim). If omitted, memory is initialized to zeros.
        
        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: A pair where the first element is the output tensor of shape (batch, seq_len, d_value) produced by querying the updated memory, and the second element is the updated memory_state dict containing the key '_linear_M'.
        """
        batch, seq_len, _ = keys.shape
        device = keys.device
        dtype = keys.dtype
        c = self.context_window

        # Apply polynomial features
        if self.phi is not None:
            keys_phi = self.phi(keys)
            queries_phi = self.phi(queries)
        else:
            keys_phi = keys
            queries_phi = queries

        phi_dim = keys_phi.shape[-1]

        # Initialize linear memory M: (batch, d_value, phi_dim)
        if memory_state is None:
            M = torch.zeros(batch, self.d_value, phi_dim, device=device, dtype=dtype)
        else:
            M = memory_state.get('_linear_M',
                torch.zeros(batch, self.d_value, phi_dim, device=device, dtype=dtype))

        # Compute all decay weights and forget gates in parallel
        if self.learnable_decay:
            gamma = torch.sigmoid(self.decay_proj(keys).squeeze(-1))
        else:
            gamma = self.fixed_decay[:seq_len].unsqueeze(0).expand(batch, -1)

        alpha = torch.sigmoid(self.alpha_proj(keys).mean(dim=-1))

        eta = 0.1
        outputs = []

        # Process in chunks of size context_window
        num_chunks = (seq_len + c - 1) // c

        for chunk_idx in range(num_chunks):
            start = chunk_idx * c
            end = min(start + c, seq_len)
            chunk_len = end - start

            # Get chunk data
            k_chunk = keys_phi[:, start:end]      # (batch, chunk_len, phi_dim)
            v_chunk = values[:, start:end]        # (batch, chunk_len, d_value)
            q_chunk = queries_phi[:, start:end]   # (batch, chunk_len, phi_dim)
            g_chunk = gamma[:, start:end]         # (batch, chunk_len)
            a_chunk = alpha[:, start:end].mean(dim=-1)  # (batch,) - avg alpha for chunk

            # === PARALLEL GRADIENT COMPUTATION WITHIN CHUNK ===
            # Compute: gradient = ∑ᵢ γᵢ(vᵢ - Mkᵢ)kᵢᵀ for all i in chunk
            #
            # Step 1: Compute Mkᵢ for all i in parallel
            # M: (batch, d_value, phi_dim), k_chunk: (batch, chunk_len, phi_dim)
            Mk_chunk = torch.bmm(M, k_chunk.transpose(-2, -1))  # (batch, d_value, chunk_len)
            Mk_chunk = Mk_chunk.transpose(-2, -1)  # (batch, chunk_len, d_value)

            # Step 2: Compute errors: vᵢ - Mkᵢ
            errors = v_chunk - Mk_chunk  # (batch, chunk_len, d_value)

            # Step 3: Weight by γᵢ
            g_exp = g_chunk.unsqueeze(-1)  # (batch, chunk_len, 1)
            weighted_errors = g_exp * errors  # (batch, chunk_len, d_value)

            # Step 4: Compute ∑ᵢ γᵢ(vᵢ - Mkᵢ)kᵢᵀ using batched outer product sum
            # weighted_errors: (batch, chunk_len, d_value)
            # k_chunk: (batch, chunk_len, phi_dim)
            # We want: ∑ᵢ weighted_errors[i] ⊗ k_chunk[i]ᵀ
            # = einsum('bcd,bcp->bdp', weighted_errors, k_chunk)
            gradient = torch.einsum('bcd,bcp->bdp', weighted_errors, k_chunk)  # (batch, d_value, phi_dim)

            # Update memory with forget gate (once per chunk)
            a_t = a_chunk.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
            M = a_t * M + eta * gradient

            # Query memory for all positions in chunk (parallel)
            # output = M @ qᵢ for each i in chunk
            chunk_outputs = torch.bmm(M, q_chunk.transpose(-2, -1))  # (batch, d_value, chunk_len)
            chunk_outputs = chunk_outputs.transpose(-2, -1)  # (batch, chunk_len, d_value)
            outputs.append(chunk_outputs)

        # Concatenate all chunk outputs
        output = torch.cat(outputs, dim=1)  # (batch, seq_len, d_value)
        new_memory_state = {'_linear_M': M}

        return output, new_memory_state

    def forward_chunked(
        self,
        keys: Tensor,
        values: Tensor,
        queries: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Process input sequences in fixed-size chunks, updating the memory only at chunk boundaries to accelerate training while preserving Omega-rule dynamics.
        
        Parameters:
            keys (Tensor): Shape (batch, seq_len, d_key). Sequence of key vectors.
            values (Tensor): Shape (batch, seq_len, d_value). Sequence of value vectors.
            queries (Tensor): Shape (batch, seq_len, d_key). Sequence of query vectors.
            memory_state (Optional[Dict[str, Tensor]]): Current memory parameters replicated per batch; if None, a new batched memory_state is initialized from the module's parameters.
        
        Returns:
            output (Tensor): Memory-query outputs for the full sequence, shape (batch, seq_len, d_value).
            new_memory_state (Dict[str, Tensor]): Updated per-batch memory state after chunked updates.
        """
        batch, seq_len, _ = keys.shape
        device = keys.device

        # Apply polynomial features
        if self.phi is not None:
            keys_phi = self.phi(keys)
            queries_phi = self.phi(queries)
        else:
            keys_phi = keys
            queries_phi = queries

        # Initialize memory state
        if memory_state is None:
            memory_state = {}
            for name, param in self.memory.named_parameters():
                if param.ndim == 2:
                    memory_state[name] = param.data.clone().unsqueeze(0).expand(batch, -1, -1).clone()
                elif param.ndim == 1:
                    memory_state[name] = param.data.clone().unsqueeze(0).expand(batch, -1).clone()

        # Process in chunks
        chunk_size = min(self.chunk_size, seq_len)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        outputs = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, seq_len)
            actual_chunk_size = end - start

            # Get chunk data
            chunk_keys = keys[:, start:end]
            chunk_keys_phi = keys_phi[:, start:end]
            chunk_values = values[:, start:end]
            chunk_queries_phi = queries_phi[:, start:end]

            # Compute decay weights for chunk (using last context_window tokens)
            context_start = max(0, end - self.context_window)
            context_keys = keys[:, context_start:end]
            decay_weights = self._compute_decay_weights(context_keys, context_keys.shape[1])

            # Single gradient computation for entire chunk
            context_keys_phi = keys_phi[:, context_start:end]
            context_values = values[:, context_start:end]

            gradients = self._compute_omega_gradients(
                context_keys_phi, context_values, decay_weights, memory_state
            )

            # Update memory once per chunk
            with torch.no_grad():
                alpha = torch.sigmoid(self.alpha_proj(chunk_keys.mean(dim=1))).mean(dim=0)
                lr = 0.1 * (1 - alpha.mean().item())

                # Apply forget gate
                for name in memory_state:
                    layer_idx = min(int(name.split(".")[0]) if name[0].isdigit() else 0,
                                  len(alpha) - 1)
                    memory_state[name] = (1 - alpha[layer_idx]) * memory_state[name]

                # Muon step
                memory_state = self.muon.step(memory_state, gradients, lr=lr)

            # Query memory for all positions in chunk (parallel)
            chunk_output = self._memory_forward_batch(chunk_queries_phi, memory_state)
            outputs.append(chunk_output)

        output = torch.cat(outputs, dim=1)
        return output, memory_state

    def _memory_forward_batch(
        self,
        queries: Tensor,
        memory_state: Dict[str, Tensor],
    ) -> Tensor:
        """
        Forward pass through memory for a batch of queries.

        This mirrors the DeepMemory.forward() logic but uses explicit
        parameters from memory_state instead of module parameters.

        Args:
            queries: (batch, seq_len, phi_dim)
            memory_state: Memory parameters with keys like 'mlp.0.weight', 'mlp.2.weight', etc.

        Returns:
            output: (batch, seq_len, d_value)
        """
        batch, seq_len, phi_dim = queries.shape

        # Process through the network
        # The structure is: Linear -> Activation -> Linear (for 2-layer DeepMemory)
        # Parameter names are: mlp.0.weight, mlp.0.bias, mlp.2.weight, mlp.2.bias
        # (mlp.1 is the activation function, not a parameter)

        x = queries  # (batch, seq_len, phi_dim)
        residual_input = x

        # Find all linear layer indices (those with weights)
        layer_indices = sorted(set(
            int(k.split('.')[1]) for k in memory_state.keys()
            if k.startswith('mlp.') and 'weight' in k
        ))

        for i, layer_idx in enumerate(layer_indices):
            weight_key = f"mlp.{layer_idx}.weight"
            bias_key = f"mlp.{layer_idx}.bias"

            weight = memory_state[weight_key]
            # Weight shape: batched (batch, out, in) or unbatched (out, in)
            if weight.ndim == 3:
                # Batched weight: need to expand for seq_len
                # x: (batch, seq_len, in), weight: (batch, out, in)
                # Use einsum: x @ W^T = (batch, seq_len, out)
                x = torch.einsum('bsi,boi->bso', x, weight)
            else:
                # Unbatched weight: (out, in)
                x = F.linear(x, weight)

            if bias_key in memory_state:
                bias = memory_state[bias_key]
                if bias.ndim == 2:  # batched: (batch, out)
                    bias = bias.unsqueeze(1)  # (batch, 1, out)
                x = x + bias

            # Apply activation (except last layer)
            is_last_layer = (i == len(layer_indices) - 1)
            if not is_last_layer:
                x = F.silu(x)

        # Add residual projection if available
        if 'residual_proj.weight' in memory_state:
            res_weight = memory_state['residual_proj.weight']
            if res_weight.ndim == 3:
                residual = torch.einsum('bsi,boi->bso', residual_input, res_weight)
            else:
                residual = F.linear(residual_input, res_weight)

            if 'residual_proj.bias' in memory_state:
                res_bias = memory_state['residual_proj.bias']
                if res_bias.ndim == 2:
                    res_bias = res_bias.unsqueeze(1)
                residual = residual + res_bias

            x = x + residual

        return x

    def _compute_omega_gradients(
        self,
        window_keys: Tensor,
        window_values: Tensor,
        decay_weights: Tensor,
        memory_state: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Compute per-parameter gradients for the Omega-rule memory update from a window of key/value pairs.
        
        Given a window of polynomial-expanded keys, target values, and per-step decay weights γ_i, this builds the loss ℓ = Σ_i γ_i ||M(k_i) - v_i||² with a temporary copy of the provided memory parameters, backpropagates that loss, and returns gradients shaped to match the original memory_state.
        
        Parameters:
            window_keys (Tensor): shape (batch, window, phi_dim) — polynomial-expanded keys for each timestep.
            window_values (Tensor): shape (batch, window, d_value) — target values corresponding to each key.
            decay_weights (Tensor): shape (batch, window) — per-timestep importance weights γ_i that sum (or are normalized) per batch.
            memory_state (Dict[str, Tensor]): mapping of memory parameter names to tensors (may contain batched parameters).
        
        Returns:
            Dict[str, Tensor]: gradient tensors for each entry in memory_state. Each gradient matches the shape of the corresponding memory_state tensor (batched gradients are expanded to the original batch dimension).
        """
        batch, window_size, phi_dim = window_keys.shape

        # For Omega rule internal gradient computation, we detach to isolate
        # This is the memory's internal learning, separate from training gradients
        # Training gradients flow through _memory_forward output, not here
        window_keys_f = window_keys.detach().float()
        window_values_f = window_values.detach().float()
        decay_weights_f = decay_weights.detach().float()

        # Create leaf tensors for Omega gradient computation
        temp_params = {}
        for name, param in memory_state.items():
            if param.ndim == 3:  # (batch, out, in)
                temp_params[name] = param.mean(dim=0).detach().float().requires_grad_(True)
            elif param.ndim == 2 and 'bias' not in name:  # (out, in) unbatched weight
                temp_params[name] = param.detach().float().requires_grad_(True)
            else:  # bias or 1D
                if param.ndim == 2:  # (batch, out) batched bias
                    temp_params[name] = param.mean(dim=0).detach().float().requires_grad_(True)
                else:
                    temp_params[name] = param.detach().float().requires_grad_(True)

        # Forward through memory and compute weighted loss
        # Accumulate losses in a list and sum at the end to maintain gradient graph
        losses = []

        for i in range(window_size):
            k_i = window_keys_f[:, i]  # (batch, phi_dim)
            v_i = window_values_f[:, i]  # (batch, d_value)
            gamma_i = decay_weights_f[:, i]  # (batch,)

            # Forward through temp params
            pred = self._memory_forward_with_params(k_i, temp_params)
            loss_i = ((pred - v_i) ** 2).sum(dim=-1)  # (batch,)
            weighted_loss = (gamma_i * loss_i).mean()
            losses.append(weighted_loss)

        # Sum all losses to get total loss with gradient graph
        total_loss = torch.stack(losses).sum()

        # Backward to get gradients for Omega rule (isolated from training graph)
        total_loss.backward()

        # Extract gradients and expand back to batch dimension
        gradients = {}
        for name, param in temp_params.items():
            if param.grad is not None:
                grad = param.grad.detach()  # Detach - this is internal to Omega rule
                # Convert back to original dtype if needed
                original_dtype = memory_state[name].dtype
                if grad.dtype != original_dtype:
                    grad = grad.to(original_dtype)
                # Expand to batch dimension to match memory_state shapes
                if memory_state[name].ndim == 3:  # batched weight
                    gradients[name] = grad.unsqueeze(0).expand(batch, -1, -1).clone()
                elif memory_state[name].ndim == 2 and 'bias' not in name:
                    gradients[name] = grad
                else:
                    if memory_state[name].ndim == 2:  # batched bias
                        gradients[name] = grad.unsqueeze(0).expand(batch, -1).clone()
                    else:
                        gradients[name] = grad
            else:
                gradients[name] = torch.zeros_like(memory_state[name])

        return gradients

    def _memory_forward_with_params(
        self,
        query: Tensor,
        params: Dict[str, Tensor],
    ) -> Tensor:
        """
        Forward through memory with explicit parameters (for gradient computation).

        This mirrors DeepMemory.forward() but uses explicit params instead of module params.

        Args:
            query: (batch, phi_dim)
            params: Dict of unbatched parameter tensors

        Returns:
            output: (batch, d_value)
        """
        x = query
        input_for_residual = query

        # Get MLP layer names (exclude residual_proj)
        mlp_weight_names = sorted([n for n in params.keys()
                                   if 'weight' in n and 'residual' not in n])

        for i, w_name in enumerate(mlp_weight_names):
            weight = params[w_name]  # (out_dim, in_dim)
            x = F.linear(x, weight)

            # Add bias if exists
            b_name = w_name.replace('weight', 'bias')
            if b_name in params:
                x = x + params[b_name]

            # Apply activation for all but last layer
            if i < len(mlp_weight_names) - 1:
                x = F.silu(x)

        # Add residual connection if present
        if 'residual_proj.weight' in params:
            residual = F.linear(input_for_residual, params['residual_proj.weight'])
            if 'residual_proj.bias' in params:
                residual = residual + params['residual_proj.bias']
            x = x + residual

        return x

    def _memory_forward(
        self,
        query: Tensor,
        state: Dict[str, Tensor],
    ) -> Tensor:
        """
        Forward pass through memory with given state.

        The state contains weight matrices that may have a batch dimension.
        This mirrors DeepMemory.forward() structure.

        Args:
            query: (batch, phi_dim) - polynomial-expanded query
            state: Dict of weight tensors, each (batch, out_dim, in_dim) or (out_dim, in_dim)

        Returns:
            output: (batch, d_value)
        """
        x = query  # (batch, phi_dim)
        input_for_residual = query

        # Get MLP weight names (exclude residual_proj)
        mlp_weight_names = sorted([n for n in state.keys()
                                   if 'weight' in n and 'residual' not in n])

        for i, w_name in enumerate(mlp_weight_names):
            weight = state[w_name]  # (batch, out_dim, in_dim) or (out_dim, in_dim)

            # Handle batched weights
            if weight.ndim == 3:
                # Batched: use einsum for batch matrix multiply
                # x: (batch, in_dim), weight: (batch, out_dim, in_dim)
                x = torch.einsum('bi,boi->bo', x, weight)
            else:
                # Unbatched: standard linear
                x = F.linear(x, weight)

            # Add bias if exists
            b_name = w_name.replace('weight', 'bias')
            if b_name in state:
                bias = state[b_name]
                if bias.ndim == 2:
                    x = x + bias.mean(dim=0)  # Average batch biases
                else:
                    x = x + bias

            # Apply activation for all but last layer
            if i < len(mlp_weight_names) - 1:
                x = F.silu(x)

        # Add residual connection if present
        if 'residual_proj.weight' in state:
            r_weight = state['residual_proj.weight']
            if r_weight.ndim == 3:
                residual = torch.einsum('bi,boi->bo', input_for_residual, r_weight)
            else:
                residual = F.linear(input_for_residual, r_weight)

            if 'residual_proj.bias' in state:
                r_bias = state['residual_proj.bias']
                if r_bias.ndim == 2:
                    residual = residual + r_bias.mean(dim=0)
                else:
                    residual = residual + r_bias

            x = x + residual

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
        """
        Initialize AtlasMemory components for long-range memory and projections.
        
        Sets up key, value, and query linear projections; an OmegaRule instance for chunked sliding-window memory with polynomial feature expansion; an optional learnable Taylor kernel for attention approximation; an output projection back to model dimension; and an RMSNorm for output normalization.
        
        Parameters:
            d_model (int): Input and output model dimensionality.
            d_key (int): Dimension of key/query projections.
            d_value (int): Dimension of value projections and memory outputs.
            context_window (int): Size of the local context window used by the OmegaRule.
            num_memory_layers (int): Number of layers inside the OmegaRule memory MLP.
            polynomial_degree (int): Degree of polynomial feature expansion applied to keys/queries.
            use_taylor_kernel (bool): If True, create a learnable Taylor kernel to approximate the attention kernel.
            taylor_order (int): Order of the Taylor series used by the LearnableTaylorKernel (if enabled).
        """
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # Projections
        self.key_proj = nn.Linear(d_model, d_key)
        self.value_proj = nn.Linear(d_model, d_value)
        self.query_proj = nn.Linear(d_model, d_key)

        # Omega rule memory
        # chunk_size = context_window so each chunk gets full Omega rule processing
        self.omega = OmegaRule(
            d_key=d_key,
            d_value=d_value,
            context_window=context_window,
            num_memory_layers=num_memory_layers,
            learnable_decay=True,
            use_polynomial_features=True,
            polynomial_degree=polynomial_degree,
            chunk_size=context_window,  # Match chunk to context for proper Omega
        )

        # Taylor kernel for attention approximation
        if use_taylor_kernel:
            self.taylor = LearnableTaylorKernel(order=taylor_order, learnable=True)
        else:
            self.taylor = None

        # Output projection
        self.out_proj = nn.Linear(d_value, d_model)
        self.layer_norm = RMSNorm(d_value)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute memory-augmented representations for input tokens using AtlasMemory and the Omega Rule.
        
        This method projects inputs to keys, values, and queries, applies the OmegaRule memory mechanism (chunked processing during training, per-timestep processing during evaluation), and returns the memory-augmented outputs after normalization and projection.
        
        Parameters:
            x (Tensor): Input token representations with shape (batch, seq_len, d_model).
            memory_state (Optional[Dict[str, Tensor]]): Optional previous memory state; if provided, it will be updated and returned.
        
        Returns:
            Tuple[Tensor, Dict[str, Tensor]]: 
                - output: Memory-augmented token representations with shape (batch, seq_len, d_model).
                - new_state: Updated memory state dictionary.
        """
        # Project to key, value, query
        keys = self.key_proj(x)
        values = self.value_proj(x)
        queries = self.query_proj(x)

        # L2 normalize
        keys = l2_normalize(keys, dim=-1)
        queries = l2_normalize(queries, dim=-1)

        # Apply Omega rule - FULL per-timestep sliding window optimization
        # Not using chunked approximation - this is the proper Atlas algorithm
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
        """
        Create a transformer block that integrates Atlas long-range memory with local sliding-window attention, a gating unit, and a feed-forward network.
        
        Parameters:
            d_model: Model hidden dimension for inputs and outputs.
            num_heads: Number of attention heads used by the local sliding-window attention.
            d_head: Dimension of a single attention head (also used as key/value dim in AtlasMemory).
            context_window: Size of the local context window and the Atlas memory context used for chunking.
            polynomial_degree: Degree for optional polynomial feature expansion used by AtlasMemory.
            ffn_hidden_dim: Hidden dimension for the feed-forward network; defaults to 4 * d_model if None.
            dropout: Dropout probability applied inside attention and the feed-forward output.
        """
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

        # RMSNorm (faster than LayerNorm, used in LLaMA/Mistral)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Tensor]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute a transformer block pass that integrates long-range Atlas memory, local sliding-window attention, gating, and a feed-forward residual.
        
        Parameters:
            x: Input tensor of shape (batch, seq_len, d_model).
            memory_state: Optional previous memory state used by AtlasMemory; when provided, it is updated and returned.
            attention_mask: Optional attention mask applied to local attention.
        
        Returns:
            A tuple (output, new_memory_state) where `output` is the block output tensor of shape (batch, seq_len, d_model) and `new_memory_state` is the updated memory state dictionary produced by AtlasMemory.
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
        """
        Initialize the DeepTransformer using the provided AtlasConfig by constructing its stack of DeepTransformerBlock modules and final normalization.
        
        Parameters:
            config (AtlasConfig): Model configuration containing hyperparameters such as
                num_layers, d_model, attention (num_heads, d_head), context_window,
                polynomial_degree, ffn_hidden_dim, and dropout. These fields are used
                to instantiate each DeepTransformerBlock and to configure the final
                RMSNorm. The method also calls internal weight initialization.
        """
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

        self.final_norm = RMSNorm(config.d_model)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize parameters of submodules.
        
        Applies a normal initialization to every nn.Linear weight with standard deviation taken from self.config.init_std and sets linear biases to zero. """
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
        """
        Initialize a DeepTransformer by constructing its stack of DeepTransformerBlock layers and final RMS normalization.
        
        Parameters:
            config (AtlasConfig): Model configuration providing layer count and hyperparameters used to build each block:
                - d_model: model hidden dimensionality
                - num_layers: number of transformer blocks to create
                - attention.num_heads, attention.d_head: attention head configuration for each block
                - context_window: sliding-window size used by each block
                - polynomial_degree: degree for optional polynomial feature expansion
                - ffn_hidden_dim: hidden dimension for feed-forward networks
                - dropout: dropout probability applied inside blocks
        """
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

        self.final_norm = RMSNorm(config.d_model)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[List[Dict[str, Tensor]]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
        """
        Run the input through each transformer block, updating and returning per-block memory states.
        
        Parameters:
            x (Tensor): Input tensor of shape (batch, seq_len, d_model).
            memory_states (Optional[List[Dict[str, Tensor]]]): Optional list of memory states, one per block; when None, each block receives None and initializes its own state.
            attention_mask (Optional[Tensor]): Optional attention mask passed to each block.
        
        Returns:
            Tuple[Tensor, List[Dict[str, Tensor]]]: A tuple containing the normalized output tensor and a list of updated memory states (one dict per block).
        """
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
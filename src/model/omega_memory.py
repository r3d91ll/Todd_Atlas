"""
Atlas Memory with Omega Rule and Polynomial Features.

Implements the proper Atlas update equations from arXiv:2505.23735v1:

Omega Rule Objective (Equation 9):
    min_M Σ(i=t-c+1 to t) γᵢ(t) ||M(ϕ(kᵢ)) - vᵢ||²₂

Update Equations (Equation 11-13):
    Mₜ = αₜMₜ₋₁ + Sₜ
    Sₜ = θₜSₜ₋₁ - ηₜ∇ℓ(Mₜ₋₁; context_window)

Key components:
    - Polynomial feature mapping ϕₚ(k) with coefficients aᵢ = 1/i!
    - Context window optimization (not single-token)
    - Input-dependent gamma gates for selective token inclusion
    - Self-stabilizing update dynamics via ϕ(k)ϕ(k)ᵀ damping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class OmegaMemoryConfig:
    """Configuration for Atlas Omega Memory."""
    d_model: int = 512
    d_key: int = 512
    d_value: int = 512

    # Polynomial features
    poly_degree: int = 2  # Degree p of polynomial features

    # Omega rule parameters
    context_window: int = 16  # c: number of past tokens to consider

    # Initialization
    init_alpha: float = 0.99  # Memory decay (high = more retention)
    init_theta: float = 0.9   # Momentum coefficient
    init_eta: float = 0.1     # Learning rate for memory updates

    # Numerical stability
    eps: float = 1e-6
    max_memory_norm: float = 50.0


class PolynomialFeatures(nn.Module):
    """
    Polynomial feature mapping ϕₚ(x) = [xᵝ]_{|β|≤p}.

    For efficiency, we use a simplified version that computes:
    ϕ(x) = a₀ + a₁x + a₂x² + ... + aₚxᵖ (element-wise)

    With coefficients initialized as aᵢ = 1/i! (Taylor expansion of exp).

    This provides O(d^p) effective capacity while keeping computation tractable.
    """

    def __init__(self, d_input: int, degree: int = 2, learnable: bool = True):
        super().__init__()
        self.d_input = d_input
        self.degree = degree

        # Initialize polynomial coefficients: aᵢ = 1/i!
        # This approximates exp(x) and provides natural scaling
        coeffs = torch.zeros(degree + 1)
        for i in range(degree + 1):
            coeffs[i] = 1.0 / math.factorial(i)

        if learnable:
            self.coeffs = nn.Parameter(coeffs)
        else:
            self.register_buffer("coeffs", coeffs)

        # Output dimension: concatenation of all polynomial terms
        # For degree p: [1, x, x², ..., xᵖ] -> (p+1) * d_input
        self.d_output = (degree + 1) * d_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply polynomial feature mapping.

        Args:
            x: Input tensor [..., d_input]

        Returns:
            ϕ(x): Polynomial features [..., (degree+1) * d_input]
        """
        # Compute powers: x⁰, x¹, x², ..., xᵖ
        powers = []
        x_power = torch.ones_like(x)  # x⁰ = 1

        for i in range(self.degree + 1):
            # Weight by coefficient aᵢ = 1/i!
            powers.append(self.coeffs[i] * x_power)
            x_power = x_power * x  # x^(i+1)

        # Concatenate all polynomial terms
        return torch.cat(powers, dim=-1)

    def get_effective_dimension(self) -> int:
        """Return the output dimension of polynomial features."""
        return self.d_output


class ContextGates(nn.Module):
    """
    Input-dependent context gates γᵢ(t) ∈ [0,1].

    These gates determine how much each token in the context window
    contributes to the memory update. Enables learned sparse context selection.
    """

    def __init__(self, d_model: int, context_window: int):
        super().__init__()
        self.d_model = d_model
        self.context_window = context_window

        # Project current token to gate values for each position in window
        # Uses attention-like mechanism: current token attends to context
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)

        # Temperature for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1))

        # Initialize for full context initially (gates ≈ 1)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)

    def forward(
        self,
        current: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute context gates γᵢ(t) for each position in context window.

        Args:
            current: Current token [batch, d_model]
            context: Context window [batch, c, d_model]

        Returns:
            gates: γᵢ(t) values [batch, c] in range [0, 1]
        """
        batch_size, c, _ = context.shape

        # Query from current token
        q = self.query_proj(current)  # [batch, d_model]

        # Keys from context
        k = self.key_proj(context)  # [batch, c, d_model]

        # Attention scores
        scores = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)  # [batch, c]
        scores = scores / (self.d_model ** 0.5 * self.temperature.clamp(min=0.1))

        # Sigmoid for independent gates (not softmax - we want sparse selection)
        gates = torch.sigmoid(scores)

        return gates


class OmegaMemory(nn.Module):
    """
    Atlas Memory with Omega Rule.

    Implements the full Atlas memory update equations:

    1. Polynomial feature expansion: ϕₚ(k) with aᵢ = 1/i!
    2. Context window optimization: Σ(i=t-c+1 to t) γᵢ ||M·ϕ(kᵢ) - vᵢ||²
    3. Omega rule update with momentum:
        Sₜ = θₜSₜ₋₁ - ηₜ∇ℓ
        Mₜ = αₜMₜ₋₁ + Sₜ

    Key stability mechanisms:
    - Polynomial coefficients 1/i! bound growth
    - ϕ(k)ϕ(k)ᵀ in gradient provides natural damping
    - Input-dependent α, η, θ adapt to content
    """

    def __init__(self, config: OmegaMemoryConfig):
        super().__init__()
        self.config = config

        # Polynomial feature mapping
        self.poly_features = PolynomialFeatures(
            d_input=config.d_key,
            degree=config.poly_degree,
            learnable=True,
        )
        self.d_poly = self.poly_features.get_effective_dimension()

        # Projections
        self.key_proj = nn.Linear(config.d_model, config.d_key, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_value, bias=False)
        self.query_proj = nn.Linear(config.d_model, config.d_key, bias=False)

        # Q-K alignment projection (TNT insight)
        self.qk_proj = nn.Linear(config.d_key, config.d_key, bias=False)

        # Context gates for selective token inclusion
        self.context_gates = ContextGates(config.d_model, config.context_window)

        # Input-dependent parameter generators
        # α: decay (how much old memory to retain)
        self.alpha_proj = nn.Linear(config.d_model, 1, bias=True)
        # θ: momentum coefficient
        self.theta_proj = nn.Linear(config.d_model, 1, bias=True)
        # η: learning rate for memory update
        self.eta_proj = nn.Linear(config.d_model, 1, bias=True)

        # Output projection (from d_value back to d_model)
        self.out_proj = nn.Linear(config.d_value, config.d_model, bias=False)

        # Initialize
        self._init_weights()

    def _logit(self, p: float) -> float:
        """Compute logit (inverse sigmoid) for initialization."""
        p = max(min(p, 0.999), 0.001)
        return math.log(p / (1 - p))

    def _init_weights(self):
        """Initialize weights for stability."""
        # Xavier for projections
        for proj in [self.key_proj, self.value_proj, self.query_proj, self.out_proj]:
            nn.init.xavier_uniform_(proj.weight)

        # Q-K projection starts as identity
        nn.init.eye_(self.qk_proj.weight)

        # Initialize parameter projections with desired initial values
        nn.init.normal_(self.alpha_proj.weight, std=0.01)
        nn.init.normal_(self.theta_proj.weight, std=0.01)
        nn.init.normal_(self.eta_proj.weight, std=0.01)

        # Set biases for initial sigmoid outputs
        with torch.no_grad():
            self.alpha_proj.bias.fill_(self._logit(self.config.init_alpha))
            self.theta_proj.bias.fill_(self._logit(self.config.init_theta))
            self.eta_proj.bias.fill_(self._logit(self.config.init_eta))

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Initialize memory state.

        Returns:
            M: Memory matrix [batch, d_poly, d_value]
            S: Momentum accumulator [batch, d_poly, d_value]
            context_buffer: List of past inputs for context window
        """
        # Memory matrix: maps polynomial features to values
        M = torch.zeros(batch_size, self.d_poly, self.config.d_value,
                       device=device, dtype=dtype)

        # Momentum accumulator (same shape as M)
        S = torch.zeros_like(M)

        # Context buffer for past c tokens
        context_buffer = []

        return M, S, context_buffer

    def reset_state(self) -> None:
        """
        Reset any cached memory state.

        Note: OmegaMemory doesn't maintain persistent internal state -
        state is passed through forward() calls. This method exists
        for API compatibility with episodic training.
        """
        # OmegaMemory is stateless between forward calls
        # (state is passed explicitly), so nothing to reset
        pass

    def get_state(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, List]]:
        """
        Get current memory state.

        Note: OmegaMemory doesn't maintain persistent internal state -
        state is passed through forward() calls. This method exists
        for API compatibility with episodic training.

        Returns:
            None (state is passed explicitly through forward())
        """
        # State is managed externally, not internally
        return None

    def _compute_omega_gradient(
        self,
        M: torch.Tensor,
        context_k: torch.Tensor,
        context_v: torch.Tensor,
        gamma: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradient of Omega rule objective over context window.

        Objective: Σᵢ γᵢ ||M·ϕ(kᵢ) - vᵢ||²

        Gradient: ∇_M = 2 Σᵢ γᵢ (M·ϕ(kᵢ) - vᵢ) ϕ(kᵢ)ᵀ

        This can be written in matrix form as:
        ∇_M = 2 (M·Φᵀ·Γ·Φ - V·Γ·Φ)

        where Φ = [ϕ(k₁), ..., ϕ(kc)]ᵀ and Γ = diag(γ)

        Args:
            M: Memory [batch, d_poly, d_value]
            context_k: Keys [batch, c, d_poly] (already polynomial-expanded)
            context_v: Values [batch, c, d_value]
            gamma: Gates [batch, c]

        Returns:
            grad: Gradient [batch, d_poly, d_value]
            error: Mean prediction error (for metrics)
        """
        batch_size, c, d_poly = context_k.shape

        # Predictions: M @ k for each k in context
        # M: [batch, d_poly, d_value], context_k: [batch, c, d_poly]
        # pred[i] = M @ k[i] -> [batch, c, d_value]
        pred = torch.bmm(context_k, M)  # [batch, c, d_value]

        # Error: pred - v, weighted by gamma
        error = pred - context_v  # [batch, c, d_value]

        # Apply gamma weights: [batch, c, 1] * [batch, c, d_value]
        weighted_error = gamma.unsqueeze(-1) * error  # [batch, c, d_value]

        # Gradient: Σᵢ γᵢ (error_i) @ ϕ(kᵢ)ᵀ
        # = (gamma * error)ᵀ @ context_k
        # weighted_error: [batch, c, d_value], context_k: [batch, c, d_poly]
        grad = torch.bmm(
            context_k.transpose(-1, -2),  # [batch, d_poly, c]
            weighted_error,                # [batch, c, d_value]
        )  # [batch, d_poly, d_value]

        # Scale by 2/c (average over context)
        grad = 2.0 * grad / c

        # Compute mean error for metrics
        mean_error = (gamma.unsqueeze(-1) * error.abs()).sum(dim=1).mean()

        return grad, mean_error

    def _clip_memory_norm(self, M: torch.Tensor) -> torch.Tensor:
        """Clip memory norm for stability."""
        norms = M.norm(dim=(-2, -1), keepdim=True)
        scale = torch.clamp(self.config.max_memory_norm / (norms + self.config.eps), max=1.0)
        return M * scale

    def update(
        self,
        M: torch.Tensor,
        S: torch.Tensor,
        context_buffer: List[torch.Tensor],
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], dict]:
        """
        Update memory using Omega rule with context window.

        OPTIMIZED: Process in chunks rather than token-by-token to save memory.
        Updates happen every `context_window` tokens instead of every token.

        IMPORTANT: Memory updates are detached from the main autograd graph.
        This is by design - the Omega rule performs its own gradient computation
        internally (online learning), and only the read/query operation participates
        in the main loss backward pass. This prevents OOM from accumulating gradients
        across many update steps.

        Omega Rule Update:
            Sₜ = θₜSₜ₋₁ - ηₜ∇ℓ(Mₜ₋₁; context_window)
            Mₜ = αₜMₜ₋₁ + Sₜ

        Args:
            M: Current memory [batch, d_poly, d_value]
            S: Current momentum [batch, d_poly, d_value]
            context_buffer: List of past inputs (unused in chunked mode)
            x: Current input [batch, seq_len, d_model]

        Returns:
            M_new: Updated memory
            S_new: Updated momentum
            context_buffer_new: Updated context buffer
            metrics: Observability metrics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        c = self.config.context_window

        # Process in chunks of context_window size
        # This is more memory efficient than token-by-token
        n_chunks = (seq_len + c - 1) // c

        # Detach M and S from the main graph - updates are internal to Omega rule
        M_current = M.detach()
        S_current = S.detach()
        all_metrics = []

        # Memory update loop - detached from main autograd graph
        # The projection parameters still get gradients through the read operation
        with torch.no_grad():
            for chunk_idx in range(n_chunks):
                start = chunk_idx * c
                end = min(start + c, seq_len)
                chunk = x[:, start:end, :]  # [batch, chunk_len, d_model]

                # Use the chunk as the context window
                # Get the last token for parameter generation
                x_t = chunk[:, -1, :]  # [batch, d_model]

                # Compute context gates γᵢ(t)
                gamma = self.context_gates(x_t, chunk)  # [batch, chunk_len]

                # Project to keys and values
                chunk_k_raw = self.key_proj(chunk)  # [batch, chunk_len, d_key]
                chunk_v = self.value_proj(chunk)     # [batch, chunk_len, d_value]

                # Apply polynomial features to keys
                chunk_k_poly = self.poly_features(chunk_k_raw)  # [batch, chunk_len, d_poly]

                # Compute input-dependent parameters (from last token in chunk)
                alpha_t = torch.sigmoid(self.alpha_proj(x_t))  # [batch, 1]
                theta_t = torch.sigmoid(self.theta_proj(x_t))  # [batch, 1]
                eta_t = torch.sigmoid(self.eta_proj(x_t)) * 0.5  # [batch, 1], scale down

                # Expand for broadcasting
                alpha_t = alpha_t.unsqueeze(-1)  # [batch, 1, 1]
                theta_t = theta_t.unsqueeze(-1)
                eta_t = eta_t.unsqueeze(-1)

                # Compute Omega gradient over context window
                grad, pred_error = self._compute_omega_gradient(
                    M_current, chunk_k_poly, chunk_v, gamma
                )

                # Omega rule update:
                # Sₜ = θₜSₜ₋₁ - ηₜ∇ℓ
                S_new = theta_t * S_current - eta_t * grad

                # Mₜ = αₜMₜ₋₁ + Sₜ
                M_new = alpha_t * M_current + S_new

                # Clip for stability
                M_new = self._clip_memory_norm(M_new)

                # Update state
                M_current = M_new
                S_current = S_new

                # Collect metrics (only on last chunk to save compute)
                if chunk_idx == n_chunks - 1:
                    all_metrics.append({
                        "alpha": alpha_t.mean().item(),
                        "theta": theta_t.mean().item(),
                        "eta": eta_t.mean().item(),
                        "gamma_mean": gamma.mean().item(),
                        "gamma_sparsity": (gamma < 0.5).float().mean().item(),
                        "pred_error": pred_error.item(),
                        "memory_norm": M_new.norm(dim=(-2, -1)).mean().item(),
                        "momentum_norm": S_new.norm(dim=(-2, -1)).mean().item(),
                        "grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
                    })

        # Keep last chunk as context buffer for next call
        # IMPORTANT: Detach to prevent gradient accumulation across steps
        buffer = [x[:, -min(c, seq_len):, :].detach()]

        # Return metrics
        if all_metrics:
            metrics = all_metrics[-1]
        else:
            metrics = {}
        metrics["context_length"] = c
        metrics["n_chunks"] = n_chunks

        return M_current, S_current, buffer, metrics

    def retrieve(
        self,
        M: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using queries.

        Args:
            M: Memory matrix [batch, d_poly, d_value]
            x: Input [batch, seq_len, d_model]

        Returns:
            output: Retrieved values [batch, seq_len, d_model]
        """
        # Project to queries
        q = self.query_proj(x)  # [batch, seq, d_key]

        # Q-K alignment
        q = self.qk_proj(q)

        # Apply polynomial features
        q_poly = self.poly_features(q)  # [batch, seq, d_poly]

        # Retrieve: q @ M -> [batch, seq, d_value]
        retrieved = torch.bmm(q_poly, M)

        # Project to output dimension
        output = self.out_proj(retrieved)  # [batch, seq, d_model]

        return output

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], Optional[dict]]:
        """
        Full forward pass: update memory then retrieve.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Optional (M, S, context_buffer) tuple
            return_metrics: Whether to return metrics

        Returns:
            output: Retrieved values [batch, seq_len, d_model]
            new_state: Updated (M, S, context_buffer) tuple
            metrics: Optional observability metrics
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Initialize state if needed
        if state is None:
            M, S, context_buffer = self.init_state(batch_size, device, dtype)
        else:
            M, S, context_buffer = state

        # Update memory with new information
        M_new, S_new, buffer_new, metrics = self.update(M, S, context_buffer, x)

        # Retrieve from updated memory
        output = self.retrieve(M_new, x)

        new_state = (M_new, S_new, buffer_new)

        if return_metrics:
            return output, new_state, metrics
        return output, new_state, None


# Factory function for easy creation
def create_omega_memory(
    d_model: int = 512,
    d_key: int = 512,
    d_value: int = 512,
    poly_degree: int = 2,
    context_window: int = 16,
) -> OmegaMemory:
    """Create an OmegaMemory module with the given configuration."""
    config = OmegaMemoryConfig(
        d_model=d_model,
        d_key=d_key,
        d_value=d_value,
        poly_degree=poly_degree,
        context_window=context_window,
    )
    return OmegaMemory(config)

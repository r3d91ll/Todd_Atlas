"""
Matrix-valued memory module following Behrouz et al. papers:
- Titans: Surprise-based memory with momentum
- Miras: Unified framework with retention gates
- Atlas: Second-order optimization insights
- TNT: Chunk-based training with Q-K projection
- Nested Learning: Multi-frequency update dynamics

Core equations (from Titans):
    S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; k_t, v_t)   # Surprise accumulation
    M_t = (1 - α_t) · M_{t-1} + S_t                      # Memory update with decay

Key insights:
    - α_t, η_t, θ_t are INPUT-DEPENDENT (functions of x_t)
    - Memory has explicit forgetting via (1 - α_t) decay
    - Surprise accumulates momentum of gradients
    - Q-K projection aligns query space to key space (TNT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TitansMemory(nn.Module):
    """
    Titans-style matrix memory with surprise-based updates.

    Implements the correct update equations:
        S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; k_t, v_t)
        M_t = (1 - α_t) · M_{t-1} + S_t

    Where α_t, η_t, θ_t are input-dependent via linear projections.

    Args:
        d_model: Model dimension (input dimension)
        d_key: Key/query dimension for memory
        d_value: Value/output dimension for memory
        init_alpha: Initial value for forgetting gate (default 0.1 = 10% forget)
        init_eta: Initial value for surprise decay (default 0.9)
        init_theta: Initial value for gradient scaling (default 0.1)
    """

    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 512,
        d_value: int = 512,
        init_alpha: float = 0.1,
        init_eta: float = 0.9,
        init_theta: float = 0.1,
        # Numerical stability parameters
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_key = d_key
        self.d_value = d_value

        # Stability parameters
        self.grad_clip = grad_clip
        self.memory_max_norm = memory_max_norm
        self.surprise_max_norm = surprise_max_norm

        # Projections for key, value, query from input
        self.key_proj = nn.Linear(d_model, d_key, bias=False)
        self.value_proj = nn.Linear(d_model, d_value, bias=False)
        self.query_proj = nn.Linear(d_model, d_key, bias=False)

        # Q-K projection: aligns query space to key space (TNT insight)
        # Critical for retrieval to work - queries must be in same space as keys
        self.qk_proj = nn.Linear(d_key, d_key, bias=False)

        # Input-dependent parameter generators (Titans insight)
        # These produce per-token α, η, θ values
        # Use sigmoid to bound outputs to [0, 1]

        # α_t: Forgetting gate - how much to decay memory (0 = full retain, 1 = full forget)
        self.alpha_proj = nn.Linear(d_model, 1, bias=True)

        # η_t: Surprise decay - how much past surprise to retain (0 = no momentum, 1 = full momentum)
        self.eta_proj = nn.Linear(d_model, 1, bias=True)

        # θ_t: Gradient scaling - how strongly current gradient affects surprise
        self.theta_proj = nn.Linear(d_model, 1, bias=True)

        # Initialize projection biases to achieve desired initial values
        # sigmoid(bias) ≈ init_value, so bias ≈ logit(init_value)
        with torch.no_grad():
            self.alpha_proj.bias.fill_(self._logit(init_alpha))
            self.eta_proj.bias.fill_(self._logit(init_eta))
            self.theta_proj.bias.fill_(self._logit(init_theta))

        # Initialize projections
        self._init_weights()

    def _logit(self, p: float) -> float:
        """Compute logit (inverse sigmoid) for initialization."""
        p = max(min(p, 0.999), 0.001)  # Clamp to avoid inf
        return torch.tensor(p / (1 - p)).log().item()

    def _init_weights(self):
        """Initialize projection weights."""
        # Xavier for key/value/query projections
        for proj in [self.key_proj, self.value_proj, self.query_proj]:
            nn.init.xavier_uniform_(proj.weight)

        # Q-K projection starts as identity (TNT)
        nn.init.eye_(self.qk_proj.weight)

        # Small weights for parameter projections (let bias dominate initially)
        for proj in [self.alpha_proj, self.eta_proj, self.theta_proj]:
            nn.init.normal_(proj.weight, std=0.01)

    def _clip_state_norm(
        self,
        state: torch.Tensor,
        max_norm: float,
    ) -> torch.Tensor:
        """
        Clip state tensor norm to prevent explosion.

        Uses per-batch-element clipping to preserve relative magnitudes
        within each batch sample.

        Args:
            state: Tensor [batch, d_key, d_value]
            max_norm: Maximum allowed Frobenius norm per batch element

        Returns:
            Clipped state tensor
        """
        # Compute per-element norm: [batch]
        norms = state.norm(dim=(-2, -1), keepdim=True)  # [batch, 1, 1]

        # Compute scaling factor (only scale down, never up)
        scale = torch.clamp(max_norm / (norms + 1e-8), max=1.0)

        return state * scale

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize memory state M and surprise S.

        Returns:
            M: Memory matrix [batch, d_key, d_value] - initialized to small values
            S: Surprise accumulator [batch, d_key, d_value] - initialized to zeros
        """
        # Initialize memory with small random values (not zeros!)
        # This provides a starting point for gradient flow
        M = torch.randn(batch_size, self.d_key, self.d_value, device=device) * 0.01

        # Surprise starts at zero
        S = torch.zeros(batch_size, self.d_key, self.d_value, device=device)

        return M, S

    def compute_gradient(
        self,
        M: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of L2 attentional bias: ℓ(M; k, v) = ||M·k - v||²

        ∇_M ℓ = 2(M·k - v)·k^T

        Args:
            M: Memory [batch, d_key, d_value]
            k: Keys [batch, seq, d_key]
            v: Values [batch, seq, d_value]

        Returns:
            grad: Gradient [batch, d_key, d_value]
        """
        batch_size, seq_len, _ = k.shape

        # Prediction: M @ k for each position
        # M: [batch, d_key, d_value], k: [batch, seq, d_key]
        # Transpose M to [batch, d_value, d_key], then M @ k^T gives [batch, d_value, seq]
        k_t = k.transpose(-1, -2)  # [batch, d_key, seq]
        pred = torch.bmm(M.transpose(-1, -2), k_t)  # [batch, d_value, seq]

        # Error: pred - v^T
        v_t = v.transpose(-1, -2)  # [batch, d_value, seq]
        error = pred - v_t  # [batch, d_value, seq]

        # Gradient: 2 * k @ error^T (averaged over sequence)
        # k_t: [batch, d_key, seq], error: [batch, d_value, seq]
        # grad should be [batch, d_key, d_value]
        grad = 2.0 * torch.bmm(k_t, error.transpose(-1, -2)) / seq_len

        return grad, error

    def update(
        self,
        M: torch.Tensor,
        S: torch.Tensor,
        x: torch.Tensor,
        retention_penalty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Update memory using Titans equations:
            S_t = η_t · S_{t-1} - θ_t · ∇ℓ(M_{t-1}; k_t, v_t)
            M_t = (1 - α_t) · M_{t-1} + S_t

        Args:
            M: Current memory [batch, d_key, d_value]
            S: Current surprise [batch, d_key, d_value]
            x: Input [batch, seq_len, d_model]
            retention_penalty: Optional additional gradient from retention gates

        Returns:
            M_new: Updated memory
            S_new: Updated surprise
            metrics: Observability metrics
        """
        batch_size, seq_len, _ = x.shape

        # Project input to keys and values
        k = self.key_proj(x)  # [batch, seq, d_key]
        v = self.value_proj(x)  # [batch, seq, d_value]

        # Compute input-dependent parameters
        # Average over sequence for chunk-level parameters (TNT insight)
        x_mean = x.mean(dim=1)  # [batch, d_model]

        alpha_t = torch.sigmoid(self.alpha_proj(x_mean))  # [batch, 1] - forgetting
        eta_t = torch.sigmoid(self.eta_proj(x_mean))      # [batch, 1] - surprise decay
        theta_t = torch.sigmoid(self.theta_proj(x_mean))  # [batch, 1] - gradient scale

        # Expand for broadcasting with [batch, d_key, d_value]
        alpha_t = alpha_t.unsqueeze(-1)  # [batch, 1, 1]
        eta_t = eta_t.unsqueeze(-1)      # [batch, 1, 1]
        theta_t = theta_t.unsqueeze(-1)  # [batch, 1, 1]

        # Compute gradient
        grad, error = self.compute_gradient(M, k, v)

        # Add retention penalty if provided (Miras)
        if retention_penalty is not None:
            grad = grad + retention_penalty

        # STABILITY: Clip gradient norm to prevent explosion
        # This is critical for persistent memory states
        grad = self._clip_state_norm(grad, self.grad_clip)

        # Titans update equations:
        # S_t = η_t · S_{t-1} - θ_t · ∇ℓ
        S_new = eta_t * S - theta_t * grad

        # STABILITY: Clip surprise norm
        S_new = self._clip_state_norm(S_new, self.surprise_max_norm)

        # M_t = (1 - α_t) · M_{t-1} + S_t
        M_new = (1 - alpha_t) * M + S_new

        # STABILITY: Clip memory norm
        M_new = self._clip_state_norm(M_new, self.memory_max_norm)

        # Compute metrics for observability
        metrics = {
            "memory_norm": M_new.norm(dim=(-2, -1)).mean().item(),
            "surprise_norm": S_new.norm(dim=(-2, -1)).mean().item(),
            "grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
            "prediction_error": error.abs().mean().item(),
            "alpha_mean": alpha_t.mean().item(),
            "eta_mean": eta_t.mean().item(),
            "theta_mean": theta_t.mean().item(),
        }

        return M_new, S_new, metrics

    def retrieve(
        self,
        M: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retrieve from memory using query with Q-K projection (TNT).

        Args:
            M: Memory matrix [batch, d_key, d_value]
            x: Input [batch, seq_len, d_model]

        Returns:
            output: Retrieved values [batch, seq_len, d_value]
        """
        # Project to query
        q = self.query_proj(x)  # [batch, seq, d_key]

        # Q-K projection - align query to key space (TNT critical insight)
        # This ensures retrieval operates in the same space memory was trained on
        q = self.qk_proj(q)  # [batch, seq, d_key]

        # Retrieve: q @ M -> [batch, seq, d_value]
        output = torch.bmm(q, M)

        return output

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        retention_penalty: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        """
        Full forward pass: update memory then retrieve.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Optional (M, S) tuple
            retention_penalty: Optional retention gradient
            return_metrics: Whether to return metrics

        Returns:
            output: Retrieved values [batch, seq_len, d_value]
            new_state: Updated (M, S) tuple
            metrics: Optional observability metrics
        """
        batch_size = x.shape[0]
        device = x.device

        # Initialize state if needed
        if state is None:
            state = self.init_state(batch_size, device)

        M, S = state

        # Update memory with new information
        M_new, S_new, metrics = self.update(M, S, x, retention_penalty)

        # Retrieve from updated memory
        output = self.retrieve(M_new, x)

        if return_metrics:
            return output, (M_new, S_new), metrics
        return output, (M_new, S_new), None


# Backward compatibility alias
class MatrixMemory(TitansMemory):
    """
    Alias for TitansMemory for backward compatibility.

    Maps old parameter names to new ones:
        d_key -> d_key (same)
        d_value -> d_value (same)
        momentum_beta -> init_eta (surprise decay)
        init_lr -> init_theta (gradient scaling)
        learn_lr -> always True (input-dependent)
    """

    def __init__(
        self,
        d_key: int = 512,
        d_value: int = 512,
        momentum_beta: float = 0.9,
        learn_lr: bool = True,  # Ignored - always input-dependent now
        init_lr: float = 0.1,
        # Pass through stability parameters
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__(
            d_model=d_key,  # Assume d_model == d_key for compatibility
            d_key=d_key,
            d_value=d_value,
            init_alpha=0.1,  # Default forgetting rate
            init_eta=momentum_beta,  # Map momentum_beta to surprise decay
            init_theta=init_lr,  # Map init_lr to gradient scaling
            grad_clip=grad_clip,
            memory_max_norm=memory_max_norm,
            surprise_max_norm=surprise_max_norm,
        )


class ChunkedTitansMemory(TitansMemory):
    """
    Titans memory with chunk-based processing for TNT-style training.

    Key TNT insights:
    - Compute gradients relative to chunk start state (not per-token state)
    - Q-K projection aligns retrieval to training space
    - Hierarchical: global (large chunks) + local (small chunks) memory
    """

    def __init__(
        self,
        d_model: int = 512,
        d_key: int = 512,
        d_value: int = 512,
        chunk_size: int = 2048,
        init_alpha: float = 0.1,
        init_eta: float = 0.9,
        init_theta: float = 0.1,
        grad_clip: float = 1.0,
        memory_max_norm: float = 100.0,
        surprise_max_norm: float = 100.0,
    ):
        super().__init__(
            d_model=d_model,
            d_key=d_key,
            d_value=d_value,
            init_alpha=init_alpha,
            init_eta=init_eta,
            init_theta=init_theta,
            grad_clip=grad_clip,
            memory_max_norm=memory_max_norm,
            surprise_max_norm=surprise_max_norm,
        )
        self.chunk_size = chunk_size

    def forward_chunked(
        self,
        x: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[dict]]:
        """
        Chunked forward pass for TNT-style training.

        Within each chunk, all gradients are computed relative to the chunk's
        starting memory state, enabling parallel gradient accumulation.

        Args:
            x: Input [batch, seq_len, d_model]
            state: Global memory state
            return_metrics: Whether to return metrics

        Returns:
            output: Retrieved values
            state: Updated global state
            metrics: Aggregated metrics
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device

        # Initialize global state
        if state is None:
            state = self.init_state(batch_size, device)

        M_global, S_global = state

        # Split into chunks
        chunks = x.split(self.chunk_size, dim=1)

        outputs = []
        all_metrics = []

        for i, chunk in enumerate(chunks):
            # Process chunk with current global state
            output, (M_new, S_new), chunk_metrics = self.forward(
                chunk,
                state=(M_global, S_global),
                return_metrics=return_metrics,
            )
            outputs.append(output)

            # Update global state for next chunk
            # Detach to prevent gradient flow between chunks (TNT)
            M_global = M_new.detach()
            S_global = S_new.detach()

            if return_metrics and chunk_metrics:
                chunk_metrics["chunk_idx"] = i
                all_metrics.append(chunk_metrics)

        # Concatenate outputs
        output = torch.cat(outputs, dim=1)

        # For gradient computation, we need the final non-detached state
        # Re-run last chunk without detach for proper gradients
        if len(chunks) > 0:
            last_chunk = chunks[-1]
            # Use second-to-last state (or initial if only one chunk)
            if len(chunks) > 1:
                # Recompute with gradient flow
                _, (M_global, S_global), _ = self.forward(
                    last_chunk,
                    state=(M_global, S_global),
                    return_metrics=False,
                )

        # Aggregate metrics
        if return_metrics and all_metrics:
            agg_metrics = {
                "n_chunks": len(chunks),
                "final_memory_norm": M_global.norm(dim=(-2, -1)).mean().item(),
                "avg_alpha": sum(m["alpha_mean"] for m in all_metrics) / len(all_metrics),
                "avg_eta": sum(m["eta_mean"] for m in all_metrics) / len(all_metrics),
                "avg_theta": sum(m["theta_mean"] for m in all_metrics) / len(all_metrics),
            }
            return output, (M_global, S_global), agg_metrics

        return output, (M_global, S_global), None


# Keep old name for imports
ChunkedMatrixMemory = ChunkedTitansMemory

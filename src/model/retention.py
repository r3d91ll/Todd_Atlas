"""
Retention gate module following Miras framework.

From "It's All Connected" (Behrouz et al.):
- Reframes "forgetting" as "retention regularization"
- Learning-Retaining viewpoint: W_t = argmin[loss + retention_penalty]
- Local retention: proximity to previous state
- Global retention: memory size normalization

Reference: Section 4 of "It's All Connected"
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class RetentionGate(nn.Module):
    """
    Retention gate with local and global components.

    Learning-Retaining objective:
        W_t = argmin_W [ℓ(W; k_t, v_t) + Ret_t(W, W_{t-1})]

    Where:
        Ret_t = λ_local · ||W - W_{t-1}||²_F + λ_global · ||W||²_F

    The gradient of this retention term is added to the memory update.

    Args:
        d_key: Key dimension
        d_value: Value dimension
        init_local: Initial local retention coefficient (0-1)
        init_global: Initial global retention coefficient (0-1)
        learn_coefficients: Whether to learn retention coefficients
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        init_local: float = 0.5,
        init_global: float = 0.1,
        learn_coefficients: bool = True,
    ):
        super().__init__()

        self.d_key = d_key
        self.d_value = d_value

        if learn_coefficients:
            # Parameterize via sigmoid for (0, 1) range
            # Store pre-sigmoid values
            self.local_logit = nn.Parameter(
                torch.tensor(init_local).logit()
            )
            self.global_logit = nn.Parameter(
                torch.tensor(init_global).logit()
            )
        else:
            self.register_buffer(
                "local_logit", torch.tensor(init_local).logit()
            )
            self.register_buffer(
                "global_logit", torch.tensor(init_global).logit()
            )

    @property
    def lambda_local(self) -> torch.Tensor:
        """Local retention coefficient in (0, 1)."""
        return torch.sigmoid(self.local_logit)

    @property
    def lambda_global(self) -> torch.Tensor:
        """Global retention coefficient in (0, 1)."""
        return torch.sigmoid(self.global_logit)

    def compute_penalty_gradient(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute gradient of retention penalty w.r.t. W.

        Retention penalty:
            Ret = λ_local · ||W - W_prev||²_F + λ_global · ||W||²_F

        Gradient:
            ∂Ret/∂W = 2·λ_local·(W - W_prev) + 2·λ_global·W

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]

        Returns:
            grad: Retention penalty gradient [batch, d_key, d_value]
            metrics: Observable metrics
        """
        λ_local = self.lambda_local
        λ_global = self.lambda_global

        # Local retention gradient: pull toward previous state
        local_grad = 2 * λ_local * (W - W_prev)

        # Global retention gradient: regularize magnitude
        global_grad = 2 * λ_global * W

        # Combined gradient
        grad = local_grad + global_grad

        # Compute actual penalty values for logging
        local_penalty = λ_local * (W - W_prev).pow(2).sum(dim=(-2, -1)).mean()
        global_penalty = λ_global * W.pow(2).sum(dim=(-2, -1)).mean()

        metrics = {
            "lambda_local": λ_local.item(),
            "lambda_global": λ_global.item(),
            "local_penalty": local_penalty.item(),
            "global_penalty": global_penalty.item(),
            "total_penalty": (local_penalty + global_penalty).item(),
            "retention_grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
        }

        return grad, metrics

    def forward(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute retention penalty gradient.

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]

        Returns:
            grad: Retention penalty gradient
            metrics: Observable metrics
        """
        return self.compute_penalty_gradient(W, W_prev)


class AdaptiveRetentionGate(RetentionGate):
    """
    Retention gate with input-dependent coefficients.

    Extends basic retention with:
    - Per-token retention coefficients based on input
    - "Surprise" modulation: retain less when input is surprising

    This connects to Titans' surprise metric while staying within
    the Miras retention framework.
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        d_model: int,
        init_local: float = 0.5,
        init_global: float = 0.1,
    ):
        super().__init__(d_key, d_value, init_local, init_global, learn_coefficients=True)

        # Project input to retention modulation
        self.local_mod = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.global_mod = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

    def compute_adaptive_coefficients(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent retention coefficients.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            local_coef: Local coefficient [batch, seq_len, 1]
            global_coef: Global coefficient [batch, seq_len, 1]
        """
        # Base coefficients
        base_local = self.lambda_local
        base_global = self.lambda_global

        # Input modulation (0.5 to 1.5 range)
        local_mod = 0.5 + self.local_mod(x)  # [batch, seq, 1]
        global_mod = 0.5 + self.global_mod(x)

        # Modulated coefficients
        local_coef = (base_local * local_mod).clamp(0, 1)
        global_coef = (base_global * global_mod).clamp(0, 1)

        return local_coef, global_coef

    def forward_adaptive(
        self,
        W: torch.Tensor,
        W_prev: torch.Tensor,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute adaptive retention penalty gradient.

        Args:
            W: Current memory [batch, d_key, d_value]
            W_prev: Previous memory [batch, d_key, d_value]
            x: Input for modulation [batch, seq_len, d_model]

        Returns:
            grad: Retention penalty gradient
            metrics: Observable metrics
        """
        local_coef, global_coef = self.compute_adaptive_coefficients(x)

        # Average over sequence for single coefficient per sample
        λ_local = local_coef.mean(dim=1, keepdim=True).squeeze(-1)  # [batch, 1]
        λ_global = global_coef.mean(dim=1, keepdim=True).squeeze(-1)

        # Reshape for broadcasting
        λ_local = λ_local.unsqueeze(-1)  # [batch, 1, 1]
        λ_global = λ_global.unsqueeze(-1)

        # Compute gradients
        local_grad = 2 * λ_local * (W - W_prev)
        global_grad = 2 * λ_global * W
        grad = local_grad + global_grad

        metrics = {
            "lambda_local_mean": λ_local.mean().item(),
            "lambda_global_mean": λ_global.mean().item(),
            "lambda_local_std": local_coef.std().item(),
            "lambda_global_std": global_coef.std().item(),
            "retention_grad_norm": grad.norm(dim=(-2, -1)).mean().item(),
        }

        return grad, metrics

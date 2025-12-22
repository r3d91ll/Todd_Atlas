"""
StableMax Activation - Numerically Stable Alternative to Softmax.

Based on "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2)
by Darshil Doshi, Tianyu He, Aritra Das, Andrey Gromov.

StableMax avoids the floating-point absorption errors that cause softmax
collapse during grokking. It enables grokking without weight decay by
preventing the numerical instability that blocks gradient flow.

References:
- Doshi et al. (2025): "Grokking at the Edge of Numerical Stability"
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class StableMax(nn.Module):
    """
    StableMax activation function.

    Definition:
        StableMax(x_i) = s(x_i) / sum_j(s(x_j))

        where s(x) = {
            x + 1       if x >= 0
            1/(1-x)     if x < 0
        }

    This is equivalent to applying g(x) before standard softmax:
        g(x) = {
            log(x+1)    if x >= 0
            -log(-x+1)  if x < 0
        }

    StableMax maintains numerical stability by avoiding exponentials
    that can overflow or cause absorption errors.
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply StableMax activation.

        Args:
            x: Input tensor of any shape

        Returns:
            StableMax probabilities (same shape as input)
        """
        # Apply piecewise transformation s(x)
        s_x = torch.where(
            x >= 0,
            x + 1,  # x + 1 for x >= 0
            1.0 / (1 - x)  # 1/(1-x) for x < 0
        )

        # Normalize
        return s_x / s_x.sum(dim=self.dim, keepdim=True)

    @staticmethod
    def log_stablemax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute log(StableMax(x)) in a numerically stable way.

        Useful for computing StableCE loss.
        """
        s_x = torch.where(
            x >= 0,
            x + 1,
            1.0 / (1 - x)
        )

        log_s_x = torch.where(
            x >= 0,
            torch.log(x + 1),
            -torch.log(1 - x)
        )

        log_sum = torch.log(s_x.sum(dim=dim, keepdim=True))

        return log_s_x - log_sum


class StableCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss using StableMax instead of Softmax.

    L_StCE(f(x), y) = -log(StableMax(z_y))

    This loss function enables grokking without weight decay by
    avoiding the numerical instability of standard cross-entropy.
    """

    def __init__(
        self,
        reduction: str = 'mean',
        ignore_index: int = -100,
    ):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute StableCE loss.

        Args:
            logits: [batch, vocab] or [batch, seq, vocab]
            targets: [batch] or [batch, seq]

        Returns:
            Loss value
        """
        # Flatten if needed
        if logits.dim() == 3:
            batch, seq, vocab = logits.shape
            logits = logits.view(-1, vocab)
            targets = targets.view(-1)

        # Compute log(StableMax)
        log_probs = StableMax.log_stablemax(logits, dim=-1)

        # Gather log probs for target classes
        # Create mask for ignore_index
        mask = targets != self.ignore_index
        targets_masked = targets.clone()
        targets_masked[~mask] = 0  # Placeholder for gather

        # Gather
        nll = -log_probs.gather(dim=-1, index=targets_masked.unsqueeze(-1)).squeeze(-1)

        # Apply mask
        nll = nll * mask.float()

        # Reduction
        if self.reduction == 'none':
            return nll
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'mean':
            return nll.sum() / mask.float().sum().clamp(min=1)
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def stablemax_softmax_hybrid(
    logits: torch.Tensor,
    alpha: float = 0.5,
    dim: int = -1,
) -> torch.Tensor:
    """
    Hybrid between softmax and stablemax.

    Allows gradual transition from softmax to stablemax during training.

    Args:
        logits: Input logits
        alpha: Mixing factor (0 = softmax, 1 = stablemax)
        dim: Dimension to normalize over

    Returns:
        Probability distribution
    """
    softmax_probs = F.softmax(logits, dim=dim)
    stablemax_probs = StableMax(dim=dim)(logits)

    return (1 - alpha) * softmax_probs + alpha * stablemax_probs

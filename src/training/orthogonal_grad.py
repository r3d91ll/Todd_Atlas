"""
Orthogonal Gradient Projection (PerpGrad) - Prevents Naive Loss Minimization.

Based on "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2)
by Darshil Doshi, Tianyu He, Aritra Das, Andrey Gromov.

Key insight: After memorization, gradients align with weights, causing the
model to simply scale up logits rather than learn generalizing features.
By projecting out the weight-aligned component, the model is forced to find
other ways to reduce loss -> immediate generalization without delayed grokking.

Update rule:
    grad_perp(w) = grad(w) - (w^T grad(w) / w^T w) * w

Usage:
    loss.backward()
    if use_orthogonal_grad:
        apply_orthogonal_projection(model, strength=1.0)
    optimizer.step()

References:
- Doshi et al. (2025): "Grokking at the Edge of Numerical Stability"
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting"
"""

import torch
import torch.nn as nn


@torch.no_grad()
def apply_orthogonal_projection(
    model: nn.Module,
    strength: float = 1.0,
) -> None:
    """
    Project out weight-aligned gradient components from all parameters.

    This removes the gradient component in the weight direction, preventing
    naive loss minimization through weight scaling.

    Args:
        model: Model with gradients computed (.backward() already called)
        strength: How much to project out (0=none, 1=full projection)
    """
    if strength <= 0:
        return

    for param in model.parameters():
        if param.grad is None:
            continue

        # Flatten for dot product
        grad_flat = param.grad.view(-1)
        weight_flat = param.data.view(-1)

        # Compute weight norm squared
        weight_norm_sq = (weight_flat * weight_flat).sum()

        if weight_norm_sq < 1e-10:
            continue  # Skip near-zero weights

        # Compute projection coefficient: (w^T grad) / (w^T w)
        proj_coeff = (grad_flat * weight_flat).sum() / weight_norm_sq

        # Remove weight-aligned component: grad_perp = grad - coeff * w
        param.grad -= strength * proj_coeff * param.data

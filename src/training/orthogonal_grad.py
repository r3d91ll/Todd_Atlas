"""
Orthogonal Gradient (⊥Grad) Optimizer - Removes Weight-Scaling Component.

Based on "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2)
by Darshil Doshi, Tianyu He, Aritra Das, Andrey Gromov.

The ⊥Grad optimizer projects out the gradient component that aligns with
the weight direction, preventing Naïve Loss Minimization (NLM) that causes
weight scaling without learning.

Key insight: After memorization, gradients align with weights, causing the
model to simply scale up logits rather than learn generalizing features.
By removing this component, the model is forced to find other ways to
reduce loss → immediate generalization without delayed grokking.

References:
- Doshi et al. (2025): "Grokking at the Edge of Numerical Stability"
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
"""

import torch
from torch.optim import Optimizer
from typing import Optional, Callable, Iterable, Dict, Any, List


class OrthogonalGrad(Optimizer):
    """
    Optimizer wrapper that projects out weight-aligned gradient components.

    Update rule:
        θ_{t+1} = θ_t - η * ∇_⊥L(θ_t)

    where:
        ∇_⊥L(θ) = ∇L(θ) - (θ^T ∇L(θ) / θ^T θ) * θ

    This removes the gradient component in the weight direction,
    preventing naïve loss minimization through weight scaling.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        projection_strength: float = 1.0,
        apply_to_layers: Optional[List[str]] = None,
    ):
        """
        Initialize OrthogonalGrad optimizer.

        Args:
            params: Model parameters
            lr: Learning rate
            betas: Adam beta coefficients
            eps: Adam epsilon
            weight_decay: Weight decay (applied before projection)
            projection_strength: How much to project out (0=none, 1=full)
            apply_to_layers: If specified, only apply projection to these layers
                           (by parameter name substring). None = apply to all.
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            projection_strength=projection_strength,
            apply_to_layers=apply_to_layers,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        Perform a single optimization step with orthogonal gradient projection.

        Args:
            closure: Optional closure that reevaluates the model and returns loss

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            projection_strength = group['projection_strength']
            apply_to_layers = group['apply_to_layers']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                param_name = self._get_param_name(p)

                # Check if we should apply projection to this layer
                should_project = True
                if apply_to_layers is not None:
                    should_project = any(layer in param_name for layer in apply_to_layers)

                # Apply orthogonal projection
                if should_project and projection_strength > 0:
                    grad = self._project_orthogonal(grad, p.data, projection_strength)

                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])

                # Get state
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second raw moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size
                step_size = group['lr'] / bias_correction1

                # Compute denominator
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

                # Update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _project_orthogonal(
        self,
        grad: torch.Tensor,
        weight: torch.Tensor,
        strength: float,
    ) -> torch.Tensor:
        """
        Project gradient orthogonal to weight direction.

        ∇_⊥L = ∇L - (θ^T ∇L / θ^T θ) * θ

        Args:
            grad: Gradient tensor
            weight: Weight tensor
            strength: Projection strength (0-1)

        Returns:
            Projected gradient
        """
        # Flatten for dot product
        grad_flat = grad.view(-1)
        weight_flat = weight.view(-1)

        # Compute projection coefficient
        weight_norm_sq = (weight_flat * weight_flat).sum()

        if weight_norm_sq < 1e-10:
            return grad  # Avoid division by zero

        proj_coeff = (grad_flat * weight_flat).sum() / weight_norm_sq

        # Compute weight-aligned component
        weight_component = proj_coeff * weight

        # Subtract (scaled by strength)
        return grad - strength * weight_component

    def _get_param_name(self, param: torch.Tensor) -> str:
        """Get parameter name for layer filtering."""
        # This is a workaround since we don't have direct access to names
        # In practice, you'd pass named_parameters() and track names
        return str(id(param))


class OrthogonalGradWrapper:
    """
    Wrapper to add orthogonal gradient projection to any optimizer.

    Usage:
        base_optimizer = AdamW(model.parameters(), lr=1e-4)
        optimizer = OrthogonalGradWrapper(
            base_optimizer,
            model,
            projection_strength=1.0,
        )
        optimizer.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        model: torch.nn.Module,
        projection_strength: float = 1.0,
        apply_to_layers: Optional[List[str]] = None,
    ):
        """
        Initialize wrapper.

        Args:
            optimizer: Base optimizer (e.g., AdamW)
            model: Model to optimize (needed for named_parameters)
            projection_strength: How much to project (0=none, 1=full)
            apply_to_layers: Layer name substrings to apply projection to
        """
        self.optimizer = optimizer
        self.model = model
        self.projection_strength = projection_strength
        self.apply_to_layers = apply_to_layers

        # Build param name mapping
        self.param_names: Dict[int, str] = {}
        for name, param in model.named_parameters():
            self.param_names[id(param)] = name

    @torch.no_grad()
    def _apply_projection(self):
        """Apply orthogonal projection to all gradients."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # Check if we should apply to this layer
            should_project = True
            if self.apply_to_layers is not None:
                should_project = any(layer in name for layer in self.apply_to_layers)

            if should_project and self.projection_strength > 0:
                grad = param.grad
                weight = param.data

                # Flatten
                grad_flat = grad.view(-1)
                weight_flat = weight.view(-1)

                # Projection
                weight_norm_sq = (weight_flat * weight_flat).sum()
                if weight_norm_sq > 1e-10:
                    proj_coeff = (grad_flat * weight_flat).sum() / weight_norm_sq
                    weight_component = proj_coeff * weight
                    param.grad = grad - self.projection_strength * weight_component

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Perform optimization step with projection."""
        self._apply_projection()
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none: bool = False):
        """Zero gradients."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> Dict[str, Any]:
        """Get optimizer state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load optimizer state dict."""
        self.optimizer.load_state_dict(state_dict)

    @property
    def param_groups(self):
        """Access param groups."""
        return self.optimizer.param_groups

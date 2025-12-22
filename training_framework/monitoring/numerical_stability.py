"""
Numerical Stability Metrics for Grokking Detection.

Based on "Grokking at the Edge of Numerical Stability" (arXiv:2501.04697v2).

Key metrics:
- Softmax Collapse (SC): Detects floating-point absorption in softmax
- NLM Gradient Alignment: Detects naïve loss minimization (weight scaling)

These metrics help understand WHY grokking happens and when the model
is at risk of numerical instability.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


@dataclass
class NumericalStabilityMetrics:
    """Container for numerical stability metrics."""
    step: int = 0

    # Softmax Collapse metrics
    sc_fraction: float = 0.0  # Fraction of samples experiencing softmax collapse
    max_logit_mean: float = 0.0  # Mean of max logits (high = SC risk)
    logit_range_mean: float = 0.0  # Mean range of logits per sample

    # NLM Gradient Alignment metrics
    grad_weight_cosine: float = 0.0  # Cosine similarity between grad and weights
    grad_weight_cosine_output: float = 0.0  # Same but for output layer only
    weight_norm: float = 0.0  # Total weight norm (growing = NLM active)
    weight_norm_growth_rate: float = 0.0  # Rate of weight norm growth

    # Risk indicators
    sc_risk: str = "low"  # low, medium, high, critical
    nlm_active: bool = False  # True if NLM is dominating updates

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "stability/sc_fraction": self.sc_fraction,
            "stability/max_logit_mean": self.max_logit_mean,
            "stability/logit_range_mean": self.logit_range_mean,
            "stability/grad_weight_cosine": self.grad_weight_cosine,
            "stability/grad_weight_cosine_output": self.grad_weight_cosine_output,
            "stability/weight_norm": self.weight_norm,
            "stability/weight_norm_growth_rate": self.weight_norm_growth_rate,
            "stability/sc_risk": self.sc_risk,
            "stability/nlm_active": self.nlm_active,
        }


class NumericalStabilityMonitor:
    """
    Monitors numerical stability during training.

    Detects:
    1. Softmax Collapse (SC) - when logits are so large that softmax
       experiences floating-point absorption errors
    2. Naïve Loss Minimization (NLM) - when gradients align with weight
       direction, causing weight scaling without learning
    """

    def __init__(
        self,
        sc_threshold: float = 1e-7,  # Threshold for detecting SC
        nlm_cosine_threshold: float = 0.7,  # Cosine threshold for NLM detection
        history_size: int = 100,
    ):
        self.sc_threshold = sc_threshold
        self.nlm_cosine_threshold = nlm_cosine_threshold
        self.history_size = history_size

        self.weight_norm_history: List[float] = []
        self.last_metrics: Optional[NumericalStabilityMetrics] = None

    def compute_sc_fraction(self, logits: torch.Tensor) -> Tuple[float, float, float]:
        """
        Compute fraction of samples experiencing Softmax Collapse.

        SC occurs when: sum(exp(z_k)) ≈ exp(z_max)
        This means the max logit dominates and other gradients vanish.

        Args:
            logits: [batch, seq_len, vocab] or [batch, vocab]

        Returns:
            (sc_fraction, max_logit_mean, logit_range_mean)
        """
        with torch.no_grad():
            # Flatten to [N, vocab]
            if logits.dim() == 3:
                logits = logits.view(-1, logits.size(-1))

            # Get max logit per sample
            max_logits, _ = logits.max(dim=-1)

            # Compute sum of exp(logits) and exp(max_logit)
            # Use logsumexp for numerical stability in our detection
            log_sum_exp = torch.logsumexp(logits, dim=-1)

            # SC occurs when log_sum_exp ≈ max_logits
            # i.e., when exp(log_sum_exp) ≈ exp(max_logits)
            # This means: log_sum_exp - max_logits ≈ 0
            sc_indicator = log_sum_exp - max_logits

            # SC if the difference is very small (other terms negligible)
            # log(1 + small) ≈ small, so if sc_indicator < log(1 + threshold)
            sc_threshold_log = np.log(1 + self.sc_threshold)
            sc_samples = (sc_indicator < sc_threshold_log).float()

            sc_fraction = sc_samples.mean().item()
            max_logit_mean = max_logits.mean().item()

            # Logit range (max - min)
            min_logits, _ = logits.min(dim=-1)
            logit_range = max_logits - min_logits
            logit_range_mean = logit_range.mean().item()

            return sc_fraction, max_logit_mean, logit_range_mean

    def compute_grad_weight_alignment(
        self,
        model: nn.Module,
    ) -> Tuple[float, float]:
        """
        Compute cosine similarity between gradients and weights.

        High similarity indicates NLM is active - gradients are pushing
        to scale weights rather than learn new features.

        Args:
            model: The model (must have gradients computed)

        Returns:
            (overall_cosine, output_layer_cosine)
        """
        total_grad_dot_weight = 0.0
        total_grad_norm_sq = 0.0
        total_weight_norm_sq = 0.0

        output_grad_dot_weight = 0.0
        output_grad_norm_sq = 0.0
        output_weight_norm_sq = 0.0

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                grad = param.grad.flatten()
                weight = param.data.flatten()

                dot = (grad * weight).sum().item()
                grad_norm_sq = (grad * grad).sum().item()
                weight_norm_sq = (weight * weight).sum().item()

                total_grad_dot_weight += dot
                total_grad_norm_sq += grad_norm_sq
                total_weight_norm_sq += weight_norm_sq

                # Check if this is an output layer
                if 'output' in name.lower() or 'lm_head' in name.lower() or 'out_proj' in name.lower():
                    output_grad_dot_weight += dot
                    output_grad_norm_sq += grad_norm_sq
                    output_weight_norm_sq += weight_norm_sq

        # Compute cosine similarities
        overall_cosine = 0.0
        if total_grad_norm_sq > 0 and total_weight_norm_sq > 0:
            overall_cosine = total_grad_dot_weight / (
                np.sqrt(total_grad_norm_sq) * np.sqrt(total_weight_norm_sq)
            )

        output_cosine = 0.0
        if output_grad_norm_sq > 0 and output_weight_norm_sq > 0:
            output_cosine = output_grad_dot_weight / (
                np.sqrt(output_grad_norm_sq) * np.sqrt(output_weight_norm_sq)
            )

        return overall_cosine, output_cosine

    def compute_weight_norm(self, model: nn.Module) -> float:
        """Compute total weight norm of the model."""
        total_norm_sq = 0.0
        with torch.no_grad():
            for param in model.parameters():
                total_norm_sq += (param.data ** 2).sum().item()
        return np.sqrt(total_norm_sq)

    def compute_metrics(
        self,
        model: nn.Module,
        logits: Optional[torch.Tensor],
        step: int,
    ) -> NumericalStabilityMetrics:
        """
        Compute all numerical stability metrics.

        Args:
            model: The model (with gradients if computing alignment)
            logits: Model outputs [batch, seq, vocab] or None
            step: Current training step

        Returns:
            NumericalStabilityMetrics
        """
        metrics = NumericalStabilityMetrics(step=step)

        # Softmax Collapse metrics
        if logits is not None:
            sc_frac, max_logit, logit_range = self.compute_sc_fraction(logits)
            metrics.sc_fraction = sc_frac
            metrics.max_logit_mean = max_logit
            metrics.logit_range_mean = logit_range

        # Gradient-weight alignment
        grad_cosine, output_cosine = self.compute_grad_weight_alignment(model)
        metrics.grad_weight_cosine = grad_cosine
        metrics.grad_weight_cosine_output = output_cosine

        # Weight norm and growth rate
        current_norm = self.compute_weight_norm(model)
        metrics.weight_norm = current_norm

        self.weight_norm_history.append(current_norm)
        if len(self.weight_norm_history) > self.history_size:
            self.weight_norm_history.pop(0)

        if len(self.weight_norm_history) >= 2:
            growth_rate = (self.weight_norm_history[-1] - self.weight_norm_history[-2]) / self.weight_norm_history[-2]
            metrics.weight_norm_growth_rate = growth_rate

        # Risk assessment
        if metrics.sc_fraction > 0.5:
            metrics.sc_risk = "critical"
        elif metrics.sc_fraction > 0.1:
            metrics.sc_risk = "high"
        elif metrics.sc_fraction > 0.01:
            metrics.sc_risk = "medium"
        else:
            metrics.sc_risk = "low"

        metrics.nlm_active = abs(metrics.grad_weight_cosine) > self.nlm_cosine_threshold

        self.last_metrics = metrics
        return metrics


def detect_softmax_collapse_batch(
    logits: torch.Tensor,
    threshold: float = 1e-7,
) -> torch.Tensor:
    """
    Detect which samples in a batch are experiencing softmax collapse.

    Args:
        logits: [batch, vocab] or [batch, seq, vocab]
        threshold: Threshold for SC detection

    Returns:
        Boolean tensor indicating SC for each sample
    """
    with torch.no_grad():
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))

        max_logits, _ = logits.max(dim=-1)
        log_sum_exp = torch.logsumexp(logits, dim=-1)

        sc_indicator = log_sum_exp - max_logits
        sc_threshold_log = np.log(1 + threshold)

        return sc_indicator < sc_threshold_log

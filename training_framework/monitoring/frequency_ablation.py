"""
Frequency Ablation for Grokking Detection.

Implements FFT-based frequency ablation to compute excluded_loss and restricted_loss
metrics, which are leading indicators for grokking phase transitions.

Key concepts:
- excluded_loss: Model performance when key frequencies are REMOVED from embeddings.
  Rising excluded_loss indicates circuit formation (grokking is imminent).
- restricted_loss: Model performance when ONLY key frequencies are kept.
  Decreasing restricted_loss confirms circuits are forming.

The intuition is that grokking models develop structured representations that
concentrate information in specific frequency bands. By ablating these frequencies,
we can detect when this structure is forming before generalization occurs.

References:
- Nanda et al. (2023): "Progress Measures for Grokking"
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting"
"""

import logging
from dataclasses import dataclass
from typing import ClassVar, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAblationConfig:
    """Configuration for frequency ablation."""
    enabled: bool = True
    top_k: int = 10  # Number of key frequencies to identify/ablate


class FrequencyAblator:
    """
    FFT-based frequency ablation for grokking detection.

    Computes excluded_loss and restricted_loss by manipulating the frequency
    content of model embeddings.

    Note: Expected model output is either a tensor of logits or an object with
    a `.logits` attribute (e.g., HuggingFace model outputs).
    """

    VALID_MODES: ClassVar[Set[str]] = {"exclude", "restrict"}

    def __init__(self, config: FrequencyAblationConfig):
        self.config = config
        self._cached_key_freqs: Optional[np.ndarray] = None

    def compute_ablated_losses(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """
        Compute excluded_loss and restricted_loss via frequency ablation.

        Args:
            model: The model to analyze (must have embedding layer)
            input_ids: Input token IDs [batch, seq_len]
            labels: Target labels [batch, seq_len]
            criterion: Loss function (e.g., CrossEntropyLoss)

        Returns:
            Tuple of (excluded_loss, restricted_loss):
            - excluded_loss: Loss when key frequencies REMOVED (should RISE during grokking)
            - restricted_loss: Loss when ONLY key frequencies kept (should DECREASE)
        """
        if not self.config.enabled:
            return 0.0, 0.0

        # Get embedding layer
        embedding_layer = self._get_embedding_layer(model)
        if embedding_layer is None:
            logger.debug("Could not find embedding layer for frequency ablation")
            return 0.0, 0.0

        original_weight = embedding_layer.weight.data.clone()
        was_training = model.training

        try:
            # 1. Identify key frequencies via FFT
            key_freq_indices = self._identify_key_frequencies(original_weight)

            # Set model to eval mode for consistent forward passes
            model.eval()

            # 2. Compute excluded loss (key frequencies zeroed out)
            excluded_weight = self._ablate_frequencies(
                original_weight, key_freq_indices, mode="exclude"
            )
            embedding_layer.weight.data = excluded_weight
            with torch.no_grad():
                excluded_logits = model(input_ids)
                if hasattr(excluded_logits, "logits"):
                    excluded_logits = excluded_logits.logits
                excluded_loss = criterion(
                    excluded_logits.view(-1, excluded_logits.size(-1)),
                    labels.view(-1)
                ).item()

            # 3. Compute restricted loss (only key frequencies kept)
            restricted_weight = self._ablate_frequencies(
                original_weight, key_freq_indices, mode="restrict"
            )
            embedding_layer.weight.data = restricted_weight
            with torch.no_grad():
                restricted_logits = model(input_ids)
                if hasattr(restricted_logits, "logits"):
                    restricted_logits = restricted_logits.logits
                restricted_loss = criterion(
                    restricted_logits.view(-1, restricted_logits.size(-1)),
                    labels.view(-1)
                ).item()

            return excluded_loss, restricted_loss

        except (RuntimeError, ValueError) as e:
            logger.warning(f"Frequency ablation failed: {type(e).__name__}: {e}")
            return 0.0, 0.0
        except Exception as e:
            logger.warning(
                f"Unexpected error in frequency ablation: {type(e).__name__}: {e}",
                exc_info=True
            )
            return 0.0, 0.0

        finally:
            # Always restore original weights and training state
            embedding_layer.weight.data = original_weight
            if was_training:
                model.train()

    def _get_embedding_layer(self, model: nn.Module) -> Optional[nn.Embedding]:
        """Extract the token embedding layer from model.

        Only returns nn.Embedding instances to ensure correct layer type.
        """
        # Try common attribute names
        for attr in ["embed_tokens", "token_embedding", "wte", "embedding"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Embedding):
                    return layer

        # Try nested in transformer
        if hasattr(model, "transformer"):
            return self._get_embedding_layer(model.transformer)
        if hasattr(model, "model"):
            return self._get_embedding_layer(model.model)

        return None

    def _identify_key_frequencies(self, weight: torch.Tensor) -> np.ndarray:
        """
        Identify the top-k key frequency indices via FFT.

        Args:
            weight: Embedding weight [vocab_size, embed_dim]

        Returns:
            Array of frequency indices to ablate
        """
        weight_np = weight.detach().cpu().numpy()
        vocab_size = weight_np.shape[0]

        # Validate top_k against vocab size
        effective_top_k = self.config.top_k
        if vocab_size <= self.config.top_k:
            logger.warning(
                f"vocab_size ({vocab_size}) <= top_k ({self.config.top_k}), "
                f"using vocab_size - 1 = {vocab_size - 1} frequencies"
            )
            effective_top_k = max(1, vocab_size - 1)

        # Apply FFT along vocab dimension
        fft_coeffs = np.fft.fft(weight_np, axis=0)

        # Compute magnitude for each frequency (summed across embed_dim)
        magnitudes = np.sum(np.abs(fft_coeffs), axis=1)

        # Exclude DC component (index 0)
        magnitudes[0] = 0

        # Get top-k frequency indices by magnitude
        key_indices = np.argsort(magnitudes)[-effective_top_k:]

        self._cached_key_freqs = key_indices
        return key_indices

    def _ablate_frequencies(
        self,
        weight: torch.Tensor,
        freq_indices: np.ndarray,
        mode: str,
    ) -> torch.Tensor:
        """
        Create ablated embedding weights.

        Args:
            weight: Original embedding weight [vocab_size, embed_dim]
            freq_indices: Frequency indices to ablate
            mode: "exclude" (zero out these freqs) or "restrict" (keep only these freqs plus DC)

        Returns:
            Ablated weight tensor

        Note:
            In "restrict" mode, the DC component (index 0) is always preserved
            for numerical stability regardless of whether it's in freq_indices.
        """
        weight_np = weight.detach().cpu().numpy()

        # Apply FFT
        fft_coeffs = np.fft.fft(weight_np, axis=0)

        if mode == "exclude":
            # Zero out the key frequencies
            ablated_fft = fft_coeffs.copy()
            ablated_fft[freq_indices] = 0
        elif mode == "restrict":
            # Keep only the key frequencies (and DC for stability)
            ablated_fft = np.zeros_like(fft_coeffs)
            ablated_fft[0] = fft_coeffs[0]  # Keep DC
            ablated_fft[freq_indices] = fft_coeffs[freq_indices]
        else:
            raise ValueError(f"Unknown mode '{mode}', expected one of {self.VALID_MODES}")

        # Inverse FFT to get ablated weights
        ablated_weight = np.fft.ifft(ablated_fft, axis=0).real

        return torch.tensor(
            ablated_weight,
            dtype=weight.dtype,
            device=weight.device,
        )

    def get_cached_key_frequencies(self) -> Optional[np.ndarray]:
        """Return the last computed key frequency indices."""
        return self._cached_key_freqs


def create_frequency_ablator(config_dict: dict) -> FrequencyAblator:
    """Create FrequencyAblator from config dictionary."""
    grok_config = config_dict.get("monitoring", {}).get("grokking", {})
    freq_config = grok_config.get("frequency_ablation", {})

    config = FrequencyAblationConfig(
        enabled=freq_config.get("enabled", True),
        top_k=freq_config.get("top_k", 10),
    )

    return FrequencyAblator(config)

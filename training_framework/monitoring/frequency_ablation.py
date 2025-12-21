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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAblationConfig:
    """Configuration for frequency ablation."""
    enabled: bool = True
    top_k: int = 10  # Number of key frequencies to identify/ablate
    use_magnitude_ranking: bool = True  # Rank by FFT magnitude (vs energy)


class FrequencyAblator:
    """
    FFT-based frequency ablation for grokking detection.

    Computes excluded_loss and restricted_loss by manipulating the frequency
    content of model embeddings.
    """

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

        try:
            # 1. Identify key frequencies via FFT
            key_freq_indices = self._identify_key_frequencies(original_weight)

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

        except Exception as e:
            logger.debug(f"Frequency ablation failed: {e}")
            return 0.0, 0.0

        finally:
            # Always restore original weights
            embedding_layer.weight.data = original_weight

    def _get_embedding_layer(self, model: nn.Module) -> Optional[nn.Embedding]:
        """Extract the token embedding layer from model."""
        # Try common attribute names
        for attr in ["embed_tokens", "token_embedding", "wte", "embedding"]:
            if hasattr(model, attr):
                layer = getattr(model, attr)
                if isinstance(layer, nn.Embedding):
                    return layer
                if hasattr(layer, "weight") and isinstance(layer.weight, torch.Tensor):
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

        # Apply FFT along vocab dimension
        fft_coeffs = np.fft.fft(weight_np, axis=0)

        # Compute magnitude for each frequency (summed across embed_dim)
        magnitudes = np.sum(np.abs(fft_coeffs), axis=1)

        # Exclude DC component (index 0)
        magnitudes[0] = 0

        # Get top-k frequency indices by magnitude
        key_indices = np.argsort(magnitudes)[-self.config.top_k:]

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
            mode: "exclude" (zero out these freqs) or "restrict" (keep only these freqs)

        Returns:
            Ablated weight tensor
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
            raise ValueError(f"Unknown mode: {mode}")

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
        use_magnitude_ranking=freq_config.get("use_magnitude_ranking", True),
    )

    return FrequencyAblator(config)

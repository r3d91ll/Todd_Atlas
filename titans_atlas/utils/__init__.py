"""Utility functions for Titans and Atlas models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    activations = {
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "swish": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name.lower()]


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2 normalize along specified dimension."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_sliding_window_mask(
    seq_len: int,
    window_size: int,
    device: torch.device
) -> torch.Tensor:
    """Create sliding window attention mask."""
    mask = torch.ones(seq_len, seq_len, device=device)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0
    return mask.bool()


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Rotate the last dimension by swapping its two equal halves and negating the first of those halves.
        
        Parameters:
            x (torch.Tensor): Input tensor whose size along the last dimension is even.
        
        Returns:
            torch.Tensor: Tensor with the same shape as `x` where the last-dimension halves are reordered as `[-x2, x1]`.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    RMSNorm is simpler and faster than LayerNorm:
    - No mean centering (no beta parameter)
    - Only scale by RMS: x * weight / RMS(x)

    Used in LLaMA, Mistral, and other modern architectures.
    Typically 10-15% faster than LayerNorm with similar quality.

    Reference: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019)
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm module with a learnable per-dimension scale and numerical-stability epsilon.
        
        Parameters:
            dim (int): Size of the normalized feature dimension; creates a learnable weight of shape (dim,).
            eps (float): Small constant added to the denominator to avoid division by zero.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        """
        Apply Root Mean Square (RMS) normalization to the input using a per-dimension scale.
        
        Parameters:
            x (torch.Tensor): Input tensor with last dimension equal to the RMSNorm `dim`; shape (..., dim).
        
        Returns:
            torch.Tensor: The input normalized by the root-mean-square over the last dimension and scaled by the learnable per-dimension weight; same shape as `x`.
        """
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class DepthwiseConv1d(nn.Module):
    """Depthwise separable 1D convolution for Q, K, V projections."""

    def __init__(self, channels: int, kernel_size: int = 4, padding: str = "causal"):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = padding

        # Depthwise conv
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            (batch, seq_len, channels)
        """
        x = x.transpose(1, 2)  # (batch, channels, seq_len)

        # Causal padding
        if self.padding == "causal":
            x = F.pad(x, (self.kernel_size - 1, 0))

        x = self.conv(x)
        return x.transpose(1, 2)


def parallel_scan(
    gates: torch.Tensor,
    values: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Parallel associative scan for linear recurrences.

    Computes: h_t = g_t * h_{t-1} + v_t

    Args:
        gates: (batch, seq_len, dim) - multiplicative gates g_t
        values: (batch, seq_len, dim) - additive values v_t
        initial_state: (batch, dim) - initial hidden state h_0

    Returns:
        (batch, seq_len, dim) - all hidden states h_1, ..., h_T
    """
    batch, seq_len, dim = gates.shape

    if initial_state is None:
        initial_state = torch.zeros(batch, dim, device=gates.device, dtype=gates.dtype)

    # For short sequences, use sequential scan
    if seq_len <= 32:
        return _sequential_scan(gates, values, initial_state)

    # Parallel scan using associative property
    return _parallel_scan_impl(gates, values, initial_state)


def _sequential_scan(
    gates: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Sequential scan implementation."""
    batch, seq_len, dim = gates.shape
    outputs = []
    h = initial_state

    for t in range(seq_len):
        h = gates[:, t] * h + values[:, t]
        outputs.append(h)

    return torch.stack(outputs, dim=1)


def _parallel_scan_impl(
    gates: torch.Tensor,
    values: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel scan using work-efficient algorithm.

    This implements the Blelloch scan for associative operations.
    """
    batch, seq_len, dim = gates.shape
    device = gates.device
    dtype = gates.dtype

    # Pad to power of 2 for efficiency
    log_n = math.ceil(math.log2(seq_len))
    padded_len = 2 ** log_n

    if padded_len > seq_len:
        pad_size = padded_len - seq_len
        gates = F.pad(gates, (0, 0, 0, pad_size), value=1.0)
        values = F.pad(values, (0, 0, 0, pad_size), value=0.0)

    # Work arrays
    a = gates.clone()  # (batch, padded_len, dim)
    b = values.clone()  # (batch, padded_len, dim)

    # Up-sweep (reduce) phase
    for d in range(log_n):
        stride = 2 ** (d + 1)
        indices = torch.arange(stride - 1, padded_len, stride, device=device)
        prev_indices = indices - 2 ** d

        # a[i] = a[i] * a[i - 2^d]
        # b[i] = b[i] + a[i] * b[i - 2^d]
        a_prev = a[:, prev_indices]
        b_prev = b[:, prev_indices]

        b[:, indices] = b[:, indices] + a[:, indices] * b_prev
        a[:, indices] = a[:, indices] * a_prev

    # Incorporate initial state
    b[:, -1] = b[:, -1] + a[:, -1] * initial_state

    # Down-sweep phase
    a[:, -1] = 1.0
    temp_b = b[:, -1].clone()
    b[:, -1] = initial_state

    for d in range(log_n - 1, -1, -1):
        stride = 2 ** (d + 1)
        indices = torch.arange(stride - 1, padded_len, stride, device=device)
        prev_indices = indices - 2 ** d

        temp_a = a[:, prev_indices].clone()
        temp_b_prev = b[:, prev_indices].clone()

        b[:, prev_indices] = b[:, indices]
        a[:, prev_indices] = a[:, indices]

        b[:, indices] = temp_b_prev + temp_a * b[:, indices]
        a[:, indices] = temp_a * a[:, indices]

    # Apply the recurrence to get final outputs
    outputs = gates * b + values

    return outputs[:, :seq_len]


def chunk_sequence(
    x: torch.Tensor,
    chunk_size: int,
    pad_value: float = 0.0
) -> Tuple[torch.Tensor, int]:
    """
    Split sequence into chunks.

    Args:
        x: (batch, seq_len, dim)
        chunk_size: size of each chunk
        pad_value: value to use for padding

    Returns:
        chunks: (batch, num_chunks, chunk_size, dim)
        original_len: original sequence length
    """
    batch, seq_len, dim = x.shape
    original_len = seq_len

    # Pad to multiple of chunk_size
    if seq_len % chunk_size != 0:
        pad_len = chunk_size - (seq_len % chunk_size)
        x = F.pad(x, (0, 0, 0, pad_len), value=pad_value)
        seq_len = x.shape[1]

    num_chunks = seq_len // chunk_size
    chunks = x.view(batch, num_chunks, chunk_size, dim)

    return chunks, original_len


def unchunk_sequence(
    chunks: torch.Tensor,
    original_len: int
) -> torch.Tensor:
    """
    Merge chunks back into sequence.

    Args:
        chunks: (batch, num_chunks, chunk_size, dim)
        original_len: original sequence length

    Returns:
        (batch, original_len, dim)
    """
    batch, num_chunks, chunk_size, dim = chunks.shape
    x = chunks.view(batch, num_chunks * chunk_size, dim)
    return x[:, :original_len]
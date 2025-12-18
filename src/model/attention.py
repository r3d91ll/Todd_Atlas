"""
Sliding window attention module.

From the Titans/Miras framework:
- Attention serves as "short-term memory"
- Sliding window limits compute while maintaining local context
- Combined with memory module via gating

Reference: Titans paper, Section 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SlidingWindowAttention(nn.Module):
    """
    Multi-head sliding window attention.

    Each position attends only to positions within a fixed window,
    giving O(n · w) complexity instead of O(n²).

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of attention window (one side)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        # QKV projection
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _create_window_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create causal sliding window mask.

        Returns:
            mask: [seq_len, seq_len] with -inf for masked positions
        """
        # Start with causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1,
        )

        # Add window constraint: mask positions beyond window
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            if start > 0:
                mask[i, :start] = float("-inf")

        return mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with sliding window attention.

        Args:
            x: Input [batch, seq_len, d_model]
            mask: Optional additional mask
            return_weights: Whether to return attention weights

        Returns:
            output: Attended output [batch, seq_len, d_model]
            weights: Optional attention weights [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply sliding window mask
        window_mask = self._create_window_mask(seq_len, x.device)
        scores = scores + window_mask.unsqueeze(0).unsqueeze(0)

        # Apply additional mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax and dropout
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Apply attention
        output = torch.matmul(weights, v)

        # Reshape and project
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        if return_weights:
            return output, weights
        return output, None


class GatingMechanism(nn.Module):
    """
    Gating mechanism to combine memory and attention outputs.

    From Titans (MAG variant):
        g = sigmoid(W_g · [mem_out, attn_out])
        output = g * mem_out + (1 - g) * attn_out

    Args:
        d_model: Model dimension
    """

    def __init__(self, d_model: int):
        super().__init__()

        # Gate projection: takes concatenated inputs
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=True)

        self._init_weights()

    def _init_weights(self):
        # Initialize to favor balanced combination
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        mem_out: torch.Tensor,
        attn_out: torch.Tensor,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Combine memory and attention outputs via gating.

        Args:
            mem_out: Memory output [batch, seq_len, d_model]
            attn_out: Attention output [batch, seq_len, d_model]
            return_gate: Whether to return gate values

        Returns:
            output: Gated combination [batch, seq_len, d_model]
            gate: Optional gate values [batch, seq_len, d_model]
        """
        # Concatenate inputs
        combined = torch.cat([mem_out, attn_out], dim=-1)

        # Compute gate
        gate = torch.sigmoid(self.gate_proj(combined))

        # Apply gate
        output = gate * mem_out + (1 - gate) * attn_out

        if return_gate:
            return output, gate
        return output, None


class FeedForward(nn.Module):
    """
    Standard feed-forward network with GELU activation.

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

"""
Attention mechanisms for Atlas.

Implements:
- Sliding Window Attention
- Persistent Memory Attention
- Memory-augmented attention variants

Paper: "Atlas: Learning to Optimally Memorize the Context at Test Time" (arXiv:2505.23735)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math

from titans_atlas.utils import (
    l2_normalize,
    RotaryEmbedding,
    DepthwiseConv1d,
    create_causal_mask,
    create_sliding_window_mask,
    RMSNorm,
)


class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention with optional Flash Attention support.

    Restricts attention to a local window around each position,
    enabling linear memory complexity with sequence length.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_head: int = 64,
        window_size: int = 512,
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_flash: bool = True,
        use_conv: bool = True,
        conv_kernel_size: int = 4,
        use_l2_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.window_size = window_size
        self.use_flash = use_flash
        self.use_l2_norm = use_l2_norm
        self.scale = d_head ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, num_heads * d_head)
        self.k_proj = nn.Linear(d_model, num_heads * d_head)
        self.v_proj = nn.Linear(d_model, num_heads * d_head)
        self.out_proj = nn.Linear(num_heads * d_head, d_model)

        # Optional convolutions after projections
        if use_conv:
            self.q_conv = DepthwiseConv1d(num_heads * d_head, conv_kernel_size)
            self.k_conv = DepthwiseConv1d(num_heads * d_head, conv_kernel_size)
            self.v_conv = DepthwiseConv1d(num_heads * d_head, conv_kernel_size)
        else:
            self.q_conv = self.k_conv = self.v_conv = None

        # Rotary embeddings
        if use_rotary:
            self.rotary = RotaryEmbedding(d_head)
        else:
            self.rotary = None

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
        kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        Forward pass with sliding window attention.

        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional mask (batch, seq_len) or (batch, seq_len, seq_len)
            kv_cache: Optional (k_cache, v_cache) for incremental decoding

        Returns:
            output: (batch, seq_len, d_model)
            new_kv_cache: Updated cache for next step
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Apply convolutions
        if self.q_conv is not None:
            q = self.q_conv(q)
            k = self.k_conv(k)
            v = self.v_conv(v)

        # Reshape to (batch, num_heads, seq_len, d_head)
        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        # Handle KV cache for incremental decoding
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        kv_seq_len = k.shape[2]

        # Apply rotary embeddings
        if self.rotary is not None:
            # Only apply to query positions and matching key positions
            q, k = self.rotary(q, k, kv_seq_len)

        # L2 normalize Q and K (as in paper)
        if self.use_l2_norm:
            q = l2_normalize(q, dim=-1)
            k = l2_normalize(k, dim=-1)

        # Try flash attention first
        if self.use_flash and seq_len > 1:
            try:
                output = self._flash_attention(q, k, v, attention_mask)
            except (RuntimeError, ImportError):
                output = self._sliding_window_attention(q, k, v, attention_mask)
        else:
            output = self._sliding_window_attention(q, k, v, attention_mask)

        # Reshape output
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.num_heads * self.d_head)
        output = self.out_proj(output)

        # Return with KV cache
        new_cache = (k, v) if kv_cache is not None else None
        return output, new_cache

    def _sliding_window_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """Standard sliding window attention implementation."""
        batch, num_heads, q_len, d_head = q.shape
        kv_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create sliding window mask
        window_mask = create_sliding_window_mask(
            q_len, self.window_size, q.device
        )

        # Also apply causal mask
        causal_mask = create_causal_mask(q_len, q.device)

        # Combine masks
        combined_mask = window_mask | causal_mask
        scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply additional attention mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, v)
        return output

    def _flash_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor],
    ) -> Tensor:
        """Flash attention with sliding window (requires flash-attn package)."""
        try:
            from flash_attn import flash_attn_func

            # Flash attention expects (batch, seq_len, num_heads, d_head)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            output = flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(self.window_size, 0),  # Left window only for causal
            )

            return output.transpose(1, 2)
        except ImportError:
            raise ImportError("flash-attn not installed, falling back to standard attention")


class PersistentMemoryAttention(nn.Module):
    """
    Attention with Persistent Memory tokens.

    From the paper: "Learnable but input-independent parameters prepended
    to sequences. These encode task-specific knowledge separate from
    contextual information."

    P = [p_1, p_2, ..., p_np] are prepended to K and V.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_head: int = 64,
        num_persistent: int = 8,
        window_size: int = 512,
        dropout: float = 0.0,
        use_rotary: bool = True,
        use_conv: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.num_persistent = num_persistent
        self.scale = d_head ** -0.5

        # Persistent memory tokens
        self.persistent_k = nn.Parameter(
            torch.randn(1, num_persistent, num_heads * d_head) * 0.02
        )
        self.persistent_v = nn.Parameter(
            torch.randn(1, num_persistent, num_heads * d_head) * 0.02
        )

        # Base sliding window attention
        self.base_attention = SlidingWindowAttention(
            d_model=d_model,
            num_heads=num_heads,
            d_head=d_head,
            window_size=window_size,
            dropout=dropout,
            use_rotary=use_rotary,
            use_conv=use_conv,
        )

        # Separate projections for persistent tokens
        self.q_proj = self.base_attention.q_proj
        self.k_proj = self.base_attention.k_proj
        self.v_proj = self.base_attention.v_proj
        self.out_proj = self.base_attention.out_proj

    def forward(
        self,
        x: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with persistent memory.

        The persistent tokens are prepended to K and V, allowing
        all queries to attend to them in addition to the context.

        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Project input
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Expand persistent memory for batch
        pk = self.persistent_k.expand(batch, -1, -1)
        pv = self.persistent_v.expand(batch, -1, -1)

        # Prepend persistent tokens to K and V
        k = torch.cat([pk, k], dim=1)  # (batch, num_persistent + seq_len, dim)
        v = torch.cat([pv, v], dim=1)

        # Reshape
        q = q.view(batch, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, -1, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, -1, self.num_heads, self.d_head).transpose(1, 2)

        kv_len = k.shape[2]

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Create mask: queries can always attend to persistent tokens
        mask = torch.zeros(seq_len, kv_len, device=x.device, dtype=torch.bool)

        # Apply causal mask only to non-persistent positions
        for i in range(seq_len):
            # Can attend to all persistent tokens
            # For context tokens, apply causal mask
            context_start = self.num_persistent
            for j in range(context_start, kv_len):
                context_j = j - context_start
                if context_j > i:  # Future position
                    mask[i, j] = True

        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply additional mask if provided
        if attention_mask is not None:
            # Expand mask to include persistent tokens (always attended)
            extended_mask = torch.ones(batch, 1, 1, kv_len, device=x.device)
            extended_mask[..., self.num_persistent:] = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(extended_mask == 0, float("-inf"))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Compute output
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch, seq_len, self.num_heads * self.d_head)
        output = self.out_proj(output)

        return output


class GatedAttentionUnit(nn.Module):
    """
    Gated Attention Unit for combining attention and memory outputs.

    Used in MAG (Memory as Gating) variant:
        o = y ⊗ M(x̃)

    Where ⊗ is element-wise multiplication with learned gating.
    """

    def __init__(
        self,
        d_model: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.d_model = d_model

        # Gating mechanism
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.activation = nn.SiLU() if activation == "silu" else nn.GELU()

        # Output normalization (RMSNorm for faster training)
        self.layer_norm = RMSNorm(d_model)

    def forward(
        self,
        attention_output: Tensor,
        memory_output: Tensor,
    ) -> Tensor:
        """
        Combine attention and memory outputs via gating.

        Args:
            attention_output: (batch, seq_len, d_model) from attention branch
            memory_output: (batch, seq_len, d_model) from memory branch

        Returns:
            gated_output: (batch, seq_len, d_model)
        """
        # Concatenate for gate computation
        combined = torch.cat([attention_output, memory_output], dim=-1)

        # Compute gate
        gate = torch.sigmoid(self.gate_proj(combined))

        # Apply gate: blend attention and memory
        output = gate * attention_output + (1 - gate) * memory_output

        # Alternative from paper: o = y ⊗ M(x̃)
        # output = attention_output * self.activation(memory_output)

        return self.layer_norm(output)

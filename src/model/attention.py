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
from enum import Enum
from typing import Optional, Tuple


class GateMode(Enum):
    """
    Gate operation modes for episodic memory training.

    NORMAL: Standard learned gating
    STORAGE: Force high gate values (memory writes)
    RETRIEVAL: Ensure minimum gate values (memory reads)
    """
    NORMAL = "normal"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"


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

    Enhanced with mode-based gate control for episodic memory training:
    - NORMAL: Standard learned gating with optional floor
    - STORAGE: Force high gate values (memory writes)
    - RETRIEVAL: Ensure minimum gate values (memory reads)

    Args:
        d_model: Model dimension
        gate_floor: Minimum gate value (prevents complete memory bypass)
    """

    def __init__(self, d_model: int, gate_floor: float = 0.0):
        super().__init__()

        # Gate projection: takes concatenated inputs
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=True)

        # Mode and floor settings
        self._mode = GateMode.NORMAL
        self._gate_floor = gate_floor

        # Mode-specific targets
        self._storage_gate_target = 0.8  # During storage, push gate toward this
        self._retrieval_gate_floor = 0.3  # During retrieval, minimum gate

        # Store last gate values for metrics
        self._last_raw_gate: Optional[torch.Tensor] = None
        self._last_gate: Optional[torch.Tensor] = None

        self._init_weights()

    def _init_weights(self):
        # Initialize to favor balanced combination
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def set_mode(self, mode: GateMode) -> None:
        """Set the gate operation mode."""
        self._mode = mode

    def get_mode(self) -> GateMode:
        """Get the current gate operation mode."""
        return self._mode

    def set_gate_floor(self, floor: float) -> None:
        """Set the minimum gate value (0.0 to 1.0)."""
        self._gate_floor = max(0.0, min(1.0, floor))

    def get_gate_floor(self) -> float:
        """Get the current gate floor."""
        return self._gate_floor

    def set_storage_target(self, target: float) -> None:
        """Set the target gate value during storage mode."""
        self._storage_gate_target = max(0.0, min(1.0, target))

    def set_retrieval_floor(self, floor: float) -> None:
        """Set the minimum gate value during retrieval mode."""
        self._retrieval_gate_floor = max(0.0, min(1.0, floor))

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

        # Compute raw gate (before mode adjustments)
        raw_gate = torch.sigmoid(self.gate_proj(combined))

        # Store raw gate for metrics
        self._last_raw_gate = raw_gate.detach()

        # Apply mode-specific gate constraints
        if self._mode == GateMode.STORAGE:
            # During storage: push gate toward high value (encourage memory writes)
            gate = torch.max(raw_gate, torch.full_like(raw_gate, self._storage_gate_target))
        elif self._mode == GateMode.RETRIEVAL:
            # During retrieval: ensure minimum gate (encourage memory reads)
            gate = torch.max(raw_gate, torch.full_like(raw_gate, self._retrieval_gate_floor))
        else:
            # Normal mode: apply gate floor
            if self._gate_floor > 0:
                gate = torch.max(raw_gate, torch.full_like(raw_gate, self._gate_floor))
            else:
                gate = raw_gate

        # Store applied gate for metrics
        self._last_gate = gate.detach()

        # Apply gate: high gate = use memory, low gate = use attention
        output = gate * mem_out + (1 - gate) * attn_out

        if return_gate:
            return output, gate
        return output, None

    def get_last_gates(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the last raw and applied gate values.

        Returns:
            Tuple of (raw_gate, applied_gate)
        """
        return self._last_raw_gate, self._last_gate


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

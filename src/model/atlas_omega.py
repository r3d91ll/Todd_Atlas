"""
Atlas model with proper Omega Rule memory.

This version implements the correct Atlas equations from arXiv:2505.23735v1:
- Polynomial features with 1/i! initialization
- Omega rule with context window optimization
- Input-dependent gamma gates for selective token inclusion

Architecture per block:
    Input -> OmegaMemory -> Attention -> Gate -> FFN -> Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

from .omega_memory import OmegaMemory, OmegaMemoryConfig
from .attention import SlidingWindowAttention, GatingMechanism, FeedForward


@dataclass
class AtlasOmegaConfig:
    """Configuration for Atlas model with Omega memory."""

    # Model dimensions
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 4
    d_ff: int = 2048
    vocab_size: int = 32000
    max_seq_len: int = 4096

    # Memory configuration (Omega rule)
    d_key: int = 512
    d_value: int = 512
    poly_degree: int = 2  # Polynomial feature degree
    context_window: int = 16  # Omega rule context window

    # Memory update parameters
    init_alpha: float = 0.99  # Memory decay
    init_theta: float = 0.9   # Momentum coefficient
    init_eta: float = 0.1     # Memory learning rate

    # Attention configuration
    window_size: int = 512

    # Training configuration
    dropout: float = 0.1
    chunk_size: int = 2048

    # Observability
    log_memory_stats: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0
        assert self.d_key == self.d_model, "d_key must equal d_model"
        assert self.d_value == self.d_model, "d_value must equal d_model"


class AtlasOmegaBlock(nn.Module):
    """
    Atlas block with Omega memory.

    Structure:
        1. LayerNorm -> OmegaMemory (with context window + polynomial features)
        2. LayerNorm -> Sliding Window Attention
        3. Gate(memory_out, attention_out)
        4. Residual + LayerNorm -> FFN -> Residual
    """

    def __init__(self, config: AtlasOmegaConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        # Omega Memory module
        mem_config = OmegaMemoryConfig(
            d_model=config.d_model,
            d_key=config.d_key,
            d_value=config.d_value,
            poly_degree=config.poly_degree,
            context_window=config.context_window,
            init_alpha=config.init_alpha,
            init_theta=config.init_theta,
            init_eta=config.init_eta,
        )
        self.memory = OmegaMemory(mem_config)

        # Attention
        self.attention = SlidingWindowAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
            dropout=config.dropout,
        )

        # Gating mechanism
        self.gate = GatingMechanism(config.d_model)

        # Feed-forward
        self.ffn = FeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

        # Dropout for residuals
        self.dropout = nn.Dropout(config.dropout)
        self.memory_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[Tuple] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple, Optional[Dict]]:
        """
        Forward pass through block.

        Args:
            x: Input [batch, seq_len, d_model]
            memory_state: Optional (M, S, context_buffer) tuple
            return_metrics: Whether to return metrics

        Returns:
            output: Block output [batch, seq_len, d_model]
            memory_state: Updated memory state
            metrics: Optional dict of metrics
        """
        batch_size = x.shape[0]
        device = x.device
        metrics = {} if return_metrics else None

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(batch_size, device, x.dtype)

        # === Memory path (with Omega rule) ===
        x_norm = self.norm1(x)
        mem_out, new_memory_state, mem_metrics = self.memory(
            x_norm,
            state=memory_state,
            return_metrics=return_metrics,
        )

        # Apply dropout to memory output
        mem_out = self.memory_dropout(mem_out)

        # === Attention path ===
        x_norm2 = self.norm2(x)
        attn_out, attn_weights = self.attention(x_norm2, return_weights=return_metrics)

        # === Combine via gating ===
        combined, gate_values = self.gate(mem_out, attn_out, return_gate=return_metrics)

        # Residual connection
        x = x + self.dropout(combined)

        # === FFN ===
        x_norm3 = self.norm3(x)
        ffn_out = self.ffn(x_norm3)
        x = x + self.dropout(ffn_out)

        # Collect metrics
        if return_metrics:
            metrics = {
                f"layer_{self.layer_idx}": {
                    "memory": mem_metrics,
                    "gate_mean": gate_values.mean().item() if gate_values is not None else None,
                    "gate_std": gate_values.std().item() if gate_values is not None else None,
                }
            }

        return x, new_memory_state, metrics


class AtlasOmega(nn.Module):
    """
    Full Atlas language model with Omega memory.

    Uses the proper Atlas equations from the paper:
    - Polynomial features with 1/i! coefficients
    - Context window optimization (Omega rule)
    - Input-dependent gamma gates

    Architecture:
        Token Embedding -> Positional Encoding -> N x AtlasOmegaBlock -> LM Head
    """

    def __init__(self, config: AtlasOmegaConfig):
        super().__init__()

        self.config = config

        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AtlasOmegaBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # LM head (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # Weight tying

        # Initialize weights
        self._init_weights()

        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        memory_states: Optional[List[Tuple]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple], Optional[Dict]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            memory_states: Optional list of memory states per layer
            return_metrics: Whether to return metrics

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            memory_states: Updated memory states per layer
            metrics: Optional dict of metrics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Initialize memory states if needed
        if memory_states is None:
            memory_states = [None] * self.config.n_layers

        # Embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Collect metrics
        all_metrics = {} if return_metrics else None
        new_memory_states = []

        # Forward through blocks
        for i, block in enumerate(self.blocks):
            x, new_state, block_metrics = block(
                x,
                memory_state=memory_states[i],
                return_metrics=return_metrics,
            )
            new_memory_states.append(new_state)

            if return_metrics and block_metrics:
                all_metrics.update(block_metrics)

        # Final norm and LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, new_memory_states, all_metrics

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        memory_states: Optional[List] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, List, Optional[Dict]]:
        """
        Compute cross-entropy loss.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            labels: Target token IDs [batch, seq_len]
            memory_states: Optional memory states
            return_metrics: Whether to return metrics

        Returns:
            loss: Scalar loss
            memory_states: Updated memory states
            metrics: Optional metrics including loss/perplexity
        """
        logits, memory_states, metrics = self.forward(
            input_ids,
            memory_states=memory_states,
            return_metrics=return_metrics,
        )

        # Shift for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten
        loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        if return_metrics:
            if metrics is None:
                metrics = {}
            metrics["loss"] = loss.item()
            metrics["perplexity"] = torch.exp(loss).item()

        return loss, memory_states, metrics

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        memory_states = None

        for _ in range(max_new_tokens):
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            logits, memory_states, _ = self.forward(
                input_ids, memory_states=memory_states
            )

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_atlas_omega_50m() -> AtlasOmega:
    """Create ~50M parameter Atlas model with proper Omega memory."""
    config = AtlasOmegaConfig(
        d_model=512,
        n_layers=8,
        n_heads=4,
        d_ff=2048,
        vocab_size=32000,
        max_seq_len=4096,
        d_key=512,
        d_value=512,
        poly_degree=2,          # Quadratic polynomial features
        context_window=16,       # Omega rule context window
        init_alpha=0.99,         # High retention
        init_theta=0.9,          # Standard momentum
        init_eta=0.1,            # Conservative memory LR
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return AtlasOmega(config)


def create_atlas_omega_100m() -> AtlasOmega:
    """Create ~100M parameter Atlas model with proper Omega memory."""
    config = AtlasOmegaConfig(
        d_model=768,
        n_layers=12,
        n_heads=6,
        d_ff=3072,
        vocab_size=32000,
        max_seq_len=4096,
        d_key=768,
        d_value=768,
        poly_degree=2,
        context_window=16,
        init_alpha=0.99,
        init_theta=0.9,
        init_eta=0.1,
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return AtlasOmega(config)

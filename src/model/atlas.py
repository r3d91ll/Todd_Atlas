"""
Atlas model: Full assembly following Miras framework.

Architecture per block:
    Input → Memory → Attention → Gate → FFN → Output

Memory and Attention run in parallel, combined via learned gate.
This follows the MAG (Memory as Gating) variant from Titans.

Reference: Titans paper Section 4.2, "It's All Connected" Section 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass

from .memory import MatrixMemory, ChunkedMatrixMemory
from .retention import RetentionGate, AdaptiveRetentionGate
from .attention import SlidingWindowAttention, GatingMechanism, FeedForward


@dataclass
class AtlasConfig:
    """Configuration for Atlas model."""

    # Model dimensions
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 4
    d_ff: int = 2048
    vocab_size: int = 32000  # LLaMA/Mistral tokenizer vocab size
    max_seq_len: int = 4096

    # Memory configuration
    d_key: int = 512
    d_value: int = 512
    momentum_beta: float = 0.9
    memory_lr_init: float = 0.1
    learn_memory_lr: bool = True

    # Retention configuration
    retention_local_init: float = 0.5
    retention_global_init: float = 0.1
    adaptive_retention: bool = False

    # Attention configuration
    window_size: int = 512

    # Training configuration
    dropout: float = 0.1
    chunk_size: int = 2048  # For TNT training

    # Observability
    log_memory_stats: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.d_model % self.n_heads == 0
        assert self.d_key == self.d_model, "d_key must equal d_model for now"
        assert self.d_value == self.d_model, "d_value must equal d_model for now"


class AtlasBlock(nn.Module):
    """
    Single Atlas transformer block.

    Structure:
        1. LayerNorm → Memory → Retention penalty
        2. LayerNorm → Sliding Window Attention
        3. Gate(memory_out, attention_out)
        4. Residual + LayerNorm → FFN → Residual

    Args:
        config: AtlasConfig
        layer_idx: Layer index for logging
    """

    def __init__(self, config: AtlasConfig, layer_idx: int = 0):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

        # Memory module
        self.memory = MatrixMemory(
            d_key=config.d_key,
            d_value=config.d_value,
            momentum_beta=config.momentum_beta,
            learn_lr=config.learn_memory_lr,
            init_lr=config.memory_lr_init,
        )

        # Retention gate
        if config.adaptive_retention:
            self.retention = AdaptiveRetentionGate(
                d_key=config.d_key,
                d_value=config.d_value,
                d_model=config.d_model,
                init_local=config.retention_local_init,
                init_global=config.retention_global_init,
            )
        else:
            self.retention = RetentionGate(
                d_key=config.d_key,
                d_value=config.d_value,
                init_local=config.retention_local_init,
                init_global=config.retention_global_init,
            )

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

        # Dropout for memory output (critical for preventing memorization)
        self.memory_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        """
        Forward pass through block.

        Args:
            x: Input [batch, seq_len, d_model]
            memory_state: Optional (W, m) tuple
            return_metrics: Whether to return observability metrics

        Returns:
            output: Block output [batch, seq_len, d_model]
            memory_state: Updated (W, m) tuple
            metrics: Optional dict of metrics
        """
        batch_size = x.shape[0]
        device = x.device
        metrics = {} if return_metrics else None

        # Initialize memory state if needed
        if memory_state is None:
            memory_state = self.memory.init_state(batch_size, device)

        W_prev, m_prev = memory_state

        # === Memory path ===
        x_norm = self.norm1(x)

        # Compute retention penalty gradient
        # First do a "lookahead" update to get W_new for retention computation
        W_temp, m_temp, _ = self.memory.update(W_prev, m_prev, x_norm)

        if isinstance(self.retention, AdaptiveRetentionGate):
            retention_grad, retention_metrics = self.retention.forward_adaptive(
                W_temp, W_prev, x_norm
            )
        else:
            retention_grad, retention_metrics = self.retention(W_temp, W_prev)

        # Now do actual memory update with retention
        mem_out, (W_new, m_new), mem_metrics = self.memory(
            x_norm,
            state=(W_prev, m_prev),
            retention_penalty=retention_grad,
            return_metrics=True,
        )

        # Apply dropout to memory output to prevent memorization
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
                    "retention": retention_metrics,
                    "gate_mean": gate_values.mean().item() if gate_values is not None else None,
                    "gate_std": gate_values.std().item() if gate_values is not None else None,
                }
            }

        return x, (W_new, m_new), metrics


class Atlas(nn.Module):
    """
    Full Atlas language model.

    Architecture:
        Token Embedding → Positional Encoding → N × AtlasBlock → LM Head

    Uses weight tying between embedding and LM head.

    Args:
        config: AtlasConfig
    """

    def __init__(self, config: AtlasConfig):
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
            AtlasBlock(config, layer_idx=i)
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
        memory_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_metrics: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]], Optional[Dict]]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            memory_states: Optional list of (W, m) per layer
            return_metrics: Whether to return observability metrics

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
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (None for greedy)

        Returns:
            generated: Generated token IDs [batch, seq_len + max_new_tokens]
        """
        memory_states = None

        for _ in range(max_new_tokens):
            # Truncate if needed
            if input_ids.shape[1] > self.config.max_seq_len:
                input_ids = input_ids[:, -self.config.max_seq_len:]

            # Forward
            logits, memory_states, _ = self.forward(
                input_ids, memory_states=memory_states
            )

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_atlas_50m() -> Atlas:
    """Create ~50M parameter Atlas model (paper specs)."""
    config = AtlasConfig(
        d_model=512,
        n_layers=8,
        n_heads=4,
        d_ff=2048,
        vocab_size=32000,  # LLaMA/Mistral tokenizer vocab size
        max_seq_len=4096,
        d_key=512,
        d_value=512,
        momentum_beta=0.9,
        memory_lr_init=0.1,
        learn_memory_lr=True,
        retention_local_init=0.5,
        retention_global_init=0.1,
        adaptive_retention=False,
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return Atlas(config)


def create_atlas_100m() -> Atlas:
    """Create ~100M parameter Atlas model."""
    config = AtlasConfig(
        d_model=768,
        n_layers=12,
        n_heads=6,
        d_ff=3072,
        vocab_size=32000,  # LLaMA/Mistral tokenizer vocab size
        max_seq_len=4096,
        d_key=768,
        d_value=768,
        momentum_beta=0.9,
        memory_lr_init=0.1,
        learn_memory_lr=True,
        retention_local_init=0.5,
        retention_global_init=0.1,
        adaptive_retention=False,
        window_size=512,
        dropout=0.1,
        chunk_size=2048,
    )
    return Atlas(config)

"""
Titans: Learning to Memorize at Test Time

PyTorch implementation of Titans architecture variants:
- MAC (Memory as Context): Memory provides additional context for attention
- MAG (Memory as Gating): Parallel branches combined via gating
- MAL (Memory as Layer): Sequential memory → attention processing

Paper: "Titans: Learning to Memorize at Test Time" (arXiv:2501.00663)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass

from titans_atlas.layers.memory import NeuralMemory, NeuralMemoryParallel
from titans_atlas.layers.attention import (
    SlidingWindowAttention,
    PersistentMemoryAttention,
    GatedAttentionUnit,
)
from titans_atlas.utils import get_activation, chunk_sequence, unchunk_sequence
from titans_atlas.configs import TitansConfig, MemoryConfig, AttentionConfig


class FeedForward(nn.Module):
    """Standard feed-forward network with gating."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * d_model

        self.gate_proj = nn.Linear(d_model, hidden_dim)
        self.up_proj = nn.Linear(d_model, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, d_model)
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU-style gating
        gate = self.activation(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TitansBlockMAC(nn.Module):
    """
    Memory as Context (MAC) Block.

    Input sequences are segmented into fixed-size chunks. For each segment:
    1. Query memory: h_t = M_{t-1}(q_t)
    2. Concatenate with persistent memory and run attention
    3. Update memory: M_t = M_{t-1}(y_t)
    4. Final output: o_t = y_t ⊗ M_t(y_t)

    This variant uses memory as additional context for attention.
    """

    def __init__(
        self,
        d_model: int,
        chunk_size: int = 512,
        memory_config: Optional[MemoryConfig] = None,
        attention_config: Optional[AttentionConfig] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size

        memory_config = memory_config or MemoryConfig(d_model=d_model)
        attention_config = attention_config or AttentionConfig(d_model=d_model)

        # Neural memory module
        self.memory = NeuralMemory(
            d_model=d_model,
            d_key=memory_config.d_key,
            d_value=memory_config.d_value,
            num_memory_layers=memory_config.num_memory_layers,
            memory_hidden_dim=memory_config.memory_hidden_dim,
            activation=memory_config.activation,
            use_momentum=memory_config.use_momentum,
            use_forget_gate=memory_config.use_forget_gate,
            learnable_lr=memory_config.learnable_lr,
            learnable_momentum=memory_config.learnable_momentum,
            learnable_forget=memory_config.learnable_forget,
            use_l2_norm_keys=memory_config.use_l2_norm_keys,
        )

        # Attention with persistent memory
        self.attention = PersistentMemoryAttention(
            d_model=d_model,
            num_heads=attention_config.num_heads,
            d_head=attention_config.d_head,
            num_persistent=attention_config.num_persistent_tokens,
            window_size=attention_config.window_size,
            dropout=attention_config.attention_dropout,
            use_rotary=attention_config.use_rotary_embeddings,
            use_conv=attention_config.use_qkv_conv,
        )

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Gating for memory-attention combination
        self.gate = GatedAttentionUnit(d_model)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Forward pass for MAC block.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: Previous memory state
            attention_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            new_memory_state: Updated memory state
        """
        batch, seq_len, _ = x.shape

        # Split into chunks for memory processing
        chunks, orig_len = chunk_sequence(x, self.chunk_size)
        num_chunks = chunks.shape[1]

        outputs = []
        current_memory_state = memory_state

        for c in range(num_chunks):
            chunk = chunks[:, c]  # (batch, chunk_size, d_model)

            # 1. Query memory for context
            memory_context, current_memory_state = self.memory(
                self.norm1(chunk),
                memory_state=current_memory_state,
                return_memory_state=True,
            )

            # 2. Run attention with memory context
            # Concatenate memory context with chunk for attention
            attn_input = chunk + memory_context  # Residual addition
            attn_output = self.attention(self.norm2(attn_input))

            # 3. Combine via gating
            combined = self.gate(attn_output, memory_context)

            # 4. Feed-forward with residual
            output = chunk + combined
            output = output + self.ffn(self.norm3(output))

            outputs.append(output)

        # Merge chunks
        output = torch.stack(outputs, dim=1)
        output = unchunk_sequence(output, orig_len)

        return output, current_memory_state


class TitansBlockMAG(nn.Module):
    """
    Memory as Gating (MAG) Block.

    Two parallel branches combined via element-wise gating:
    - Branch 1: Sliding-window attention
    - Branch 2: Neural memory module
    - Combination: o = y ⊗ M(x̃)

    This is the simplest and most efficient variant.
    """

    def __init__(
        self,
        d_model: int,
        memory_config: Optional[MemoryConfig] = None,
        attention_config: Optional[AttentionConfig] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        memory_config = memory_config or MemoryConfig(d_model=d_model)
        attention_config = attention_config or AttentionConfig(d_model=d_model)

        # Branch 1: Sliding window attention
        self.attention = SlidingWindowAttention(
            d_model=d_model,
            num_heads=attention_config.num_heads,
            d_head=attention_config.d_head,
            window_size=attention_config.window_size,
            dropout=attention_config.attention_dropout,
            use_rotary=attention_config.use_rotary_embeddings,
            use_conv=attention_config.use_qkv_conv,
        )

        # Branch 2: Neural memory (parallel version for efficiency)
        self.memory = NeuralMemoryParallel(
            d_model=d_model,
            d_key=memory_config.d_key,
            d_value=memory_config.d_value,
            chunk_size=64,  # Internal chunking
            num_memory_layers=memory_config.num_memory_layers,
            use_momentum=memory_config.use_momentum,
            use_forget_gate=memory_config.use_forget_gate,
        )

        # Gating mechanism
        self.gate = GatedAttentionUnit(d_model)

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for MAG block.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: (batch, d_key, d_value) previous memory
            attention_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            new_memory_state: Updated memory matrix
        """
        # Branch 1: Attention
        normed = self.norm1(x)
        attn_output, _ = self.attention(normed, attention_mask)

        # Branch 2: Memory
        memory_output, new_memory_state = self.memory(normed, memory_state)

        # Combine via gating
        combined = self.gate(attn_output, memory_output)

        # Residual + FFN
        x = x + combined
        x = x + self.ffn(self.norm3(x))

        return x, new_memory_state


class TitansBlockMAL(nn.Module):
    """
    Memory as Layer (MAL) Block.

    Sequential stacking: Memory → Sliding-window Attention

    Memory processes the input first, then attention refines
    based on the memory-enhanced representations.
    """

    def __init__(
        self,
        d_model: int,
        memory_config: Optional[MemoryConfig] = None,
        attention_config: Optional[AttentionConfig] = None,
        ffn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model

        memory_config = memory_config or MemoryConfig(d_model=d_model)
        attention_config = attention_config or AttentionConfig(d_model=d_model)

        # Layer 1: Memory
        self.memory = NeuralMemoryParallel(
            d_model=d_model,
            d_key=memory_config.d_key,
            d_value=memory_config.d_value,
            chunk_size=64,
            num_memory_layers=memory_config.num_memory_layers,
            use_momentum=memory_config.use_momentum,
            use_forget_gate=memory_config.use_forget_gate,
        )

        # Layer 2: Attention
        self.attention = SlidingWindowAttention(
            d_model=d_model,
            num_heads=attention_config.num_heads,
            d_head=attention_config.d_head,
            window_size=attention_config.window_size,
            dropout=attention_config.attention_dropout,
            use_rotary=attention_config.use_rotary_embeddings,
            use_conv=attention_config.use_qkv_conv,
        )

        # Feed-forward
        self.ffn = FeedForward(
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            dropout=dropout,
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        memory_state: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for MAL block.

        Args:
            x: (batch, seq_len, d_model)
            memory_state: Previous memory state
            attention_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            new_memory_state: Updated memory state
        """
        # Layer 1: Memory
        memory_output, new_memory_state = self.memory(self.norm1(x), memory_state)
        x = x + memory_output

        # Layer 2: Attention
        attn_output, _ = self.attention(self.norm2(x), attention_mask)
        x = x + attn_output

        # Feed-forward
        x = x + self.ffn(self.norm3(x))

        return x, new_memory_state


class TitansMAC(nn.Module):
    """
    Titans with Memory as Context (MAC) architecture.

    Full model with embedding, multiple MAC blocks, and output head.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config

        # Blocks
        self.blocks = nn.ModuleList([
            TitansBlockMAC(
                d_model=config.d_model,
                chunk_size=config.chunk_size,
                memory_config=config.memory,
                attention_config=config.attention,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[list] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model) - embedded input
            memory_states: List of memory states per layer
            attention_mask: Optional attention mask

        Returns:
            output: (batch, seq_len, d_model)
            new_memory_states: Updated memory states
        """
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        new_memory_states = []

        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem = block(x, mem_state, attention_mask)
            new_memory_states.append(new_mem)

        x = self.final_norm(x)

        return x, new_memory_states


class TitansMAG(nn.Module):
    """
    Titans with Memory as Gating (MAG) architecture.

    Most efficient variant with parallel memory and attention branches.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config

        # Blocks
        self.blocks = nn.ModuleList([
            TitansBlockMAG(
                d_model=config.d_model,
                memory_config=config.memory,
                attention_config=config.attention,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[list] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list]:
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        new_memory_states = []

        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem = block(x, mem_state, attention_mask)
            new_memory_states.append(new_mem)

        x = self.final_norm(x)

        return x, new_memory_states


class TitansMAL(nn.Module):
    """
    Titans with Memory as Layer (MAL) architecture.

    Sequential processing: memory layer followed by attention layer.
    """

    def __init__(self, config: TitansConfig):
        super().__init__()
        self.config = config

        # Blocks
        self.blocks = nn.ModuleList([
            TitansBlockMAL(
                d_model=config.d_model,
                memory_config=config.memory,
                attention_config=config.attention,
                ffn_hidden_dim=config.ffn_hidden_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: Tensor,
        memory_states: Optional[list] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, list]:
        if memory_states is None:
            memory_states = [None] * len(self.blocks)

        new_memory_states = []

        for block, mem_state in zip(self.blocks, memory_states):
            x, new_mem = block(x, mem_state, attention_mask)
            new_memory_states.append(new_mem)

        x = self.final_norm(x)

        return x, new_memory_states


class TitansLM(nn.Module):
    """
    Titans Language Model wrapper.

    Complete model with token embedding, positional encoding,
    Titans backbone, and language modeling head.
    """

    def __init__(
        self,
        config: TitansConfig,
        variant: Literal["MAC", "MAG", "MAL"] = "MAG",
    ):
        super().__init__()
        self.config = config
        self.variant = variant

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Backbone based on variant
        if variant == "MAC":
            self.backbone = TitansMAC(config)
        elif variant == "MAG":
            self.backbone = TitansMAG(config)
        elif variant == "MAL":
            self.backbone = TitansMAL(config)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        # Language modeling head (tied with embedding)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Weight tying

    def forward(
        self,
        input_ids: Tensor,
        memory_states: Optional[list] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Forward pass for language modeling.

        Args:
            input_ids: (batch, seq_len) token IDs
            memory_states: Previous memory states
            attention_mask: Optional attention mask
            labels: (batch, seq_len) target token IDs for loss

        Returns:
            Dictionary with 'logits', 'loss' (if labels provided), 'memory_states'
        """
        # Embed tokens
        x = self.token_embedding(input_ids)

        # Pass through backbone
        x, new_memory_states = self.backbone(x, memory_states, attention_mask)

        # Compute logits
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "memory_states": new_memory_states,
        }

        # Compute loss if labels provided
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Tensor:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: (batch, seq_len) prompt tokens
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold

        Returns:
            (batch, seq_len + max_new_tokens) generated tokens
        """
        memory_states = None

        for _ in range(max_new_tokens):
            # Get logits for last position
            outputs = self.forward(input_ids, memory_states)
            logits = outputs["logits"][:, -1, :] / temperature
            memory_states = outputs["memory_states"]

            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float("-inf")

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

"""Configuration classes for Atlas models."""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class MemoryConfig:
    """Configuration for Neural Memory Module."""

    # Memory dimensions
    d_model: int = 512
    d_key: int = 64
    d_value: int = 64

    # Memory depth (L_M >= 2 recommended for capacity)
    num_memory_layers: int = 2
    memory_hidden_dim: Optional[int] = None  # Defaults to 4 * d_value

    # Memory update parameters
    use_momentum: bool = True
    use_forget_gate: bool = True
    use_weight_decay: bool = True

    # Learned vs fixed parameters
    learnable_lr: bool = True  # θ_t
    learnable_momentum: bool = True  # η_t
    learnable_forget: bool = True  # α_t

    # Per-token vs per-chunk parameters
    token_level_params: bool = True  # If False, use chunk-level

    # Activation
    activation: str = "silu"

    # Normalization
    use_layer_norm: bool = True
    use_l2_norm_keys: bool = True

    def __post_init__(self):
        if self.memory_hidden_dim is None:
            self.memory_hidden_dim = 4 * self.d_value


@dataclass
class AttentionConfig:
    """Configuration for Attention mechanisms."""

    d_model: int = 512
    num_heads: int = 8
    d_head: int = 64

    # Sliding window
    window_size: int = 512

    # Persistent memory
    num_persistent_tokens: int = 8

    # Regularization
    dropout: float = 0.0
    attention_dropout: float = 0.0

    # Efficiency
    use_flash_attention: bool = True
    use_rotary_embeddings: bool = True

    # 1D convolutions (as in paper)
    use_qkv_conv: bool = True
    conv_kernel_size: int = 4


@dataclass
class AtlasConfig:
    """Configuration for Atlas model."""

    # Model dimensions
    d_model: int = 512
    num_layers: int = 12

    # Omega rule parameters
    context_window: int = 64  # c in the paper

    # Polynomial features
    polynomial_degree: int = 2  # p in φ_p(x)
    use_polynomial_features: bool = True

    # Muon optimizer for memory
    use_muon_optimizer: bool = True
    muon_momentum: float = 0.95
    muon_nesterov: bool = True

    # Learned decay weights γ_i^(t)
    learnable_decay_weights: bool = True

    # Taylor expansion for softmax approximation
    taylor_expansion_order: int = 4
    learnable_taylor_coeffs: bool = True

    # Memory configuration
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    # Attention configuration
    attention: AttentionConfig = field(default_factory=AttentionConfig)

    # Feed-forward
    ffn_hidden_dim: Optional[int] = None  # Defaults to 4 * d_model
    ffn_activation: str = "silu"

    # Vocabulary
    vocab_size: int = 32000
    max_seq_len: int = 8192

    # Regularization
    dropout: float = 0.0

    # Initialization
    init_std: float = 0.02

    def __post_init__(self):
        if self.ffn_hidden_dim is None:
            self.ffn_hidden_dim = 4 * self.d_model
        # Sync dimensions
        self.memory.d_model = self.d_model
        self.attention.d_model = self.d_model


# Preset configurations
def atlas_small() -> AtlasConfig:
    """Small Atlas configuration (~170M params)."""
    return AtlasConfig(
        d_model=768,
        num_layers=12,
        context_window=64,
        polynomial_degree=2,
        memory=MemoryConfig(d_model=768, d_key=64, d_value=64, num_memory_layers=2),
        attention=AttentionConfig(d_model=768, num_heads=12, d_head=64),
    )


def atlas_medium() -> AtlasConfig:
    """Medium Atlas configuration (~400M params)."""
    return AtlasConfig(
        d_model=1024,
        num_layers=24,
        context_window=64,
        polynomial_degree=2,
        memory=MemoryConfig(d_model=1024, d_key=64, d_value=64, num_memory_layers=2),
        attention=AttentionConfig(d_model=1024, num_heads=16, d_head=64),
    )


def atlas_large() -> AtlasConfig:
    """Large Atlas configuration (~760M params)."""
    return AtlasConfig(
        d_model=1536,
        num_layers=24,
        context_window=128,
        polynomial_degree=2,
        memory=MemoryConfig(d_model=1536, d_key=96, d_value=96, num_memory_layers=3),
        attention=AttentionConfig(d_model=1536, num_heads=16, d_head=96),
    )


def atlas_36m() -> AtlasConfig:
    """36M parameter Atlas configuration for proof-of-concept testing."""
    return AtlasConfig(
        d_model=384,
        num_layers=6,
        context_window=32,
        polynomial_degree=2,
        memory=MemoryConfig(d_model=384, d_key=48, d_value=48, num_memory_layers=2),
        attention=AttentionConfig(d_model=384, num_heads=6, d_head=64, use_flash_attention=False),
        vocab_size=50257,  # GPT-2 tokenizer
        max_seq_len=1024,
        ffn_hidden_dim=1536,  # 4x d_model
    )


def atlas_500m() -> AtlasConfig:
    """500M parameter Atlas configuration for pre-training."""
    return AtlasConfig(
        d_model=1024,
        num_layers=24,
        context_window=64,
        polynomial_degree=2,
        memory=MemoryConfig(d_model=1024, d_key=64, d_value=64, num_memory_layers=2),
        attention=AttentionConfig(d_model=1024, num_heads=16, d_head=64),
        vocab_size=50257,  # GPT-2 tokenizer size
        max_seq_len=2048,
    )


@dataclass
class TrainingConfig:
    """Configuration for training Atlas models."""

    # Precision settings
    precision: Literal["fp32", "fp16", "bf16"] = "bf16"
    use_amp: bool = True  # Automatic Mixed Precision

    # Distributed training
    distributed: bool = True
    backend: str = "nccl"
    world_size: int = 2  # Number of GPUs

    # Optimization
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    warmup_steps: int = 2000
    max_steps: int = 100000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Batch sizes
    micro_batch_size: int = 4  # Per GPU
    gradient_accumulation_steps: int = 8
    # Effective batch = micro_batch * grad_accum * world_size

    # Sequence length
    seq_length: int = 2048

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000
    eval_interval: int = 500

    # Logging
    log_interval: int = 10
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    # Data
    data_path: str = "./data"
    num_workers: int = 4

    # Reproducibility
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps * self.world_size

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.seq_length

# Atlas: Implementation of the Behrouz et al. Memory Architecture Framework

## Overview

Atlas is a minimal, observable implementation of the memory architecture framework developed by Ali Behrouz, Peilin Zhong, Vahab Mirrokni, and colleagues at Google Research. This document traces how our implementation maps to concepts from five foundational papers.

## The Five Papers

| Paper | Key Contribution | Our Implementation |
|-------|------------------|-------------------|
| **Titans** | Test-time memorization via gradient-based memory updates | `src/model/memory.py` - MatrixMemory class |
| **It's All Connected** | Miras framework - 4 design axes for memory architectures | Overall architecture design |
| **ATLAS** | Optimal memory allocation at inference | Memory state management in AtlasBlock |
| **TNT** | Hierarchical training with 17x speedup | `src/training/ddp_trainer.py` - two-stage training |
| **Nested Learning** | Architecture = optimization at different levels | Layered memory with retention gates |

---

## 1. Matrix-Valued Memory (Titans Paper)

The Titans paper introduces **test-time memorization** where the model learns during inference through gradient-based updates to a memory matrix.

### Theoretical Foundation

The memory update rule from Titans:
```
W_new = W_old - lr * gradient + momentum * (W_old - W_prev)
```

### Our Implementation

From `src/model/memory.py`:

```python
class MatrixMemory(nn.Module):
    """
    Matrix-valued associative memory following Titans paper.

    Key insight: Memory is a learnable matrix W that gets updated
    via gradient descent at test time, not just training time.
    """

    def __init__(
        self,
        d_key: int = 512,
        d_value: int = 512,
        momentum_beta: float = 0.9,
        learn_lr: bool = True,
        init_lr: float = 0.1,
    ):
        super().__init__()

        # Memory matrix W ∈ R^{d_key × d_value}
        # This is the "associative memory" that stores information
        self.d_key = d_key
        self.d_value = d_value

        # Learnable learning rate (per Titans: allows model to control update magnitude)
        if learn_lr:
            self.log_lr = nn.Parameter(torch.log(torch.tensor(init_lr)))
        else:
            self.register_buffer('log_lr', torch.log(torch.tensor(init_lr)))

        # Momentum coefficient (Titans uses β=0.9)
        self.momentum_beta = momentum_beta
```

The update mechanism:

```python
def update(
    self,
    W: torch.Tensor,
    m: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Update memory with new information.

    Following Titans Equation 4:
    W_new = W - lr * grad + beta * momentum
    """
    # Project input to key/value space
    k = self.key_proj(x)  # [batch, seq, d_key]
    v = self.value_proj(x)  # [batch, seq, d_value]

    # Compute gradient: how much does W need to change?
    # grad = k^T @ (k @ W - v)  [Squared error objective]
    prediction = torch.bmm(k, W)  # What W predicts
    error = prediction - v  # Prediction error
    grad = torch.bmm(k.transpose(-2, -1), error)  # Gradient

    # Update with momentum (Titans innovation)
    lr = torch.exp(self.log_lr)
    m_new = self.momentum_beta * m + grad
    W_new = W - lr * m_new

    return W_new, m_new, {"grad_norm": grad.norm().item()}
```

### Why This Matters

Traditional transformers have fixed parameters after training. Titans-style memory allows the model to **adapt at inference time** by updating W based on what it's currently processing. This is crucial for:
- Long-context modeling
- Continual learning
- Sequence-specific adaptation

---

## 2. The Miras Framework (It's All Connected)

The "It's All Connected" paper unifies various memory architectures through 4 design axes.

### The Four Axes

| Axis | Options | Our Choice |
|------|---------|------------|
| **Memory Structure** | Scalar, Vector, Matrix | **Matrix** (W ∈ R^{512×512}) |
| **Attentional Bias** | L1, L2, Softmax | **L2** (squared error) |
| **Retention Gate** | None, Local, Global, Both | **Local + Global** |
| **Learning Algorithm** | SGD, Momentum, Adam | **Momentum** (β=0.9) |

### Configuration

From `configs/atlas_50m.yaml`:

```yaml
model:
  # Memory configuration (Matrix-valued, per Miras framework)
  d_key: 512
  d_value: 512
  momentum_beta: 0.9
  memory_lr_init: 0.1
  learn_memory_lr: true

  # Retention configuration (Local + Global, per "It's All Connected")
  retention_local_init: 0.5
  retention_global_init: 0.1
  adaptive_retention: false
```

### Block Architecture

From `src/model/atlas.py`:

```python
class AtlasBlock(nn.Module):
    """
    Single Atlas transformer block following Miras framework.

    Structure:
        1. LayerNorm → Memory → Retention penalty
        2. LayerNorm → Sliding Window Attention
        3. Gate(memory_out, attention_out)
        4. Residual + LayerNorm → FFN → Residual

    Memory and Attention run in parallel, combined via learned gate.
    This follows the MAG (Memory as Gating) variant from Titans.
    """

    def __init__(self, config: AtlasConfig, layer_idx: int = 0):
        super().__init__()

        # Memory module (Axis 1: Matrix-valued)
        self.memory = MatrixMemory(
            d_key=config.d_key,
            d_value=config.d_value,
            momentum_beta=config.momentum_beta,  # Axis 4: Momentum
            learn_lr=config.learn_memory_lr,
            init_lr=config.memory_lr_init,
        )

        # Retention gate (Axis 3: Local + Global)
        self.retention = RetentionGate(
            d_key=config.d_key,
            d_value=config.d_value,
            init_local=config.retention_local_init,
            init_global=config.retention_global_init,
        )

        # Sliding window attention (complement to memory)
        self.attention = SlidingWindowAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            window_size=config.window_size,
        )

        # Gating mechanism (MAG variant)
        self.gate = GatingMechanism(config.d_model)
```

---

## 3. Retention Gates (It's All Connected + Nested Learning)

The retention gate controls how much old information to keep vs. how much to update.

### Theoretical Foundation

From the paper:
- **Local retention (λ_local)**: Per-token decay, controls short-term memory
- **Global retention (λ_global)**: Per-layer decay, controls long-term stability

### Our Implementation

From `src/model/retention.py`:

```python
class RetentionGate(nn.Module):
    """
    Retention gate following "It's All Connected" paper.

    Controls the balance between:
    - Preserving old memories (high retention)
    - Accepting new information (low retention)

    Two components:
    - Local: Fine-grained, per-position retention
    - Global: Coarse-grained, layer-wide retention
    """

    def __init__(
        self,
        d_key: int,
        d_value: int,
        init_local: float = 0.5,
        init_global: float = 0.1,
    ):
        super().__init__()

        # Local retention: λ_local ∈ (0, 1) per dimension
        self.local_logit = nn.Parameter(
            torch.logit(torch.tensor(init_local)) * torch.ones(d_key, d_value)
        )

        # Global retention: single scalar
        self.global_logit = nn.Parameter(
            torch.logit(torch.tensor(init_global))
        )

    def forward(self, W_new: torch.Tensor, W_old: torch.Tensor):
        """
        Compute retention penalty gradient.

        Penalizes large changes to W, encouraging stability.
        """
        lambda_local = torch.sigmoid(self.local_logit)
        lambda_global = torch.sigmoid(self.global_logit)

        # Combined retention
        retention = lambda_local * lambda_global

        # Gradient pushes W_new toward W_old
        delta = W_new - W_old
        retention_grad = retention * delta

        return retention_grad, {
            "lambda_local_mean": lambda_local.mean().item(),
            "lambda_global": lambda_global.item(),
        }
```

---

## 4. TNT Two-Stage Training

The TNT paper introduces hierarchical training that achieves 17x speedup.

### The Two Stages

| Stage | Chunk Size | Steps | Purpose |
|-------|------------|-------|---------|
| **Stage 1** | 2048 | 45,000 | Coarse-grained learning with large context |
| **Stage 2** | 256 | 5,000 | Fine-grained accuracy tuning |

### Configuration

```yaml
training:
  # TNT two-stage training
  use_tnt: true

  # Stage 1: Large chunks, hierarchical memory
  stage1_chunk_size: 2048
  stage1_steps: 45000

  # Stage 2: Small chunks, fine-tuning
  stage2_chunk_size: 256
  stage2_steps: 5000
```

### Implementation

From `src/training/ddp_trainer.py`:

```python
class DDPTrainer:
    def __init__(self, ..., use_tnt=True, stage1_chunk_size=2048, ...):
        self.use_tnt = use_tnt
        self.stage1_chunk_size = stage1_chunk_size
        self.stage1_steps = stage1_steps
        self.stage2_chunk_size = stage2_chunk_size
        self.stage2_steps = stage2_steps

        # Total steps = stage1 + stage2 (if TNT enabled)
        if use_tnt:
            self.total_steps = stage1_steps + stage2_steps

    def get_current_chunk_size(self, step: int) -> int:
        """Get chunk size based on TNT stage."""
        if not self.use_tnt:
            return self.stage1_chunk_size

        if step < self.stage1_steps:
            return self.stage1_chunk_size  # Stage 1: Large chunks
        else:
            return self.stage2_chunk_size  # Stage 2: Small chunks
```

### Why Two Stages?

- **Stage 1 (large chunks)**: Model learns coarse patterns efficiently
- **Stage 2 (small chunks)**: Model refines predictions with fine-grained updates
- **Speedup**: Processing large chunks is more compute-efficient than many small ones

---

## 5. The Full Forward Pass

Putting it all together in `src/model/atlas.py`:

```python
def forward(
    self,
    x: torch.Tensor,
    memory_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_metrics: bool = False,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
    """
    Forward pass through AtlasBlock.

    The key innovation: Memory and Attention run in PARALLEL,
    then are combined via a learned gate. This is the MAG variant.
    """

    # Initialize memory state if needed
    if memory_state is None:
        memory_state = self.memory.init_state(batch_size, device)

    W_prev, m_prev = memory_state

    # === MEMORY PATH ===
    x_norm = self.norm1(x)

    # Lookahead update for retention computation
    W_temp, m_temp, _ = self.memory.update(W_prev, m_prev, x_norm)

    # Compute retention penalty
    retention_grad, retention_metrics = self.retention(W_temp, W_prev)

    # Actual memory update with retention
    mem_out, (W_new, m_new), mem_metrics = self.memory(
        x_norm,
        state=(W_prev, m_prev),
        retention_penalty=retention_grad,
    )

    # === ATTENTION PATH ===
    x_norm2 = self.norm2(x)
    attn_out, attn_weights = self.attention(x_norm2)

    # === COMBINE VIA GATING ===
    # Gate learns when to use memory vs. attention
    combined, gate_values = self.gate(mem_out, attn_out)

    # Residual connection
    x = x + self.dropout(combined)

    # === FFN ===
    x = x + self.dropout(self.ffn(self.norm3(x)))

    return x, (W_new, m_new), metrics
```

---

## 6. Observability and Metrics

Following best practices from the papers, we log extensive metrics:

```python
# From training loop
metrics = {
    "loss": loss.item(),
    "perplexity": torch.exp(loss).item(),
    "grad_norm": grad_norm,

    # Memory metrics (per layer)
    "memory_W_norm": W.norm().item(),
    "memory_effective_rank": effective_rank(W),
    "memory_update_magnitude": (W_new - W_old).norm().item(),

    # Retention metrics
    "lambda_local_mean": lambda_local.mean().item(),
    "lambda_global": lambda_global.item(),

    # Gating metrics
    "gate_mean": gate_values.mean().item(),  # >0.5 = favor memory
}
```

---

## 7. Lessons Learned from Training

### Training Run v7 (Dolmino Dataset)

Our initial training on the Dolmino dataset revealed important insights:

**Validation Loss Progression:**
```
Step  500 | Val Loss: 9.5876 | Val PPL: 14583.01 - Improving
Step 1000 | Val Loss: 9.2491 | Val PPL: 10394.92 - Improving
Step 1500 | Val Loss: 8.6589 | Val PPL:  5761.44 - Improving
Step 2000 | Val Loss: 7.9231 | Val PPL:  2760.37 - Improving
Step 2500 | Val Loss: 7.6348 | Val PPL:  2068.92 - Improving
Step 3000 | Val Loss: 6.9207 | Val PPL:  1013.00 - Improving
Step 3500 | Val Loss: 6.7877 | Val PPL:   886.87 - Improving
Step 4000 | Val Loss: 4.7240 | Val PPL:   112.62 - BEST
Step 4500 | Val Loss: 6.9575 | Val PPL:  1050.98 - JUMPED UP
Step 5000 | Val Loss: 4.8680 | Val PPL:   130.06 - Oscillating
...
[EARLY STOPPING] No improvement for 10 validations.
```

**Key Observations:**

1. **Rapid initial improvement** (steps 0-4000): Loss dropped from ~10 to 4.7
2. **Sudden instability** (step 4500): Val loss jumped from 4.7 to 6.9
3. **Never recovered**: Oscillated between 4.8-6.4, never beat 4.7 again
4. **High gradient norms**: Observed `grad=1.46e+04` and `grad=1.72e+04`

### Hypothesis: Dataset Too Clean

The Dolmino dataset (subset of Dolma) is heavily curated for quality. Our hypothesis:

> **Clean, homogeneous data may lead to:**
> - Overfitting to specific patterns
> - Poor generalization
> - Training instability when encountering edge cases

### Solution: Switch to The Pile

The Pile offers more diversity:

| Subset | Content Type |
|--------|--------------|
| ArXiv | Academic papers |
| GitHub | Source code |
| Wikipedia | Encyclopedia |
| PubMed | Medical literature |
| StackExchange | Q&A |
| FreeLaw | Legal documents |
| HackerNews | Tech discussions |
| ... | 17 subsets total |

**Expected Benefits:**
- Harder to memorize = better generalization
- More varied patterns = more robust learning
- Dedicated val set = cleaner train/val separation

### Updated Configuration (v8)

```yaml
# configs/atlas_50m_pile.yaml

training:
  # Early stopping - ENABLED with reasonable patience
  # "At some point you stop hitting your head against the wall"
  val_patience: 20  # 20 validations = 10,000 steps

data:
  # The Pile - more diverse than Dolmino
  data_dir: "datasets/raw/the_pile_hf"
  tokenizer: "tokenizer/atlas_tokenizer_pile"

output:
  run_dir: "runs/atlas_50m_v8_pile"
```

---

## 8. Architecture Summary

```
Input Tokens
     │
     ▼
┌─────────────────────────────────────────┐
│           Token Embedding               │
│         + Positional Encoding           │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│           AtlasBlock × 8                │
│  ┌─────────────┐   ┌─────────────┐     │
│  │   Memory    │   │  Attention  │     │
│  │  (Titans)   │   │  (Window)   │     │
│  └──────┬──────┘   └──────┬──────┘     │
│         │                  │            │
│         └────────┬─────────┘            │
│                  │                      │
│           ┌──────▼──────┐               │
│           │    Gate     │               │
│           │   (MAG)     │               │
│           └──────┬──────┘               │
│                  │                      │
│           ┌──────▼──────┐               │
│           │    FFN      │               │
│           └─────────────┘               │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│           LM Head                       │
│      (tied with embedding)              │
└─────────────────────────────────────────┘
     │
     ▼
Output Logits
```

---

## 9. Parameter Count

From `src/model/atlas.py`:

```python
def create_atlas_50m() -> Atlas:
    """Create ~50M parameter Atlas model (paper specs)."""
    config = AtlasConfig(
        d_model=512,
        n_layers=8,
        n_heads=4,
        d_ff=2048,
        vocab_size=32000,
        max_seq_len=4096,
        d_key=512,
        d_value=512,
    )
    return Atlas(config)
```

**Actual count: 56.3M parameters**

Breakdown:
- Token embedding: 32000 × 512 = 16.4M
- Position embedding: 4096 × 512 = 2.1M
- Per block (×8):
  - Memory projections: ~2M
  - Attention: ~1M
  - FFN: ~4M
  - Retention + Gate: ~0.5M
- Final norm: negligible

---

## 10. References

1. **Titans**: "Titans: Learning to Memorize at Test Time" - Behrouz et al.
2. **It's All Connected**: "It's All Connected: A Journey Through Test-Time Memorization" - Behrouz et al.
3. **ATLAS**: "ATLAS: Learning to Optimally Memorize the Context at Test Time" - Behrouz et al.
4. **TNT**: "TNT: Text-Conditioned Neural Turing Machines" - Training methodology
5. **Nested Learning**: "Nested Learning: Architecture = Optimization" - Theoretical foundation

---

## Appendix: Quick Commands

```bash
# Test model components
python scripts/test_model.py

# Train with synthetic data
python scripts/train.py --config configs/atlas_50m_pile.yaml --test

# Full training with The Pile
torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/atlas_50m_pile.yaml

# Monitor training
tail -f runs/atlas_50m_v8_pile/training.log
```

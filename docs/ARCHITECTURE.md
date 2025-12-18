# Atlas 50M Architecture Design

## Framework: Miras (from "It's All Connected")

The Miras framework defines sequence models through four design axes:

```
Model = f(Memory Architecture, Attentional Bias, Retention Gate, Learning Algorithm)
```

This document specifies our choices for each axis.

---

## 1. Memory Architecture: Matrix-Valued

### Choice: W ∈ R^{d_k × d_v}

**Why matrix over MLP:**
- More observable (can directly inspect W)
- Sufficient capacity for 50M model
- Simpler gradient dynamics
- Clear connection to associative memory theory

**Memory update (from paper):**
```
W_t = W_{t-1} + η_t · (v_t - W_{t-1} · k_t) · k_t^T
```

Where:
- `k_t ∈ R^{d_k}` is the key (projected from input)
- `v_t ∈ R^{d_v}` is the value (projected from input)
- `η_t` is the learning rate (can be learned or fixed)
- `(v_t - W_{t-1} · k_t)` is the prediction error

**Retrieval:**
```
output_t = W_t · q_t
```

Where `q_t` is the query (projected from input).

### Q-K Projection (from TNT)

To fix domain mismatch between compression and retrieval:
```
q_projected = Project(q, KeySpace)
output_t = W_t · q_projected
```

This ensures retrieval operates in the learned key domain.

---

## 2. Attentional Bias: L2 (Baseline)

### Choice: ℓ(W; k, v) = ||W·k - v||²₂

**Why L2:**
- Standard baseline from paper
- Well-understood optimization dynamics
- Closed-form solutions available for analysis
- Prove baseline works before trying Huber, Lp, KL alternatives

**Connection to associative memory:**
The L2 objective treats memory as regression: minimize squared error between retrieved value and target value.

---

## 3. Retention Gate: Local + Global

### From the paper's Learning-Retaining viewpoint:

```
W_t = argmin_W [ℓ(W; k_t, v_t) + Ret_t(W, W_{t-1})]
```

Where `Ret_t` is the retention regularizer balancing new learning vs. preserving past.

### Local Retention (proximity to previous state):
```
Ret_local = λ_local · ||W - W_{t-1}||²_F
```

### Global Retention (memory size normalization):
```
Ret_global = λ_global · ||W||²_F
```

### Combined:
```
Ret_t = Ret_local + Ret_global
```

**Key insight from paper:** This reframes "forgetting" as "retention regularization" - we're not deciding what to forget, we're deciding how strongly to retain.

### Learnable Retention Coefficients

```python
λ_local = sigmoid(learned_param_local)   # Per-layer or global
λ_global = sigmoid(learned_param_global)
```

---

## 4. Learning Algorithm: Momentum

### From TNT: Momentum is essential for stability

```
g_t = ∇_W ℓ(W_{t-1}; k_t, v_t)           # Gradient
m_t = β · m_{t-1} + (1 - β) · g_t         # Momentum
W_t = W_{t-1} - η · m_t + Ret_t           # Update with retention
```

**Parameters:**
- `β = 0.9` (momentum coefficient)
- `η` learnable per-layer or fixed schedule

### Why not Adam?
- Adam adds complexity (second moment)
- Momentum sufficient per TNT experiments
- Keep it simple for observability

---

## Model Architecture

### Block Structure (per layer)

```
Input x_t
    │
    ├──► Key Projection ──► k_t
    ├──► Value Projection ──► v_t
    ├──► Query Projection ──► q_t
    │
    ▼
┌─────────────────────────────────────┐
│  Memory Module                       │
│  1. Update: W_t = Update(W_{t-1},   │
│             k_t, v_t, retention)     │
│  2. Retrieve: mem_out = W_t · q_t   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Sliding Window Attention            │
│  attn_out = Attention(x_t, window)  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  Gating                              │
│  g = sigmoid(W_g · [mem, attn])     │
│  combined = g * mem_out +           │
│             (1-g) * attn_out        │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  FFN (standard)                      │
│  out = FFN(combined) + residual     │
└─────────────────────────────────────┘
    │
    ▼
Output
```

### 50M Parameter Budget

| Component | Parameters | Notes |
|-----------|------------|-------|
| Embedding | 32K × 512 = 16.4M | Vocab 32K, dim 512 |
| Memory (×8 layers) | 8 × (512×512 + 512×512) = 4.2M | K,V projections + W |
| Attention (×8) | 8 × 4 × 512² = 8.4M | 4 heads per layer |
| Gating (×8) | 8 × 2 × 512² = 4.2M | Gate projections |
| FFN (×8) | 8 × 2 × 512 × 2048 = 16.8M | 4x expansion |
| LM Head | Tied with embedding | 0 (weight tying) |
| **Total** | **~50M** | |

### Hyperparameters

```yaml
# Model
d_model: 512
n_layers: 8
n_heads: 4
d_head: 128
d_ff: 2048
vocab_size: 32000
max_seq_len: 4096

# Memory
d_key: 512
d_value: 512
momentum_beta: 0.9
retention_local_init: 0.5
retention_global_init: 0.1

# Attention
window_size: 512

# Training (Stage 1)
chunk_size: 2048
batch_size: 8
learning_rate: 1e-4
warmup_steps: 1000
total_steps: 50000
```

---

## TNT Two-Stage Training

### Stage 1: Efficient Pre-training

**Hierarchical Memory:**
- Global memory: Sequential, large chunks (2048)
- Local memory: Parallel, periodic resets

```python
# Stage 1 pseudocode
for batch in dataloader:
    chunks = split(batch, chunk_size=2048)

    # Global memory update (sequential)
    global_mem = initial_state
    for chunk in chunks:
        global_mem = update_memory(global_mem, chunk)

    # Local memory update (parallel with resets)
    local_mems = parallel_process(chunks, reset_each=True)

    # Combine and compute loss
    output = combine(global_mem, local_mems)
    loss = cross_entropy(output, targets)
    loss.backward()
```

**Why this works:**
- Global captures long-range dependencies (sequential)
- Local captures fine-grained patterns (parallelizable)
- Resets break sequential dependency for parallelism

### Stage 2: Performance Fine-tuning

```python
# Stage 2: smaller chunks, no resets
chunk_size = 64  # or even token-by-token
# Fine-tune for ~5% additional compute
```

---

## Observability

### Metrics to Log (every N steps)

**Memory State:**
- `||W||_F` - Frobenius norm (memory magnitude)
- `rank(W)` - Effective rank via SVD
- `||W_t - W_{t-1}||_F` - Update magnitude
- Singular value spectrum of W

**Retention Dynamics:**
- `λ_local` values per layer
- `λ_global` values per layer
- Retention vs. update ratio

**Gating:**
- Mean gate value per layer (memory vs. attention balance)
- Gate variance (how much it varies across tokens)

**Training:**
- Loss, perplexity
- Gradient norms per component
- Learning rate schedule

---

## Implementation Plan

1. **memory.py**: MatrixMemory class with update/retrieve
2. **retention.py**: RetentionGate with local+global
3. **attention.py**: SlidingWindowAttention
4. **atlas.py**: Full model assembly
5. **trainer.py**: TNT two-stage training loop
6. **metrics.py**: Observability instrumentation

Each module: <200 lines, heavily commented, type-hinted.

---

## Success Criteria

**50M Model:**
- [ ] Training converges (loss decreases)
- [ ] Perplexity competitive with baseline transformer (~same params)
- [ ] Memory utilization observable (rank increases, then stabilizes)
- [ ] Retention gates learn meaningful values
- [ ] No NaN/Inf during training

**If successful:** Scale to 100M, then migrate to Todd_Atlas repo.

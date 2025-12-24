# Atlas Memory Tuning Guide

## Purpose

This document catalogs the parameters that affect **implicit memory learning** in Atlas. The goal is to help the model discover when and how to use its memory on its own, without forcing explicit retrieval patterns.

---

## Success: First Grokking Achievement (December 2024)

Both Shakespeare and Dumas models achieved grokking with 90%+ masked word accuracy.

**Results:**
| Model | Final Accuracy | Steps to Grok | Phase |
|-------|---------------|---------------|-------|
| Shakespeare | 90.5% | ~42,500 | grokked |
| Dumas | 90.8% | ~45,500 | grokked |

**Successful Configuration:**
```yaml
training:
  weight_decay: 0.0               # DISABLED - language needs rich representations

monitoring:
  stability:
    use_stablemax: false          # DISABLED - caused embedding collapse
    use_orthogonal_grad: false    # DISABLED - any strength caused collapse
    orthogonal_grad_strength: 0.0
```

**Key Observations:**
- Effective dimensionality stayed healthy at 60-65% throughout training
- Accuracy plateaued at ~10% for ~13k steps, then began climbing
- Grokking transition happened between steps 13k-20k
- Both models followed similar trajectories despite different corpora (English vs French)

**Lesson Learned:**
Grokking techniques from math papers (Doshi et al. 2025) don't directly apply to language tasks. Math has strict internal logic that benefits from StableMax and PerpGrad, but language is more ambiguous and requires richer embedding spaces.

---

## Failure Modes (Learned from Experiments)

### Failure Mode 1: Embedding Collapse (December 2024)

**Symptoms:**
- Effective dimensionality ratio crashed from ~78% to **0.39%** very early in training (by step 1500)
- Accuracy plateaued at ~10% and never improved
- Softmax Collapse (SC) metric at 100% throughout training
- Gradient-weight cosine stuck at 0 (PerpGrad fully active)

**Root Cause Analysis:**
After systematic testing, **StableMax was identified as the primary cause** of embedding collapse for language tasks.

**Testing sequence performed:**
1. PerpGrad at 1.0 → collapse
2. PerpGrad at 0.75 → collapse
3. PerpGrad at 0.50 → collapse
4. PerpGrad at 0.25 → collapse
5. PerpGrad at 0.15 → collapse
6. PerpGrad at 0.01 → collapse
7. PerpGrad off, StableMax on → **still collapse**
8. PerpGrad off, StableMax off → **SUCCESS** (effective dim stayed at 60-65%)

**Why StableMax causes collapse for language:**
- StableMax modifies the softmax function for numerical stability
- Works well for math tasks with discrete, unambiguous outputs
- For language, the modification appears to over-constrain the embedding space
- Language requires richer, higher-dimensional representations to capture ambiguity

**Key Difference from Grokking:**
| Metric | Grokking Pattern | Collapse Pattern (observed) |
|--------|------------------|----------------------------|
| Effective Dim | Gradual drop (80%→30%) | Immediate crash (78%→0.4%) |
| Timing | After accuracy rises | Before accuracy improves |
| Accuracy | Rises then stabilizes | Plateaus early (~10%) |
| Duration | Over 10k+ steps | Within first 1.5k steps |

**Configuration that caused this:**
```yaml
monitoring:
  stability:
    use_stablemax: true           # PRIMARY CAUSE for language tasks
    use_orthogonal_grad: true     # Also contributes to collapse
    orthogonal_grad_strength: 1.0
```

**Fix:** Disable both StableMax and PerpGrad for language training:
```yaml
monitoring:
  stability:
    use_stablemax: false
    use_orthogonal_grad: false
    orthogonal_grad_strength: 0.0
```

---

## Current Baseline Configuration (December 2024)

```yaml
# configs/atlas_shakespeare.yaml / atlas_dumas.yaml
# SUCCESSFUL - Both models achieved grokking (90%+ accuracy)

training:
  weight_decay: 0.0
  learning_rate: 3e-4

monitoring:
  stability:
    use_stablemax: false
    use_orthogonal_grad: false
```

---

## Parameter Categories

### 1. Gate Dynamics

Gates control how much memory influences the output. The model must learn to open gates when memory is useful.

| Parameter | Current Value | Effect on Memory Learning |
|-----------|---------------|---------------------------|
| `phase1_gate_floor` | 0.30 | High floor forces gates open early, ensures memory pathway is active |
| `phase2_gate_floor` | 0.10 | Medium floor allows more gate freedom |
| `phase3_gate_floor` | 0.05 | Low floor lets model decide when to use memory |
| `phase1_steps` | 3000 | Duration of forced high gates |
| `phase2_steps` | 4000 | Duration of medium gate phase |
| `storage_gate_target` | 0.80 | Forces high gates during storage (write to memory) |
| `retrieval_gate_floor` | 0.30 | Minimum gate during retrieval |

**Tuning Intuitions**:
- **If gates collapse (< 10%)**: Increase floors, model is bypassing memory
- **If accuracy plateaus with healthy gates**: Model isn't learning WHEN to use memory
- **Longer phase1**: More time with forced memory usage, may help model discover utility
- **Higher retrieval_gate_floor**: Force memory to always contribute during retrieval

**Experiments to Try**:
- [ ] Extend `phase1_steps` to 10000 (longer forced memory period)
- [ ] Increase `retrieval_gate_floor` to 0.50 (force more memory influence)
- [ ] Keep `phase3_gate_floor` at 0.05 but monitor if gates stay healthy

---

### 2. Memory Architecture

These control the capacity and structure of the memory system.

| Parameter | Current Value | Effect on Memory Learning |
|-----------|---------------|---------------------------|
| `d_key` | 256 | Dimension of memory keys (query matching) |
| `d_value` | 256 | Dimension of memory values (stored content) |
| `poly_degree` | 2 | Polynomial degree in memory (expressiveness) |
| `context_window` | 16 | Local context for memory operations |
| `window_size` | 128 | Sliding window for attention |
| `init_alpha` | 0.99 | Memory retention rate (how much to keep) |
| `init_theta` | 0.9 | Memory gate initialization |
| `init_eta` | 0.1 | Memory learning rate factor |

**Tuning Intuitions**:
- **`init_alpha` close to 1.0**: Memory retains more, slower forgetting
- **`init_alpha` lower (0.9)**: Memory updates faster, more responsive
- **`init_eta` higher**: Memory learns faster from each storage step
- **`poly_degree` higher**: More expressive memory, but harder to train

**Experiments to Try**:
- [ ] Increase `init_eta` to 0.2 or 0.3 (faster memory learning)
- [ ] Decrease `init_alpha` to 0.95 (more aggressive memory updates)
- [ ] Keep `poly_degree` at 2 (already reasonable)

---

### 3. Episodic Structure

How episodes are structured affects what the model can learn about memory.

| Parameter | Current Value | Effect on Memory Learning |
|-----------|---------------|---------------------------|
| `storage_samples` | 5 | Batches stored per episode |
| `retrieval_samples` | 5 | Retrieval attempts per episode |
| `reset_memory_between_episodes` | false | Memory persists across episodes |
| `num_masks` | 13 | Words masked per sequence (~10%) |

**Tuning Intuitions**:
- **More storage_samples**: More content stored before retrieval test
- **More retrieval_samples**: More practice retrieving from same stored content
- **reset_memory = true**: Each episode is independent (may help isolate learning)
- **reset_memory = false**: Memory accumulates (more realistic, but noisier signal)
- **Higher num_masks**: More retrieval practice per sequence, but harder task

**Experiments to Try**:
- [ ] Increase `storage_samples` to 10 (store more before testing)
- [ ] Try `reset_memory_between_episodes: true` (cleaner signal per episode)
- [ ] Reduce `num_masks` to 8 if accuracy struggles (easier task)

---

### 4. Loss Weights

These control how strongly the model is pushed to use memory for retrieval.

| Parameter | Current Value | Effect on Memory Learning |
|-----------|---------------|---------------------------|
| `retrieval_loss_weight` | 5.0 | Penalty multiplier for retrieval errors |
| `storage_loss_weight` | 1.0 | Standard LM loss during storage |

**Tuning Intuitions**:
- **Higher retrieval_loss_weight**: Stronger pressure to get retrieval right
- **Lower retrieval_loss_weight**: Less pressure, model may ignore memory
- **Ratio matters**: 5:1 means retrieval errors hurt 5x more than storage

**Experiments to Try**:
- [ ] Increase to 10.0 (very strong memory pressure)
- [ ] Decrease to 2.0 (see if model naturally uses memory without pressure)

---

### 5. Training Dynamics

General training parameters that affect learning, including grokking behavior.

| Parameter | Current Value | Effect on Memory Learning |
|-----------|---------------|---------------------------|
| `learning_rate` | 3e-4 | Overall learning speed |
| `weight_decay` | 0.0 | **DISABLED** - language needs rich representations |
| `warmup_steps` | 500 | LR warmup period |
| `grad_clip` | 1.0 | Gradient clipping |
| `max_steps` | 100000 | Maximum training steps |

**Tuning Intuitions**:
- **weight_decay = 0.0**: Works for language tasks - allows rich embedding representations
- **weight_decay > 0**: May help generalization, but test carefully (can cause collapse with other settings)
- **Lower learning_rate**: Slower but potentially more stable learning
- **Longer training**: Grokking often happens after apparent plateau (~13k steps in our experiments)

**Validated Configuration (December 2024)**:
- [x] `weight_decay: 0.0` - achieved 90%+ accuracy on both Shakespeare and Dumas
- [ ] Try small weight_decay (0.01) after grokking to improve generalization

---

### 6. Numerical Stability (StableMax / PerpGrad)

**WARNING**: These techniques from grokking literature cause embedding collapse for language tasks.

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| `use_stablemax` | **false** | DISABLED - causes embedding collapse for language |
| `use_orthogonal_grad` | **false** | DISABLED - any strength causes collapse |
| `orthogonal_grad_strength` | 0.0 | N/A when disabled |

**Findings (December 2024)**:

These techniques are designed for math/modular arithmetic tasks where:
- Outputs are discrete and unambiguous
- The solution space is highly constrained
- Compression leads to finding optimal circuits

For language tasks:
- **StableMax**: Modifies softmax in ways that crush embedding diversity
- **PerpGrad**: Projects out gradients needed for learning rich representations
- **Both**: Cause effective dimensionality to crash from 78% to <1% within 1500 steps

**Recommendation for Language Tasks**:
```yaml
monitoring:
  stability:
    use_stablemax: false
    use_orthogonal_grad: false
    orthogonal_grad_strength: 0.0
```

**For Math Tasks** (modular arithmetic, etc.): May still be beneficial - test empirically

---

## Decision Framework

After current run completes, evaluate:

### 1. Accuracy Trajectory
- **Climbing steadily**: Keep parameters, let it train longer
- **Plateaued at 10-12%**: Model not discovering memory utility
- **Plateaued at 20%+**: Model using memory but may have capacity limit

### 2. Gate Health
- **Gates healthy (30-60%)**: Memory pathway is available
- **Gates collapsed (<10%)**: Model bypassed memory, increase floors
- **Gates saturated (>90%)**: Model over-relying on memory

### 3. Loss Behavior
- **Decreasing steadily**: Learning is happening
- **Increasing**: Potential instability
- **Flat**: May need LR adjustment or longer training

---

## Recommended Configurations

### For Language Tasks (Shakespeare, Dumas, etc.)

**VALIDATED - Use this configuration:**
```yaml
training:
  weight_decay: 0.0
  learning_rate: 3e-4

monitoring:
  stability:
    use_stablemax: false
    use_orthogonal_grad: false
    orthogonal_grad_strength: 0.0
```

This achieved 90%+ masked word accuracy on both Shakespeare and Dumas corpora.

### If Accuracy Plateaus at ~10-12%

**First**: Check effective dimensionality ratio. If it's below 10%, you have embedding collapse - see Failure Mode 1.

**If effective dim is healthy (30%+)** and accuracy still plateaus:

#### Option A: Force More Memory Usage
```yaml
episodic:
  retrieval_gate_floor: 0.50  # Up from 0.30
  phase1_steps: 10000         # Up from 3000
  retrieval_loss_weight: 10.0 # Up from 5.0
```

#### Option B: Faster Memory Learning
```yaml
model:
  init_eta: 0.2               # Up from 0.1
  init_alpha: 0.95            # Down from 0.99

episodic:
  reset_memory_between_episodes: true  # Cleaner signal
```

### For Math Tasks (Modular Arithmetic)

Math tasks may benefit from grokking techniques - test empirically:
```yaml
training:
  weight_decay: 0.1           # Regularization helps math grokking

monitoring:
  stability:
    use_stablemax: true       # May help with numerical stability
    use_orthogonal_grad: true # May accelerate grokking
    orthogonal_grad_strength: 0.5  # Start conservative
```

**Note**: These settings caused collapse for language - only use for math/algorithmic tasks.

---

## Metrics to Track

When evaluating next run, focus on:

1. **masked_word_accuracy**: Primary metric (should climb above 15-20%)
2. **gate_mean**: Should stay healthy (20-60%)
3. **storage_loss vs retrieval_loss**: Retrieval should be harder but improving
4. **memory_magnitude per layer**: Should show activity, not flat
5. **grokking phase**: Watch for transition from "memorization" to "circuit_formation"

---

## Notes

- Implicit learning takes time. Grokking can happen after 50k+ steps.
- The model must discover that memory helps, not be forced to use it.
- Small adjustments are better than large ones.
- Let each run complete before making changes.

### Key Lessons (December 2024)

1. **Task type matters**: Grokking techniques from math papers don't directly apply to language
2. **Monitor effective dim ratio**: Below 10% early = collapse, not grokking
3. **StableMax is dangerous for language**: Caused immediate embedding collapse
4. **PerpGrad at any strength caused collapse** for language tasks
5. **Patience pays off**: Accuracy plateau at 10% for 13k steps before grokking began
6. **Cross-lingual consistency**: Both English (Shakespeare) and French (Dumas) showed similar trajectories

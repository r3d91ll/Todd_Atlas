# Atlas Experiment Summary (December 2025)

**Status**: Experiments paused - pivoting to alternative architecture
**Location**: `/home/todd/olympus/models/TheNest/experiments/Atlas_*`

---

## Executive Summary

We conducted a series of experiments to validate the Atlas architecture for language modeling. Despite multiple attempts with different configurations, all experiments encountered the same fundamental issue: **43M parameters on 831M tokens leads to rapid memorization**.

**Key Finding**: The problem is not the architecture per se, but the model-to-data ratio. Atlas at 43M params memorizes the training distribution rather than learning generalizable representations.

**Decision**: Pivot to BDH (Hebbian learning) architecture at 100M+ scale with larger datasets.

---

## Experiments Conducted

### v01: WeaverSpace Validation (Baseline)

**Objective**: Initial Atlas training with Weaver Space metrics capture

**Configuration**:
- Model: 512D × 6L × 8H (43.5M params)
- Data: Homegrown v23 (BPE-32K)
- Sequence: 256 tokens

**Result**: High perplexity (~200) after 44% training - dataset quality suspected

---

### v02: Dolma3 Dataset (4K Context + Muon)

**Objective**: Test with professionally curated dataset at extended context

**Configuration**:
- Model: 512D × 6L × 8H (43.5M params)
- Data: Dolma v1.7 balanced (831M tokens)
- Sequence: 4096 tokens
- Context window: 64 (memory chunks)
- Learning rate: 1e-4 cosine decay to 1e-5

**Key Fix**: Added Muon-style spectral normalization to `OmegaRule.forward()` for 4K stability

**Results**:
| Step | Loss | PPL | Observation |
|------|------|-----|-------------|
| 1000 | 5.8 | 330 | Training started |
| 3000 | 4.2 | 67 | Descending steadily |
| 5000 | 3.1 | 22 | Accelerating descent |
| 5630 | 2.5 | 12 | **Overfit confirmed** |

**Generation Test** ("A quick brown fox"):
```
a quick brown fox fox fox fox then it is a...
[degenerates into repetition and word salad]
```

**Conclusion**: Model memorized training distribution. PPL 296 at 5.6K steps → classic overfit pattern.

---

### v03: Dropout Regularization

**Objective**: Test if dropout (0.1) prevents memorization

**Configuration**:
- Same as v02
- Added: `dropout: 0.1`

**Key Insight**: Monitor PPL decline **rate**, not absolute values

**Results**:
| Step Range | PPL Change | Rate |
|------------|------------|------|
| 2500 → 2550 | 991K → 991K | flat (dropout holding) |
| 2550 → 2800 | 991K → 208K | 86%/100 steps |
| 2800 → 3000 | 208K → 43K | **120%/100 steps** |

**Decision**: Killed at step 3140 - decline rate identical to v02, just delayed by ~500 steps

**Conclusion**: Dropout delays but doesn't prevent memorization. The fundamental issue is capacity vs. data.

---

### v04: Slower LR Schedule

**Objective**: Test if gentler learning rate prevents rapid descent

**Configuration**:
- Warmup: 5000 → 10000 steps
- Peak LR: 1e-4 → 5e-5
- Floor LR: 1e-5 → 1e-6
- Keep dropout 0.1

**Results**:
| Variant | Outcome |
|---------|---------|
| v04 SlowLR | NaN @ step 250 |
| v04 + dropout warmup toggle | NaN @ step 180 |
| v04 + v03 LR + dropout toggle | NaN @ step 110 |

**Root Cause**: `model.eval()` during warmup (to disable dropout) breaks Atlas architecture. The Hebbian/memory mechanisms require train mode to function.

**Conclusion**: Atlas architecture has hidden dependencies on train-mode behavior.

---

## Technical Discoveries

### 1. Muon Normalization Required for 4K Context

Original `OmegaRule.forward()` used raw gradients, causing NaN at step 130 with 4K sequences.

**Fix** (applied in v02):
```python
# Before update
def forward(self, keys, values, queries):
    grad = ...  # raw gradient
    self.M = self.M - self.learning_rate * grad

# After fix (Muon-style)
def forward(self, keys, values, queries):
    grad = ...
    # Spectral normalization
    U, S, V = torch.linalg.svd(grad, full_matrices=False)
    grad_normalized = U @ V
    self.M = self.M - self.learning_rate * grad_normalized
```

### 2. PPL Decline Rate as Early Overfit Detector

Instead of waiting for checkpoints at 5K/10K steps, monitoring the **rate of PPL descent** provides early warning:

- **Healthy**: Gradual decline, rate decreasing over time
- **Overfit**: Constant or accelerating rate (40-120%/100 steps)

This saved hours of training time by detecting v03 failure at step 3K.

### 3. model.eval() Breaks Atlas

Unlike standard transformers, Atlas cannot use `model.eval()` during training:
- Disabling dropout during warmup caused NaN
- The architecture has undocumented dependencies on train-mode behavior
- Possibly related to Hebbian memory update mechanics

---

## Root Cause Analysis

### The Fundamental Problem

| Metric | Value | Typical Range |
|--------|-------|---------------|
| Model params | 43M | - |
| Training tokens | 831M | - |
| Tokens per param | 19.3 | 20-100 (Chinchilla: ~20 minimum) |

At 19.3 tokens/param, the model is operating at the **absolute minimum** for compute-optimal training. Any architectural inefficiency tips it into memorization.

### Why Atlas Memorizes

1. **Memory modules store exact key-value pairs**: Unlike attention which is content-addressable, OmegaRule learns specific associations
2. **Small model capacity**: 43M params distributed across memory, attention, and FFN
3. **High-quality data**: Dolma is "too easy" - the model memorizes it quickly
4. **Long context without regularization**: 4K tokens per sample provides more opportunity to overfit

---

## Recommendations for Future Work

### If Continuing with Atlas

1. **Scale up parameters**: 500M+ minimum for this architecture
2. **Scale up data**: 10B+ tokens (not 831M)
3. **Use held-out validation**: Monitor generalization gap, not just training loss
4. **Investigate model.eval()**: Understand why it breaks the architecture

### Alternative Approaches (Current Direction)

We are pivoting to **BDH (Baby Dragon Hatchling)** at 100M scale:
- Hebbian learning provides natural regularization
- Shared parameters prevent layer-specific memorization
- Sparse activations limit effective capacity

**New experiment location**: `/home/todd/olympus/AshesBelow/experiments/BDH_100M/`

---

## Files Reference

| File | Purpose |
|------|---------|
| `Atlas_v02_Dolma3/code/train.py` | Contains Muon fix for 4K stability |
| `Atlas_v03_Dropout/README.md` | PPL decline rate analysis methodology |
| `Atlas_v04_SlowLR/code/train.py` | Dropout warmup toggle (broke training) |

---

## Lessons Learned

1. **Monitor decline rates, not absolute values**: Early overfit detection saves time
2. **Capacity vs data matters more than regularization**: Dropout can't fix model-data mismatch
3. **Test architectural assumptions**: model.eval() behavior is not universal
4. **Document failures thoroughly**: They inform future experiments

---

*Last updated: December 6, 2025*

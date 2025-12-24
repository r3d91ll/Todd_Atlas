# Atlas Phase Detection Framework - Implementation Plan

## Overview

This document outlines the complete set of changes for the Atlas experimental framework update, incorporating tokenizer changes, model scaling, phase detection, and cross-lingual dataset preparation.

---

## 1. Tokenizer Change: T5 → Pruned mT5

### Rationale
- T5 has English bias (Spanish requires 15-30% more tokens for equivalent content)
- This contaminates cross-lingual comparison
- mT5 designed for fair multilingual treatment

### Implementation

**New file:** `src/tokenizer/prune_mt5.py`

```python
# Prune mT5 tokenizer to only tokens used in our corpora
# Input: Shakespeare + de Vega texts
# Output: Pruned tokenizer with ~35-45K vocab
```

**Steps:**
1. Load full mT5 tokenizer (250K vocab)
2. Tokenize entire Shakespeare corpus
3. Tokenize entire de Vega corpus
4. Count token frequencies across both
5. Keep: all special tokens + any token appearing ≥1 time
6. Save pruned tokenizer to `tokenizer/atlas_multilingual/`

**Expected outcome:**
- Vocab size: ~35-45K tokens
- Fair treatment of English and Spanish
- Embedding params: ~4.5-5.8M (with d_model=128)

**Config changes:**
```yaml
model:
  vocab_size: 40000  # Updated after pruning (use actual number)

data:
  tokenizer_path: "tokenizer/atlas_multilingual"
```

---

## 2. Model Architecture: 6-7M Parameters

### Final Architecture

```yaml
model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 512
  vocab_size: ~40000  # From pruned mT5
  max_seq_len: 512

  # Memory (scaled to d_model)
  d_key: 64
  d_value: 64
  poly_degree: 2
  init_alpha: 0.99
  init_theta: 0.9
  init_eta: 0.1
```

### Parameter Breakdown (estimated)

| Component | Params |
|-----------|--------|
| Embeddings (40K × 128) | 5.1M |
| 4× Transformer layers | ~1.0M |
| Memory matrices | ~0.5M |
| **Total** | **~6.6M** |

### Rationale for accepting 6-7M
- Still much faster than 10M+ models
- Sparsity in vocab may help monosemantic feature discovery
- Clean geometric signatures for Conveyance Hypothesis

---

## 3. Dataset Preparation

### 3.1 Shakespeare (English) - EXISTS
- Location: `data/shakespeare/`
- Format: Sharded compressed JSONL
- Content: Complete works

### 3.2 de Vega (Spanish) - NEW

**New file:** `data/prepare_devega_dataset.py`

**Sources (Priority Order):**
1. Project Gutenberg:
   - Fuenteovejuna (#60198)
   - La Moza de Cántaro (#23206)
   - Comedias inéditas (#57035)

2. Government/Educational PDFs:
   - El Caballero de Olmedo (seducoahuila.gob.mx)
   - Fuenteovejuna RAE edition

**Output:** `data/devega/complete_works_shard{XX}.jsonl.zst`

**Token balancing:**
- Match by character count, not token count
- Add poetry if needed (de Vega's Rimas or Shakespeare's Sonnets)

### 3.3 Tokenizer Pruning Script

**New file:** `data/prepare_tokenizer.py`

```python
def prune_mt5_tokenizer(
    shakespeare_dir: Path,
    devega_dir: Path,
    output_dir: Path,
    min_frequency: int = 1
) -> int:
    """
    Prune mT5 to tokens used in both corpora.
    Returns final vocab size.
    """
```

---

## 4. Phase Detection Framework

### 4.1 Phase Definitions

| Phase | Name | Detection Criteria |
|-------|------|-------------------|
| 1 | Memory Learning | Accuracy trending up OR loss decreasing |
| 2 | Converged | Accuracy plateau (low variance) at any level |
| 3 | Overfitting | Train acc ↑, Val acc ↓ (gap growing) |
| 4 | Grokking | Val acc recovers, eff_dim stabilizes |

### 4.2 Implementation

**New file:** `training_framework/phase_detector.py`

```python
@dataclass
class PhaseDetectorConfig:
    convergence_window: int = 2000      # Steps for plateau detection
    stability_window: int = 1000        # Steps for stability check
    variance_threshold: float = 0.001   # Plateau = low variance
    eff_dim_healthy_min: float = 0.30   # Below this = collapse

class PhaseDetector:
    def __init__(self, config: PhaseDetectorConfig):
        self.config = config
        self.history = []

    def detect_phase(self, metrics: Dict) -> str:
        """Returns: 'memory_learning', 'converged', 'overfitting', 'grokking'"""

    def is_plateau(self, values: List[float]) -> bool:
        """Check if values have low variance (plateaued)."""

    def detect_overfitting(self, train_acc: List, val_acc: List) -> bool:
        """Check if train/val gap is growing."""
```

### 4.3 Metrics Used (Language Tasks)

**USE:**
- `effective_dim_ratio` - embedding health
- `masked_word_accuracy` - task performance
- `grad_weight_cosine` - gradient health
- `gate_mean` - memory usage

**DO NOT USE:**
- ~~`fourier_concentration`~~ - math-specific
- ~~`circular_fit`~~ - math-specific

---

## 5. Validation Split

### Implementation

**Modify:** `src/training/episodic_trainer.py`

```python
def create_validation_split(dataset, validation_ratio=0.15):
    """
    Create held-out validation set.
    CRITICAL: Different PASSAGES, not just different masks.
    """
    total_size = len(dataset)
    val_size = int(total_size * validation_ratio)
    train_size = total_size - val_size
    return random_split(dataset, [train_size, val_size])
```

**Add to training loop:**
```python
# Compute validation metrics every N steps
if step % val_interval == 0:
    val_metrics = evaluate_on_validation(model, val_loader)
    log_validation_metrics(val_metrics)
```

---

## 6. Automatic Checkpointing

### Trigger Points

| Event | Checkpoint Name |
|-------|-----------------|
| Phase 1 → 2 (convergence) | `converged_step_{step}_acc_{acc}.pt` |
| Phase 3 detected (overfitting) | `overfitting_detected_step_{step}.pt` |
| Phase 4 (grokking) | `grokked_step_{step}_acc_{acc}_effdim_{dim}.pt` |

**Modify:** `src/training/episodic_trainer.py`

```python
def handle_phase_transition(self, new_phase: str, old_phase: str):
    if new_phase != old_phase:
        self.save_phase_checkpoint(new_phase)
        self.log_phase_transition(old_phase, new_phase)
```

---

## 7. Dashboard Updates

### Phase Indicator Widget

**Modify:** `training_framework/monitoring/streamlit_monitor.py`

```python
def render_phase_indicator(phase: str):
    colors = {
        "memory_learning": "#3498db",  # Blue
        "converged": "#2ecc71",         # Green
        "overfitting": "#f39c12",       # Orange
        "grokking": "#9b59b6",          # Purple
    }
    st.markdown(f"""
        <div style="background:{colors[phase]}; padding:20px; text-align:center;">
            <h2>Phase: {phase.upper()}</h2>
        </div>
    """, unsafe_allow_html=True)
```

### Validation Metrics Panel

Add new panel showing:
- Train vs Val accuracy comparison
- Overfitting gap trend
- Phase transition history

---

## 8. Configuration Updates

### 8.1 Shakespeare Config

**Modify:** `configs/atlas_shakespeare.yaml`

```yaml
model:
  d_model: 128
  n_layers: 4
  n_heads: 4
  d_ff: 512
  vocab_size: 40000  # Updated for pruned mT5

data:
  tokenizer_path: "tokenizer/atlas_multilingual"
  validation_ratio: 0.15

monitoring:
  stability:
    use_stablemax: false
    use_orthogonal_grad: true
    orthogonal_grad_strength: 0.5  # Current best

  phase_detection:
    enabled: true
    convergence_window: 2000
    variance_threshold: 0.001
```

### 8.2 de Vega Config

**Create:** `configs/atlas_devega.yaml`
(Already created - update vocab_size after tokenizer pruning)

---

## 9. File Summary

### New Files
```
src/tokenizer/prune_mt5.py           # Tokenizer pruning logic
data/prepare_devega_dataset.py       # de Vega dataset download/prep
data/prepare_tokenizer.py            # Combined tokenizer preparation
training_framework/phase_detector.py # Phase detection logic
tokenizer/atlas_multilingual/        # Pruned tokenizer output
data/devega/                          # de Vega dataset
```

### Modified Files
```
configs/atlas_shakespeare.yaml       # Scale down, new tokenizer
configs/atlas_devega.yaml            # Already created, update vocab
src/training/episodic_trainer.py     # Validation split, phase hooks
src/data/loader.py                   # Use new tokenizer path
training_framework/monitoring/streamlit_monitor.py  # Phase indicators
docs/MEMORY_TUNING_GUIDE.md          # Update with findings
```

---

## 10. Implementation Order

### Phase A: Tokenizer & Data (Do First)
1. [ ] Write `prepare_tokenizer.py` - prune mT5 on both corpora
2. [ ] Run pruning, get actual vocab size
3. [ ] Update `prepare_devega_dataset.py` with correct tokenizer
4. [ ] Download and prepare de Vega dataset
5. [ ] Verify token counts are balanced

### Phase B: Model & Config
6. [ ] Update `atlas_shakespeare.yaml` with new vocab_size
7. [ ] Update `atlas_devega.yaml` with new vocab_size
8. [ ] Verify parameter count (~6-7M)

### Phase C: Training Infrastructure
9. [ ] Implement `PhaseDetector` class
10. [ ] Add validation split to episodic trainer
11. [ ] Add phase transition checkpointing
12. [ ] Update dashboard with phase indicators

### Phase D: Validation
13. [ ] Run short test on Shakespeare to verify pipeline
14. [ ] Run short test on de Vega to verify Spanish handling
15. [ ] Verify phase detection triggers correctly
16. [ ] Document findings in Memory Tuning Guide

---

## 11. Success Criteria

### Tokenizer
- [ ] Pruned vocab covers 100% of both corpora (no UNK tokens)
- [ ] Special tokens preserved (PAD, EOS, MASK, etc.)
- [ ] Vocab size in 35-45K range

### Model
- [ ] Total params: 6-7M
- [ ] Loads and runs without errors
- [ ] Memory fits on single A6000

### Phase Detection
- [ ] Correctly identifies plateau (convergence)
- [ ] Overfitting detected when val acc drops
- [ ] Checkpoints saved at phase transitions

### Cross-lingual
- [ ] Shakespeare and de Vega show similar training dynamics
- [ ] Token efficiency comparable between languages
- [ ] Grokking patterns (if achieved) occur at similar step counts

---

## 12. Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Pruned vocab too small | Keep min_frequency=1, add all special tokens |
| Pruned vocab too large | Accept it - sparsity may help monosemantic features |
| de Vega sources unavailable | Multiple backup sources documented |
| Phase detection false positives | Conservative thresholds, require sustained signals |
| Spanish tokenization issues | Verify Spanish text before full training |

---

## Notes

- All changes on branch: `feature/phase-detection-framework`
- Current PerpGrad 0.5 test still running on master - don't interrupt
- Update Memory Tuning Guide after each major finding

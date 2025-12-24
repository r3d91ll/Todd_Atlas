# Baseline Training Run Analysis
# Date: 2025-12-24 15:41
# Purpose: Document English-optimized tokenizer runs for comparison with balanced tokenizer

## Experimental Context

### Tokenizer Configuration
- **Type**: English-optimized tokenizer (NOT balanced for cross-lingual comparison)
- **Known Issue**: Tokenizer was trained primarily on English data
- **Expected Impact**: English (Shakespeare) should have advantage over French (Dumas)
- **Next Run**: Will use pruned mT5 tokenizer trained on Shakespeare (EN) + de Vega (ES)

### Models Compared
| Model | Language | Corpus | GPU |
|-------|----------|--------|-----|
| Shakespeare | English | Shakespeare theatrical works | cuda:0 |
| Dumas | French | Alexandre Dumas novels | cuda:1 |

### Architecture (Identical for both)
- d_model: 256
- n_layers: 4
- n_heads: 4
- vocab_size: 32000 (English-optimized)
- batch_size: 32
- learning_rate: 3e-4

---

## Results at 25k Steps (25% of training)

### Primary Metrics
| Metric | Shakespeare (EN) | Dumas (FR) | Delta | Notes |
|--------|------------------|------------|-------|-------|
| Masked Word Accuracy | 85.46% | 68.70% | +16.76pp | EN advantage likely from tokenizer |
| Retrieval Loss | 2.889 | 6.124 | -3.235 | Lower is better |
| Storage Loss | 0.0581 | 0.1590 | - | |

### Memory Behavior
| Metric | Shakespeare | Dumas | Interpretation |
|--------|-------------|-------|----------------|
| Memory Rank | 9.73 | 22.87 | EN more compressed - tokenizer efficiency? |
| Memory Magnitude | 95.73 | 135.13 | Similar utilization |
| Memory Sparsity | 0.0028 | 0.0147 | |
| Memory Entropy | 2.429 | 2.377 | |

### Gate Health (Indicators of training stability)
| Metric | Shakespeare | Dumas | Target |
|--------|-------------|-------|--------|
| Gate Mean | 0.547 | 0.562 | 0.3-0.7 |
| Gate Std | 0.275 | 0.281 | >0.1 |
| Dead Ratio | 0.00% | 0.00% | <5% |
| Saturated Ratio | 5.82% | 7.02% | <10% |

### Layer-wise Memory Magnitude
| Layer | Shakespeare | Dumas |
|-------|-------------|-------|
| 0 | 83.9 | 114.5 |
| 1 | 82.8 | 135.3 |
| 2 | 107.4 | 136.9 |
| 3 | 108.8 | 153.8 |

### Training Dynamics
| Phase | Shakespeare Avg Acc | Dumas Avg Acc | Gap |
|-------|---------------------|---------------|-----|
| Phase 1 (0-3k) | 9.01% | 10.75% | -1.74pp |
| Phase 2 (3k-7k) | 9.71% | 11.98% | -2.27pp |
| Phase 3 (7k+) | 41.25% | 29.49% | +11.76pp |

**Note**: Dumas actually started slightly ahead in early phases, but Shakespeare
pulled ahead significantly in Phase 3. This suggests the tokenizer advantage
compounds over time as the model relies more on learned representations.

---

## Grokking Analysis

**IMPORTANT**: True grokking requires convergence of ALL metrics toward 
generalization, not just accuracy improvements. A jump in masked word 
accuracy alone could indicate better memorization, not generalization.

### Metrics to track for true grokking:
1. **Retrieval accuracy** - must improve
2. **Loss** - must decrease AND stabilize
3. **Memory rank** - should decrease (compression = generalization)
4. **Gate variance** - should remain healthy (not collapse)
5. **Generalization gap** - train vs held-out performance

### Current observations:
- Shakespeare showed accuracy jump at 20k (35.6% → 46.0%)
- But memory rank was already low (~10) - may be memorization
- Need held-out validation to distinguish memorization from generalization

---

## Key Questions for Next Run (Balanced Tokenizer)

1. **Will memory rank converge?**
   - Current: Shakespeare ~10, Dumas ~22
   - With balanced tokenizer, should these be more similar?

2. **Will accuracy gap close?**
   - Current: +22pp advantage for English
   - Balanced tokenizer should reduce this

3. **Tokenizer efficiency metrics to compare:**
   - Tokens per character for each language
   - UNK rate (should be 0% for both with pruned mT5)
   - Subword distribution

4. **True grokking indicators:**
   - Monitor ALL metrics, not just accuracy
   - Add held-out validation set
   - Track train/val gap over time

---

## Files for Reference
- Shakespeare metrics: runs/atlas_shakespeare/metrics_stream.jsonl
- Dumas metrics: runs/atlas_dumas/metrics_stream.jsonl
- Steps completed: Shakespeare 30265, Dumas 28945
- Training time: ~4.4 hours each

---

## Appendix: Tokenizer Details

### Old Tokenizer (English-optimized, vocab=32000)
- **Source**: Standard tokenizer, English-focused
- **Known bias**: French text required more tokens per character
- **Impact**: Dumas (French) at disadvantage from the start

### New Tokenizer (Balanced mT5, vocab=29153)
- **Source**: Pruned from mT5 (250k → 29k tokens)
- **Training data**: Shakespeare (EN) + de Vega (ES) corpora
- **UNK rate**: 0% for both languages (verified)
- **Expected impact**: Fair comparison between languages

### Hypothesis for Next Run
With the balanced tokenizer:
1. Memory rank gap should decrease (currently EN=10, FR=22)
2. Accuracy gap should decrease (currently +17pp for EN)
3. Training dynamics should be more similar across languages
4. If gaps persist, they reflect actual linguistic/structural differences


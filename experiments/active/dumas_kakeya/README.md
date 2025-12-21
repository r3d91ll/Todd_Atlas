# Dumas Kakeya Experiment

**Status:** Active
**Started:** December 2024
**Corpus:** Alexandre Dumas Works (French)

## Hypothesis

Train a small Atlas model (~10M parameters) on Dumas's works to:
1. Validate episodic training on smaller scale
2. Extract Kakeya set signatures for French literary language
3. Compare geometric signatures across languages

## Design

### Model Configuration
- d_model: 256
- n_layers: 4
- n_heads: 4
- Parameters: ~10M

### Training
- Episodic training with storage/retrieval phases
- Gate floors enabled (prevents gate collapse)
- Grokking detection active

### Kakeya Analysis
Concepts to track:
- Characters (D'Artagnan, Athos, Monte Cristo)
- Themes (honneur, vengeance, amitie, amour)
- Literary devices (French narrative style)

## Controlled Comparison

This experiment runs with **identical training regime** as shakespeare_kakeya.
Only difference is the corpus language/author.

This enables controlled comparison of Kakeya geometric signatures
across languages while eliminating training-related confounds.

## Research Questions

1. Do Kakeya signatures differ between languages?
2. Are concept geometries language-specific or universal?
3. How do French vs English concepts cluster in embedding space?

## Usage

```bash
# Download corpus
python scripts/download_corpus.py

# Run training
./scripts/train_dumas.sh
```

## Expected Outputs

- Checkpoints: `runs/atlas_dumas/checkpoints/`
- Metrics: `runs/atlas_dumas/metrics_stream.jsonl`
- Grokking detection: (tracked within metrics stream)

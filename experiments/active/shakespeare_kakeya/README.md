# Shakespeare Kakeya Experiment

**Status:** Active
**Started:** December 2024
**Corpus:** Shakespeare Complete Works (English)

## Hypothesis

Train a small Atlas model (~10M parameters) on Shakespeare's works to:
1. Validate episodic training on smaller scale
2. Extract Kakeya set signatures for English literary language
3. Study grokking transition in constrained domain

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
- Characters (Hamlet, Macbeth, Romeo, etc.)
- Themes (love, death, power, jealousy)
- Literary devices (metaphor, soliloquy)

## Controlled Comparison

This experiment runs with **identical training regime** as dumas_kakeya.
Only difference is the corpus language/author.

This enables controlled comparison of Kakeya geometric signatures
across languages while eliminating training-related confounds.

## Usage

```bash
# Download corpus
python scripts/download_corpus.py

# Run training
./scripts/train_shakespeare.sh
```

## Expected Outputs

- Checkpoints: `runs/atlas_shakespeare/checkpoints/`
- Metrics: `runs/atlas_shakespeare/metrics_stream.jsonl`
- Grokking detection: (tracked within metrics stream)

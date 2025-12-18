# CLAUDE.md - Atlas Project

## Project Overview

**Atlas** is a minimal, observable implementation of the Behrouz et al. memory architecture framework (Miras). The goal is to prove the framework at 50M parameters before scaling to 100M+.

**Status**: Implementation complete, awaiting GPU availability for training.

## Theoretical Foundation

Based on five papers from Behrouz, Zhong, Mirrokni et al. (Google Research):

| Paper | Key Contribution |
|-------|------------------|
| **Titans** | Test-time memorization via gradient-based updates |
| **It's All Connected** | Miras framework - 4 design axes |
| **ATLAS** | Optimal memory allocation at inference |
| **TNT** | Hierarchical training, 17× speedup |
| **Nested Learning** | Architecture = optimization at different levels |

## Architecture (Miras Framework)

Strict adherence to the 4-axis design:

| Axis | Choice | Implementation |
|------|--------|----------------|
| **Memory** | Matrix-valued W ∈ R^{512×512} | `src/model/memory.py` |
| **Attentional Bias** | L2 (baseline) | Squared error objective |
| **Retention Gate** | Local + Global | `src/model/retention.py` |
| **Learning Algorithm** | Momentum (β=0.9) | With Q-K projection from TNT |

## Key Commands

```bash
# Activate environment
cd /home/todd/olympus/models/Atlas
source venv/bin/activate

# Run tests (validates all components)
python scripts/test_model.py

# Train with synthetic data (for testing)
python scripts/train.py --config configs/atlas_50m.yaml --test --device cuda:1

# Train with real data
python scripts/train.py --config configs/atlas_50m.yaml --device cuda:1

# Resume from checkpoint
python scripts/train.py --config configs/atlas_50m.yaml --resume runs/atlas_50m/checkpoints/checkpoint_5000.pt
```

## Directory Structure

```
Atlas/
├── configs/atlas_50m.yaml     # Full training configuration
├── docs/ARCHITECTURE.md       # Detailed design rationale
├── src/
│   ├── model/
│   │   ├── memory.py          # MatrixMemory, ChunkedMatrixMemory
│   │   ├── retention.py       # RetentionGate, AdaptiveRetentionGate
│   │   ├── attention.py       # SlidingWindowAttention, Gating, FFN
│   │   └── atlas.py           # Atlas model, AtlasConfig, AtlasBlock
│   ├── training/
│   │   ├── trainer.py         # AtlasTrainer, TNTTrainer
│   │   └── metrics.py         # MetricsLogger, MemoryProfiler
│   └── data/
│       └── loader.py          # DolminoDataset, create_dataloaders
├── scripts/
│   ├── train.py               # Main entry point
│   └── test_model.py          # Component tests
└── runs/                      # Experiment outputs (metrics, checkpoints)
```

## Model Configuration (50M)

```yaml
d_model: 512
n_layers: 8
n_heads: 4
d_ff: 2048
vocab_size: 32000
max_seq_len: 4096
window_size: 512
```

**Actual parameter count**: 56.3M

## Training Strategy (TNT Two-Stage)

| Stage | Chunk Size | Steps | Purpose |
|-------|------------|-------|---------|
| **Stage 1** | 2048 | 45,000 | Efficient pre-training with memory resets |
| **Stage 2** | 256 | 5,000 | Fine-grained accuracy tuning |

## Observability Metrics

Logged to `runs/atlas_50m/metrics/`:

- **Memory**: `||W||_F`, effective rank, update magnitude, singular values
- **Retention**: λ_local, λ_global per layer
- **Gating**: Mean gate value (memory vs attention balance)
- **Training**: Loss, perplexity, gradient norms

## Data

Training data: `../datasets/raw/dolma3/dolmino_mix_100B/data/`

Default categories (subset for faster iteration):
- `ingredient1-common_crawl-high-quality_19_science_math_and_technology`
- `ingredient1-common_crawl-high-quality_19_literature`
- `ingredient1-code-meta-reasoning`

## Hardware

- **Target GPU**: cuda:1 (NVIDIA RTX A6000, 49GB)
- **Mixed precision**: BF16 enabled
- **Effective batch size**: 32 (8 × 4 gradient accumulation)

## Success Criteria

- [ ] Training converges (loss decreases monotonically after warmup)
- [ ] Perplexity competitive with baseline transformer (~same params)
- [ ] Memory rank increases then stabilizes
- [ ] Retention gates learn meaningful values (not all 0 or 1)
- [ ] No NaN/Inf during training

## Next Steps

1. Train 50M model, validate framework
2. If successful, scale to 100M (`create_atlas_100m()` in atlas.py)
3. Migrate to `/home/todd/olympus/git-repos/Todd_Atlas/` for production

## Related Resources

- Previous attempt: `/home/todd/olympus/git-repos/Todd_Atlas/` (paused due to scale)
- Papers: See README.md for arXiv links
- Parent project: `/home/todd/olympus/CLAUDE.md`

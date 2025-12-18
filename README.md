# Atlas: Miras Framework Implementation

A minimal, observable implementation of the Behrouz et al. memory architecture framework (Titans/Miras).

## Current Status

| Model | Parameters | Status | Report |
|-------|------------|--------|--------|
| 300M Omega | 389.5M | **Completed** | [Training Report](runs/atlas_300m_omega/TRAINING_REPORT.md) |
| 1B Omega | ~1B | Planning | [Build Plan](docs/1B_BUILD_PLAN.md) |

### 300M Results Summary

- **Final PPL:** 229
- **Training Time:** 38.9 hours
- **Key Finding:** Gate collapse phenomenon - model learned to bypass memory (99.3% attention, 0.7% memory)
- **Conclusion:** Architecture validated, but 389M insufficient for coherent generation. Scaling to 1B.

See the full [300M Training Report](runs/atlas_300m_omega/TRAINING_REPORT.md) for detailed analysis.

---

## Framework Reference

Based on five papers from Behrouz, Zhong, Mirrokni et al. (Google Research):

| Paper | Key Contribution |
|-------|------------------|
| **Titans** | Test-time memorization via gradient-based updates |
| **It's All Connected** | Miras framework - 4 design axes |
| **ATLAS** | Optimal memory allocation at inference |
| **TNT** | Hierarchical training, 17x speedup |
| **Nested Learning** | Architecture = optimization at different levels |

## Architecture (Miras Framework)

Strict adherence to the 4-axis design:

| Axis | Choice | Implementation |
|------|--------|----------------|
| **Memory** | Matrix-valued M | `src/model/memory.py` |
| **Attentional Bias** | L2 (baseline) | Squared error objective |
| **Retention Gate** | Local + Global | `src/model/retention.py` |
| **Learning Algorithm** | Momentum (beta=0.9) | With Q-K projection from TNT |

### Memory Module

Each layer maintains:
- **M**: Memory matrix [batch, d_key, d_value] - associative storage
- **S**: Surprise accumulator [batch, d_key, d_value] - gradient momentum
- **Gates**: Input-dependent alpha, eta, theta for retention control

## Training Strategy (TNT Two-Stage)

| Stage | Chunk Size | Steps | Purpose |
|-------|------------|-------|---------|
| **Stage 1** | 2048 | 100,000 | Bulk pre-training with memory resets |
| **Stage 2** | 256 | 10,000 | Fine-grained accuracy tuning |

## Directory Structure

```
Atlas/
├── README.md
├── CLAUDE.md                    # Development context
├── configs/
│   ├── atlas_50m.yaml           # Original 50M config
│   └── atlas_300m_omega.yaml    # 300M Omega config
├── docs/
│   ├── ARCHITECTURE.md          # Design document
│   └── 1B_BUILD_PLAN.md         # 1B scaling plan
├── src/
│   ├── model/
│   │   ├── atlas_omega.py       # AtlasOmega model
│   │   ├── memory.py            # TitansMemory module
│   │   ├── attention.py         # Sliding window attention
│   │   └── retention.py         # Retention gates
│   ├── data/
│   │   └── loader.py            # Dolmino dataset loader
│   └── training/
│       ├── trainer.py           # TNT trainer
│       └── metrics.py           # Observability metrics
├── scripts/
│   ├── train_ddp_omega.py       # Main training script
│   └── test_inference.py        # Inference testing
├── runs/
│   └── atlas_300m_omega/        # 300M run artifacts
│       ├── TRAINING_REPORT.md   # Full analysis
│       ├── checkpoints/         # Model checkpoints
│       └── metrics/             # Training metrics (JSONL)
└── tokenizer/
    └── atlas_tokenizer/         # LLaMA tokenizer
```

## Quick Start

```bash
# Setup
cd /home/todd/olympus/models/Atlas
source venv/bin/activate

# Run inference on trained model
python scripts/test_inference.py \
  --checkpoint runs/atlas_300m_omega/checkpoints/checkpoint_110000.pt \
  --prompt "Once upon a time" \
  --device cuda:0

# Train new model
CUDA_VISIBLE_DEVICES=1 python scripts/train_ddp_omega.py \
  --config configs/atlas_300m_omega.yaml \
  --output-dir runs/my_run
```

## Key Findings from 300M Run

### Gate Collapse

The most significant observation: gates progressively collapsed during training.

| Step | Avg Gate | Memory Usage |
|------|----------|--------------|
| 639 | 50.1% | Balanced |
| 33K | 11.7% | Attention favored |
| 76K | 0.7% | Memory bypassed |
| 109K | 0.7% | Memory bypassed |

**Implication:** Model optimized for attention, ignored memory. Needs regularization.

### Observability Metrics

Per-layer metrics logged every step:
- `M_norm`, `M_std`, `M_max` - Memory matrix state
- `S_norm` - Surprise accumulator activity
- `gate_mean`, `gate_std` - Attention/memory balance

## Next Steps

1. **1B Build** - Scale up with gate regularization
2. **Cloud Deployment** - H100/H200 on RunPod (~$60-100)
3. **TCF Metrics** - Add geometric analysis (D_eff, beta, curvature)
4. **Telegram Alerts** - Replace SMS notifications

See [1B Build Plan](docs/1B_BUILD_PLAN.md) for details.

## Hardware

| GPU | VRAM | Use Case |
|-----|------|----------|
| RTX A6000 (local) | 48GB | 300M training, inference |
| H100 (cloud) | 80GB | 1B training |
| H200 (cloud) | 141GB | 1B training (faster) |

## Related

- [Titans Paper](https://arxiv.org/abs/2501.00663)
- [Miras Paper](https://arxiv.org/abs/2501.01951)
- [TNT Paper](https://arxiv.org/abs/2501.02116)

---

*Last updated: December 17, 2024*

# Atlas: Miras Framework Implementation

A minimal, observable implementation of the Behrouz et al. memory architecture framework (Titans/Miras), with episodic memory training and grokking detection.

## Current Status

| Model | Parameters | Status | Description |
|-------|------------|--------|-------------|
| 10M Shakespeare | 10.4M | **Training** | Multi-task episodic memory + grokking |
| 10M Dumas | 10.4M | **Training** | Multi-task episodic memory + grokking |
| 300M Omega | 389.5M | Completed | [Training Report](runs/atlas_300m_omega/TRAINING_REPORT.md) |

### Current Focus: Domain Universality Hypothesis

Testing whether grokking is domain-universal or domain-specific via multi-task training:

| Task | Purpose | Metrics |
|------|---------|---------|
| **Masked Word Prediction** | Memory helps language | Exact word accuracy |
| **Modular Arithmetic** | Validates grokking mechanism | Fourier concentration, Circular fit |

**Core Hypothesis:** If math shows geometric signal formation (Fourier/Circular) AND language retrieval improves simultaneously, grokking is universal. If domains behave independently, different approaches needed.

### Training Regime

**Multi-Task Episodic Memory** with 50-50 split:
- 50% Language (Shakespeare/Dumas corpus, masked word prediction)
- 50% Math (modular arithmetic mod 97)
- Random interleaving (prevents pattern exploitation)

**Phase Structure per Episode:**
```
Episode N:
┌───────────────────────────────────────────────────────┐
│ STORAGE PHASE (10 batches, random Lang/Math mix)      │
│ → Memory stores both domains via gradient updates     │
│ → Gate mode: STORAGE                                  │
└───────────────────────────────────────────────────────┘
                          ↓
┌───────────────────────────────────────────────────────┐
│ RETRIEVAL PHASE (10 batches, same order as storage)   │
│ Language: "To be or [MASK] to be" → masked_word_acc   │
│ Math: "23 + 45 = [MASK]" → math_accuracy + Fourier    │
│ → Gate mode: RETRIEVAL (10x penalty for errors)       │
└───────────────────────────────────────────────────────┘
```

---

## Key Innovations

### Gate Floor Scheduling (Collapse Prevention)

The 300M run showed gate collapse (99.3% attention, 0.7% memory). Solved with floor scheduling:

| Phase | Condition | Gate Floor | Purpose |
|-------|-----------|------------|---------|
| Phase 1 | < 10K steps | 30% | Force memory engagement |
| Phase 2 | < 30K steps | 10% | Allow optimization |
| Phase 3 | ≥ 30K steps | 5% | Near-free gates |

### Grokking Detection

Real-time phase detection based on geometric metrics:

| Phase | Fourier | Math Acc | Characteristics |
|-------|---------|----------|-----------------|
| **Memorization** | < 0.3 | Low | Training data overfitting |
| **Circuit Formation** | Rising | Rising | Structure emerging |
| **Cleanup** | > 0.6 | > 70% | Generalizing circuits |
| **Grokking** | > 0.8 | > 90% | Sudden generalization |

### Numerical Stability

- **StableMax**: Softmax alternative that prevents collapse with floor value
- **PerpGrad**: Orthogonal gradient projection for better generalization
- **Gradient clipping**: 1.0 global norm

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
| **Learning Algorithm** | Momentum (β=0.9) | With Q-K projection from TNT |

### Memory Module

Each layer maintains:
- **M**: Memory matrix [batch, d_key, d_value] - associative storage
- **S**: Surprise accumulator [batch, d_key, d_value] - gradient momentum
- **Gates**: Input-dependent α, η, θ for retention control

---

## Directory Structure

```
Atlas/
├── README.md
├── CLAUDE.md                       # Development context
├── configs/
│   ├── atlas_shakespeare.yaml      # 10M Shakespeare episodic
│   ├── atlas_dumas.yaml            # 10M Dumas episodic
│   └── atlas_300m_omega.yaml       # 300M (completed)
├── docs/
│   ├── ARCHITECTURE.md             # Design document
│   └── 1B_BUILD_PLAN.md            # Future 1B scaling
├── src/
│   ├── model/
│   │   ├── atlas_omega.py          # AtlasOmega model
│   │   ├── memory.py               # TitansMemory module
│   │   ├── attention.py            # Sliding window attention
│   │   └── retention.py            # Retention gates
│   ├── data/
│   │   ├── loader.py               # Dolmino dataset loader
│   │   ├── modular_arithmetic.py   # Grokking math dataset
│   │   └── multi_task_loader.py    # Multi-task mixing
│   └── training/
│       ├── trainer.py              # Base trainer
│       ├── episodic_trainer.py     # Episodic memory training
│       ├── masking.py              # Masked word prediction
│       ├── retrieval_verifier.py   # Retrieval verification
│       └── metrics.py              # Observability metrics
├── training_framework/
│   └── monitoring/
│       └── grokking_metrics.py     # Grokking phase detection
├── scripts/
│   ├── train_episodic.py           # Multi-task episodic training
│   ├── train_ddp_omega.py          # DDP training (300M)
│   └── test_inference.py           # Inference testing
├── runs/
│   ├── shakespeare_10m/            # Current Shakespeare run
│   ├── dumas_10m/                  # Current Dumas run
│   └── atlas_300m_omega/           # 300M run (completed)
└── tokenizer/
    └── atlas_tokenizer/            # LLaMA tokenizer
```

---

## Quick Start

### Train Multi-Task Episodic Model

```bash
cd /home/todd/olympus/models/Atlas
source venv/bin/activate

# Shakespeare corpus (cuda:0)
python scripts/train_episodic.py \
  --config configs/atlas_shakespeare.yaml \
  --device cuda:0

# Dumas corpus (cuda:1)
python scripts/train_episodic.py \
  --config configs/atlas_dumas.yaml \
  --device cuda:1
```

### Monitor Training

Key metrics to watch:

| Metric | Good Sign | Concern |
|--------|-----------|---------|
| `math_accuracy` | Rising to >90% | Stuck near random (1%) |
| `masked_word_accuracy` | Rising with math | Flat while math rises |
| `fourier_concentration` | Rising 0.3→0.8+ | Stuck below 0.3 |
| `circular_fit` | Improving >0.5 | Stays near 1.0 (random) |
| `gate_mean` | Above floor, stable | Collapsing to floor |

### Run Inference

```bash
python scripts/test_inference.py \
  --checkpoint runs/shakespeare_10m/checkpoints/best.pt \
  --prompt "To be or not to be" \
  --device cuda:0
```

---

## Key Findings

### 300M Run (December 2024)

- **Final PPL:** 229
- **Gate Collapse:** 99.3% attention, 0.7% memory by step 76K
- **Conclusion:** Model bypassed memory entirely; led to floor scheduling innovation

### 10M Episodic Runs (Current)

**Innovations being validated:**
1. Gate floor scheduling prevents collapse
2. Episodic storage/retrieval phases force memory usage
3. Multi-task training enables geometric metric validation
4. Grokking detection identifies phase transitions

---

## Observability

### Per-Step Metrics

Memory state:
- `M_norm`, `M_std`, `M_max` - Memory matrix health
- `S_norm` - Surprise accumulator activity

Gates:
- `gate_mean`, `gate_std` - Attention/memory balance per layer
- `gate_collapse_risk` - Distance from floor

Grokking (math task only):
- `fourier_concentration` - Spectral structure formation
- `circular_fit` - Mod-p circular structure
- `grokking_phase` - Current phase detection

Task accuracies:
- `math_accuracy` - Modular arithmetic (meaningful for grokking)
- `masked_word_accuracy` - Language retrieval (memory utility)
- `overall_accuracy` - Combined signal

---

## Hardware

| GPU | VRAM | Use Case |
|-----|------|----------|
| RTX A6000 (cuda:0) | 48GB | 10M Shakespeare |
| RTX A6000 (cuda:1) | 48GB | 10M Dumas |
| H100 (cloud) | 80GB | Future 1B training |

---

## Related Papers

- [Titans Paper](https://arxiv.org/abs/2501.00663) - Test-time memorization via gradient-based updates
- [Miras Paper](https://arxiv.org/abs/2501.01951) - "It's All Connected" - 4 design axes framework
- [TNT Paper](https://arxiv.org/abs/2501.02116) - Hierarchical training, 17x speedup
- [ATLAS Paper](https://arxiv.org/abs/2501.02739) - Optimal memory allocation at inference
- [Nested Learning](https://arxiv.org/abs/2501.03739) - Architecture = optimization at different levels
- [Grokking (Power et al.)](https://arxiv.org/abs/2201.02177) - Generalization beyond overfitting

---

*Last updated: December 22, 2024*

# Experiment 002: 300M Omega Training Run

**Status:** Archived
**Date:** December 10-17, 2024
**Duration:** 6.75 days (162 hours)
**Outcome:** Training completed, gate collapse observed

## Hypothesis

Train a 389M parameter Atlas model with Omega (momentum-based) memory to validate:
1. The Miras 4-axis architecture at scale
2. Memory module learns useful representations
3. PPL improvements translate to coherent generation

## Approach

### Architecture (Miras Framework)

| Axis | Choice | Implementation |
|------|--------|----------------|
| **Memory** | Matrix M [1024×1024] | Per-layer associative storage |
| **Bias** | L2 | Squared error objective |
| **Retention** | Local + Global | Input-dependent alpha, eta, theta |
| **Learning** | Momentum (β=0.9) | Omega rule with surprise accumulator S |

### Configuration

| Parameter | Value |
|-----------|-------|
| d_model | 1024 |
| n_layers | 16 |
| n_heads | 8 |
| Parameters | 389.5M |
| Training tokens | ~58B |

### TNT Two-Stage Training

| Stage | Chunk Size | Steps | Purpose |
|-------|------------|-------|---------|
| Stage 1 | 2048 | 100,000 | Bulk pre-training |
| Stage 2 | 256 | 10,000 | Fine-grained tuning |

## What Happened

### The Good

- Training pipeline completed 110K steps without crashes
- PPL decreased from 30,000 → 229 (successful optimization)
- Memory architecture implementation validated
- Observability metrics worked excellently

### The Problem: Gate Collapse

**Critical finding:** The model learned to bypass memory entirely.

| Step | Avg Gate | Interpretation |
|------|----------|----------------|
| 639 | 50.1% | Balanced (initialization) |
| 33K | 11.7% | Declining |
| 55K | 2.3% | Attention favored |
| 76K | 0.7% | Memory bypassed |
| **110K** | **0.66%** | **99.3% attention, 0.7% memory** |

The gating mechanism collapsed from 50% to <1%, causing the model to ignore the memory module and rely solely on attention.

### Inference Quality

Despite good PPL, generation was incoherent:

```
Prompt: "The capital of France is"
Output: "The capital of France is a man, and the story, and the same to
         the world, and the same, and the other, and the story..."
```

## Root Cause Analysis

1. **No gate floor**: Nothing prevented gates from collapsing to zero
2. **Optimization path**: Attention gradients were stronger/more stable early on
3. **Capacity allocation**: Model found it "easier" to ignore memory
4. **Memory stagnation**: M matrix norm clamped at 50, no structure developed

## Lessons Learned

### What We Confirmed

1. Titans/Miras architecture is implementable and trainable
2. Per-layer observability (M_norm, S_norm, gates) is essential
3. Surprise accumulator S shows gradient flow (memory IS receiving signal)
4. The issue is gate dynamics, not memory capacity

### What We Learned

1. **Gate regularization required**: Need minimum gate floor (0.1-0.3)
2. **Phase-based training**: Consider separate "storage" and "retrieval" phases
3. **389M insufficient**: Need 1B+ for coherent generation with this architecture
4. **Early warning needed**: Alert when gates fall below threshold

## What Came Next

This experiment directly informed the **Episodic Training** approach:

1. **Explicit storage/retrieval phases**: Force memory usage through training structure
2. **GateMode enum**: `STORAGE` forces high gates, `RETRIEVAL` ensures minimum
3. **Phase-based gate floors**: Different constraints per training phase
4. **Retrieval accuracy metric**: Verify memory actually stores/retrieves

## Files

- Full training report: `../../runs/atlas_300m_omega/TRAINING_REPORT.md`
- Configuration: `../../configs/atlas_300m_omega.yaml`
- Checkpoints: `../../runs/atlas_300m_omega/checkpoints/`
- Metrics: `../../runs/atlas_300m_omega/metrics/`

## Key Metrics Reference

| Metric | Final Value | Target |
|--------|-------------|--------|
| Perplexity | 229 | <100 for coherent text |
| Avg Gate | 0.66% | >20% (memory utilization) |
| M_norm | 50.0 (clamped) | Should vary |
| S_norm | 600-3400 | Active (good sign) |

## References

- [Full Training Report](../../runs/atlas_300m_omega/TRAINING_REPORT.md)
- [1B Build Plan](../../docs/1B_BUILD_PLAN.md) - incorporates these lessons

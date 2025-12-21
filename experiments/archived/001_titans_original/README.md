# Experiment 001: Original Titans Implementation

**Status:** Archived
**Date:** November 2024
**Outcome:** Superseded by Omega architecture

## Hypothesis

Implement the Titans paper ("Learning to Memorize at Test Time") directly, using:
- Polynomial feature expansion (φ_p mapping)
- Muon optimizer for memory updates
- DeepTransformer with deep memory layers

## Approach

This implementation followed the Titans paper closely:
- `PolynomialFeatures`: Multivariate monomial expansion for super-linear capacity
- `DeepMemory`: Gradient-based memory updates at test time
- `SlidingWindowAttention` + `GatedAttentionUnit`: Hybrid attention mechanism

### Key Components

```text
titans_atlas/
├── layers/
│   ├── memory.py       # DeepMemory with polynomial features
│   └── attention.py    # Sliding window + gated attention
├── models/
│   └── atlas.py        # Main model combining all components
└── examples/
    └── train_atlas.py  # Training script
```

## What Happened

The implementation was technically complete but encountered challenges:

1. **Complexity**: Polynomial feature expansion added significant computational overhead
2. **Training instability**: The Muon optimizer required careful tuning
3. **Architecture evolution**: The Miras framework paper ("It's All Connected") provided a cleaner 4-axis design

## Lessons Learned

1. **Simpler is better initially**: The Omega rule (momentum-based updates) proved more stable than full polynomial expansion
2. **Miras framework**: Organizing design around 4 axes (Memory, Bias, Retention, Learning) clarified architecture decisions
3. **Observability first**: Added detailed metrics logging to understand training dynamics

## What Came Next

Transitioned to the **Omega architecture** (`src/model/atlas_omega.py`) which:
- Uses matrix-valued memory (M) instead of polynomial features
- Implements momentum-based updates (Omega rule)
- Follows Miras 4-axis design strictly
- Led to successful 300M parameter training run

## Files Preserved

- `code/` - Complete implementation (importable as `titans_atlas` package)
- `code/examples/` - Training scripts and configs used
- `code/metrics/` - Early convergence and weaver space metrics

## References

- [Titans Paper](https://arxiv.org/abs/2501.00663) - Original inspiration
- [Miras Paper](https://arxiv.org/abs/2501.01951) - Framework that guided evolution

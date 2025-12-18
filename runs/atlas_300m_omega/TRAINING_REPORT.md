# Atlas 300M Omega Training Report

**Model:** Atlas 300M Omega (389.5M parameters)
**Run ID:** atlas_300m_omega
**Completed:** December 17, 2024
**Duration:** 38.94 hours

---

## Executive Summary

The Atlas 300M Omega model completed full training (110,000 steps) implementing the Titans/Miras memory architecture. While the model did not achieve coherent text generation, the run provided **critical insights into memory module dynamics** that will inform the 1B scale-up.

**Key Finding:** The learned gating mechanism collapsed from 50% memory utilization to <1%, causing the model to bypass the memory module entirely in favor of standard attention. This "gate collapse" phenomenon is a primary target for the 1B build.

---

## 1. Model Configuration

### Architecture

| Parameter | Value |
|-----------|-------|
| `d_model` | 1024 |
| `n_layers` | 16 |
| `n_heads` | 8 |
| `d_ff` | 4096 |
| `vocab_size` | 32000 |
| `max_seq_len` | 4096 |
| `d_key` | 1024 |
| `d_value` | 1024 |
| `poly_degree` | 2 |
| `context_window` | 16 |
| **Total Parameters** | **389,531,760** |

### Memory Module (Titans/Miras)

- **Memory Matrix M:** [batch, 1024, 1024] per layer
- **Surprise Accumulator S:** [batch, 1024, 1024] per layer
- **Retention Gates:** Input-dependent alpha, eta, theta
- **Gating:** Learned attention-memory balance per layer

### Training Configuration

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Steps | 100,000 | 10,000 |
| Chunk Size | 2048 | 256 |
| Batch Size | 2 | 2 |
| Grad Accum | 64 | 64 |
| Effective Batch | 524,288 tokens | 524,288 tokens |
| Learning Rate | 1.5e-4 | 1.5e-4 (decayed) |

---

## 2. Training Summary

### Timeline

| Milestone | Step | Time | PPL |
|-----------|------|------|-----|
| Start | 0 | 0h | ~30,000 |
| 10K | 10,000 | 3.5h | ~1,200 |
| 25K | 25,000 | 8.7h | ~500 |
| 50K | 50,000 | 17.4h | ~350 |
| 70K | 70,000 | 24.4h | ~300 |
| 90K | 90,000 | 31.4h | ~250 |
| Stage 2 Start | 100,000 | 34.9h | ~230 |
| **Complete** | **110,000** | **38.9h** | **~229** |

### Final Metrics

| Metric | Value |
|--------|-------|
| Final Loss | 5.429 |
| Final Perplexity | ~228.6 |
| Total Tokens | ~57.7B |
| Tokens/Parameter | 148 (7.4x Chinchilla optimal) |
| GPU | NVIDIA RTX A6000 (48GB) |
| VRAM Usage | ~26 GB (54%) |

### PPL Trajectory

```
PPL
30K |
    |#
10K | #
    |  ##
 1K |    ####
    |        #########
 200|                  ####################
    +---------------------------------------- Steps
    0    20K   40K   60K   80K   100K  110K
```

PPL decreased rapidly in early training, then plateaued around 200-350 with high variance through Stage 2.

---

## 3. Memory Module Analysis

### Critical Finding: Gate Collapse

The most significant observation is the **progressive collapse of gate values** from balanced (50%) to near-zero (<1%):

| Step | Avg Gate | Interpretation |
|------|----------|----------------|
| 639 | 50.1% | Balanced (initialization) |
| 11,519 | 25.9% | Balanced |
| 33,279 | 11.7% | Balanced |
| 44,159 | 4.2% | Attention favored |
| 55,039 | 2.3% | Attention favored |
| 76,799 | 0.7% | Memory bypassed |
| 98,559 | 0.8% | Memory bypassed |
| **108,959** | **0.7%** | **Memory bypassed** |

**The model learned to bypass memory in favor of attention.**

### Gate Values by Layer (Final)

```
Layer  0: 1.31% |#
Layer  1: 0.54% |
Layer  2: 0.16% |
Layer  3: 0.65% |
Layer  4: 0.63% |
Layer  5: 1.09% |#
Layer  6: 0.95% |
Layer  7: 0.61% |
Layer  8: 1.04% |#
Layer  9: 0.51% |
Layer 10: 0.12% |
Layer 11: 0.27% |
Layer 12: 0.45% |
Layer 13: 0.28% |
Layer 14: 1.25% |#
Layer 15: 0.70% |

Average: 0.66% (99.3% attention, 0.7% memory)
```

### Memory Matrix State

| Metric | Step 639 | Step 55K | Step 109K | Observation |
|--------|----------|----------|-----------|-------------|
| M_norm | 50.0 | 50.0 | 50.0 | Clamped at ceiling |
| M_std | 0.0282 | 0.0282 | 0.0282 | No structure developing |
| M_max | 0.24 | 0.42 | 0.48 | Slight increase |
| M_mean | -0.00003 | 0.00001 | -0.00008 | Near zero |

The memory matrix M remained essentially static throughout training - norm clamped, std constant, no differentiation.

### Surprise Accumulator (Active)

Despite gate collapse, the surprise accumulator S showed significant activity:

```
Layer  2: S_norm = 2791  ###########################
Layer 14: S_norm = 3390  #################################
Layer  3: S_norm =  612  ######
```

**Gradient signal IS flowing to memory** (S_norm varies 600-3400), but the collapsed gates prevent it from influencing output.

### Root Cause Hypotheses

1. **Capacity Bottleneck:** 389M parameters insufficient for both attention and memory; model optimized the "easier" path (attention)

2. **Optimization Dynamics:** Attention gradients stronger/more stable than memory gradients early in training

3. **No Gate Regularization:** Nothing prevented gates from collapsing to zero

4. **Initialization:** Starting gates at 0.5 may not be optimal

---

## 4. Inference Evaluation

### Test Prompts (Final Checkpoint 110K)

**Prompt:** "The quick brown fox"
```
"You's your eyes.
"I're okay. I just anymore. I said.
"I'm going to a busy. Why?"
"I'm going to me?"
```

**Prompt:** "Once upon a time"
```
"
"
"
"I's
"I's,
"I'm you'll be a
```

**Prompt:** "She walked into the room and"
```
"I'm going to be a moment."
"I't know about the little time, and I't know that I're sure
I't want to do with him. I can't like you'm of the other.
```

**Prompt:** "The capital of France is"
```
The capital of France is a man, and the story, and the same to
the world, and the same, and the other, and the story...
```

### Quality Assessment

| Aspect | Score | Notes |
|--------|-------|-------|
| Word-level tokens | Pass | Produces valid English words |
| Grammar | Fail | Broken ("I't", "I're", "you'm") |
| Coherence | Fail | No topic maintenance |
| Repetition | Fail | Loops ("and the same, and the other") |
| Factual recall | Fail | None |
| **Overall** | **Not usable** | Requires scale-up |

### Comparison Across Training

| Checkpoint | PPL | Coherence | Repetition |
|------------|-----|-----------|------------|
| 70K | 300 | Fragments | Present |
| 90K | 250 | Fragments | Present |
| 105K | 250 | Fragments | Present |
| 110K | 229 | Fragments | Present |

No significant improvement in generation quality despite continued PPL decrease.

---

## 5. Lessons Learned

### What Worked

1. **Training Pipeline:** Full 110K steps completed without crashes (one resume after terminal disconnect)
2. **Memory Architecture:** Titans/Miras implementation is functional
3. **Observability:** Per-layer metrics (M_norm, S_norm, gates) provide excellent visibility
4. **TNT Two-Stage:** Stage transition worked correctly
5. **Checkpointing:** Regular saves enabled analysis and recovery

### What Didn't Work

1. **Gate Dynamics:** Collapsed to <1%, bypassing memory entirely
2. **Text Coherence:** 389M insufficient for coherent generation
3. **Memory Utilization:** M matrix remained static throughout training
4. **SMS Notifications:** 3-6 hour delays via carrier gateways

### Architecture Observations

1. **Memory capacity:** Each layer has 1024x1024 = 1M parameters in M alone, but model learned to ignore it
2. **Surprise active:** S_norm shows gradients flowing, but gates block utilization
3. **Layer patterns:** Early layers (0, 5, 8, 14) retained slightly higher gate values

### Data Quality Issues

Several PPL spikes observed (>1000), suggesting problematic batches:
- Step 45K: PPL spike to 1,209
- Step 90K: PPL spike to 812
- Likely noisy/malformed samples in training data

---

## 6. Recommendations for 1B Build

### Architecture Changes

| Issue | Recommendation |
|-------|----------------|
| Gate collapse | Add minimum gate regularization (floor = 0.1) |
| Gate collapse | Initialize gates at 0.3 instead of 0.5 |
| Gate collapse | Consider separate LR for gate parameters |
| Memory stagnation | Increase memory_max_norm or remove ceiling |
| Memory stagnation | Add auxiliary memory utilization loss |
| Capacity | Scale to 1B+ parameters |

### Monitoring Additions

1. **Gate Alert:** Trigger if avg gate < 0.1 (early warning)
2. **Memory Activity:** Track M matrix entropy/rank
3. **Telegram Notifications:** Replace SMS with instant alerts
4. **Batch Provenance:** Log source file/line for debugging spikes

### Training Strategy

1. **Token Budget:** 20-40B tokens (vs 58B for 300M)
2. **Hardware:** H100/H200 for cost efficiency (~$60-100)
3. **Checkpoints:** Geometric snapshots every 20K steps
4. **Evaluation:** Inference tests every 10K steps

---

## 7. Files and Artifacts

### Checkpoints

```
runs/atlas_300m_omega/checkpoints/
├── checkpoint_5000.pt
├── checkpoint_10000.pt
├── checkpoint_15000.pt
├── ...
├── checkpoint_100000.pt
├── checkpoint_105000.pt
├── checkpoint_109001.pt  (early-stop backup)
└── checkpoint_110000.pt  (final)
```

### Metrics

```
runs/atlas_300m_omega/metrics/
├── train_steps.jsonl     # Per-step training metrics
└── val_metrics.jsonl     # Validation metrics
```

### Logs

```
runs/atlas_300m_omega/
├── training.log          # Initial run
└── training_resumed.log  # After terminal disconnect
```

---

## 8. Conclusion

The Atlas 300M Omega training run successfully validated the training pipeline and memory architecture implementation, but revealed a critical **gate collapse phenomenon** that prevented memory utilization.

**Key Takeaways:**

1. The Titans/Miras architecture is implementable and trainable
2. 389M parameters insufficient for coherent generation with this architecture
3. Gate dynamics require explicit regularization to prevent collapse
4. Memory observability is excellent - we can track what's happening
5. Scale-up to 1B is the logical next step

**Next Steps:**

1. Implement gate regularization for 1B build
2. Create 1B configuration (d_model=1536, n_layers=24)
3. Set up Telegram notifications (replace SMS)
4. Deploy on cloud (H100/H200)
5. Train 1B model with enhanced monitoring

---

## Appendix A: Hardware Configuration

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A6000 |
| VRAM | 48 GB GDDR6 |
| Driver | 535.154.05 |
| CUDA | 12.2 |
| PyTorch | 2.1.0 |
| OS | Ubuntu 22.04 |

## Appendix B: Training Command

```bash
# Initial run
CUDA_VISIBLE_DEVICES=1 python scripts/train_ddp_omega.py \
  --config configs/atlas_300m_omega.yaml \
  --output-dir runs/atlas_300m_omega

# Resume after disconnect
CUDA_VISIBLE_DEVICES=1 python scripts/train_ddp_omega.py \
  --config configs/atlas_300m_omega.yaml \
  --resume runs/atlas_300m_omega/checkpoints/checkpoint_80000.pt
```

## Appendix C: Cost Analysis

| Resource | Usage | Cost |
|----------|-------|------|
| GPU Time | 38.94 hours | $0 (local) |
| Electricity | ~12 kWh | ~$2 |
| **Total** | | **~$2** |

Equivalent cloud cost (H100 @ $2.39/hr): ~$93

## Appendix D: Comparison to 62M Baseline

| Metric | 62M Model | 300M Model |
|--------|-----------|------------|
| Parameters | 62.6M | 389.5M |
| Final PPL | ~270 | ~229 |
| Training Time | ~14 hours | ~39 hours |
| Steps | 81,501 (early stop) | 110,000 (full) |
| Coherent Output | No | No |
| Memory Used | Yes (gates ~5%) | No (gates <1%) |

The 62M model maintained slightly higher gate values but still produced incoherent output. The 300M model showed complete gate collapse despite more capacity.

---

*Report generated: December 17, 2024*
*Atlas Project: https://github.com/r3d91ll/Atlas*

# Atlas: Learning to Optimally Memorize at Test Time

PyTorch implementation of the Atlas neural long-term memory architecture from:

**Atlas**: [Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735) (arXiv:2505.23735)

---

## Current Status: Experiments Paused

**December 2025**: After extensive experimentation (v01-v04), we discovered that Atlas at 43M parameters on 831M tokens leads to rapid memorization rather than generalization. See [Experiment Summary](docs/experiments/EXPERIMENT_SUMMARY.md) for details.

**Key Findings**:
1. Atlas requires significantly more parameters (500M+) for the memory mechanisms to function without overfitting
2. Muon-style spectral normalization is required for 4K+ context stability
3. `model.eval()` breaks the architecture (undocumented train-mode dependency)
4. PPL decline **rate** is a better early-overfit detector than absolute values

**Current Direction**: Pivoting to BDH (Hebbian learning) architecture at 100M scale for continued research.

---

## Overview

Atlas introduces neural memory modules that learn to memorize context at test time through gradient-based optimization. Key innovations:

- **Omega Rule**: Sliding window context optimization instead of per-token updates
- **Polynomial Features**: φ_p(x) expansion for O(d_k^p) capacity scaling
- **Muon Optimizer**: Second-order approximation for stable memory updates
- **Deep Memory MLPs**: Superlinear capacity O(d_k * d_v)
- **Learnable Taylor Kernel**: Softmax approximation with learnable coefficients

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Atlas Language Model

```python
from titans_atlas import Atlas
from titans_atlas.configs import AtlasConfig

config = AtlasConfig(
    d_model=512,
    num_layers=12,
    context_window=64,
    polynomial_degree=2,
    vocab_size=32000,
)

model = Atlas(config)
outputs = model(input_ids=tokens, labels=labels)
loss = outputs["loss"]

# Generation
generated = model.generate(prompt_tokens, max_new_tokens=100)
```

### DeepTransformer Backbone

```python
from titans_atlas import DeepTransformer
from titans_atlas.configs import AtlasConfig

config = AtlasConfig(
    d_model=512,
    num_layers=12,
    context_window=64,
    polynomial_degree=2,
)

model = DeepTransformer(config)
output, memory_states = model(x)  # x: (batch, seq_len, d_model)
```

## Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| **OmegaRule** | Sliding window optimization: min_M Σ γ_i · ℓ(M; k_i, v_i) |
| **PolynomialFeatures** | φ_p(x) = [x^β]_{|β|≤p} for capacity scaling |
| **MuonOptimizer** | Momentum + spectral normalization for memory updates |
| **DeepMemory** | Multi-layer MLP with O(d_k * d_v) capacity |
| **LearnableTaylorKernel** | exp(x) ≈ Σ a_i x^i approximation |

### Atlas Memory Update

The Omega rule optimizes memory over a sliding window:

```
min_M Σ_{i=t-c+1}^{t} γ_i^(t) · ||M(k_i) - v_i||²
```

Where:
- `c`: context window length
- `γ_i^(t)`: learned decay weights for token importance
- `M`: deep memory network

### Model Variants

| Model | Description |
|-------|-------------|
| **Atlas** | Full model with Muon optimizer |
| **OmegaNet** | Omega rule with standard gradient descent |
| **DeepTransformer** | Backbone combining Atlas memory + sliding window attention |

## Experiment Results (December 2025)

### Summary

| Experiment | Config | Outcome |
|------------|--------|---------|
| v01 | 256 seq, homegrown data | PPL ~200 (data quality issue) |
| v02 | 4K seq, Dolma, Muon fix | PPL 296 @ 5.6K steps (overfit) |
| v03 | + Dropout 0.1 | Same decline rate (killed @ 3K) |
| v04 | + SlowLR schedule | NaN @ step 110-250 |

### Key Discoveries

1. **Muon Normalization Required** (applied in v02):
   ```python
   # OmegaRule.forward() - spectral normalization for stability
   U, S, V = torch.linalg.svd(grad, full_matrices=False)
   grad_normalized = U @ V
   self.M = self.M - self.learning_rate * grad_normalized
   ```

2. **PPL Decline Rate as Overfit Detector**:
   - Healthy: Gradual decline, rate decreasing
   - Overfit: Constant/accelerating rate (40-120%/100 steps)

3. **model.eval() Breaks Atlas**:
   - Cannot disable dropout during warmup
   - Architecture has train-mode dependencies

### Root Cause

43M params on 831M tokens = 19.3 tokens/param (Chinchilla minimum ~20). Any architectural inefficiency tips into memorization.

### Recommendations

For future Atlas work:
- Scale to 500M+ parameters
- Use 10B+ tokens (not 831M)
- Monitor generalization gap with held-out validation
- Investigate model.eval() dependencies

See [docs/experiments/EXPERIMENT_SUMMARY.md](docs/experiments/EXPERIMENT_SUMMARY.md) for full analysis.

---

## Configuration Presets

```python
from titans_atlas.configs import atlas_small, atlas_medium, atlas_large

config = atlas_small()   # ~170M params
config = atlas_medium()  # ~400M params
config = atlas_large()   # ~760M params
```

## Performance

From the paper:
- Atlas achieves +80% accuracy improvement on 10M context BABILong
- Handles sequences >2M tokens with linear-time inference
- Maintains competitive performance on standard benchmarks

## Training

### Quick Start Training

```bash
cd titans_atlas/examples

# 1. Prepare data (tokenize text into binary format)
python tokenize_data.py --input /path/to/data.txt --output ./data/

# 2. Train with paper-standard settings (recommended)
python train_with_metrics.py --config config.yaml

# Or train with command-line arguments:
python train_with_metrics.py \
    --data-path ./data/train.bin \
    --batch-size 64 \
    --seq-length 256 \
    --max-steps 50000
```

### Paper-Standard Hyperparameters

The default configuration uses settings from the Atlas paper:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Learning Rate | 1e-4 | Peak LR |
| Warmup Steps | 5000 | 10% of training |
| LR Schedule | Cosine | Decays to 1e-5 |
| Batch Size | 64 | Per-GPU |
| Sequence Length | 256 | Tokens per sample |
| Context Window | 64 | Memory window size |
| Weight Decay | 0.1 | AdamW |

### Configuration File

Create a YAML config for full control:

```yaml
# config.yaml
experiment: my_experiment
model:
  d_model: 512
  num_layers: 6
  context_window: 64
  vocab_size: 32000
training:
  batch_size: 64
  seq_length: 256
  learning_rate: 1e-4
  warmup_steps: 5000
  max_steps: 50000
data:
  tokenized_path: "./data/train.bin"
```

### Monitoring Training

The training script outputs JSONL metrics compatible with various dashboards:

```bash
# Watch training progress
tail -f runs/atlas/TIMESTAMP/metrics.jsonl

# Or use the included dashboard
streamlit run dashboard.py --server.port 8050
```

### Resume Training

```bash
python train_with_metrics.py \
    --config config.yaml \
    --resume-from runs/atlas/TIMESTAMP/checkpoints/checkpoint_010000.pt
```

### Multi-GPU Training

Set the device in config or via command-line:

```bash
# Single GPU (GPU 1)
python train_with_metrics.py --config config.yaml --device cuda:1

# For DataParallel, use separate processes per GPU (Atlas memory doesn't support DP)
```

## Tests

```bash
python -m pytest titans_atlas/tests/ -v
```

## Project Context

This implementation is part of a research portfolio exploring memory-augmented architectures:

- **Todd_MemRAG**: Industry-standard RAG with embedding-based retrieval
- **[Todd_Titans](https://github.com/r3d91ll/Todd_Titans)**: Titans paper implementation (archived reference)
- **Todd_Atlas**: This repo - Atlas with continuous memory (experiments paused)
- **BDH**: Baby Dragon Hatchling - Hebbian learning architecture (current focus)

## Future Directions

### Potential Explorations (if resuming Atlas)

1. **Scale Experiments**:
   - 500M+ parameter model on larger dataset (FineWeb, SlimPajama)
   - Validate if memorization is purely a capacity issue

2. **Architecture Modifications**:
   - Add explicit regularization to memory modules
   - Test different polynomial degrees
   - Investigate Taylor kernel alternatives

3. **Weaver Space Analysis**:
   - Capture memory state trajectories
   - Analyze attention geometry evolution
   - Measure manifold curvature during training

4. **Long-Context Validation**:
   - Test on BABILong benchmark
   - Compare memory retrieval accuracy vs. standard attention
   - Measure inference latency at 1M+ tokens

### Current Research Direction

We are currently exploring **BDH (Baby Dragon Hatchling)** as an alternative architecture:

- **Location**: `/home/todd/olympus/AshesBelow/experiments/BDH_100M/`
- **Architecture**: 100M params, Hebbian learning, shared parameters
- **Hypothesis**: Hebbian gating provides natural regularization

The BDH approach offers:
- Natural regularization via sparse Hebbian gating
- Shared parameters prevent layer-specific memorization
- Simpler architecture without test-time optimization

---

## Citation

```bibtex
@article{behrouz2025atlas,
  title={Atlas: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and others},
  journal={arXiv preprint arXiv:2505.23735},
  year={2025}
}
```

## License

MIT

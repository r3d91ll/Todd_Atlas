# Atlas: Learning to Optimally Memorize at Test Time

PyTorch implementation of the Atlas neural long-term memory architecture from:

**Atlas**: [Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735) (arXiv:2505.23735)

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

## Key Components

### Polynomial Features

```python
from titans_atlas.models.atlas import PolynomialFeatures

phi = PolynomialFeatures(input_dim=64, degree=2)
# Maps 64-dim input to O(64²) dimensional features
features = phi(keys)  # Superlinear capacity
```

### Omega Rule

```python
from titans_atlas.models.atlas import OmegaRule

omega = OmegaRule(
    d_key=64,
    d_value=64,
    context_window=64,
    polynomial_degree=2,
)

output, memory_state = omega(keys, values, queries)
```

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

## Tests

```bash
python -m pytest titans_atlas/tests/ -v
```

## Project Context

This implementation is part of a research portfolio exploring memory-augmented architectures:

- **Todd_MemRAG**: Industry-standard RAG with embedding-based retrieval
- **[Todd_Titans](https://github.com/r3d91ll/Todd_Titans)**: Titans paper implementation (archived reference)
- **Todd_Atlas**: This repo - Atlas with continuous memory (active development)

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

# Titans & Atlas: Learning to Memorize at Test Time

PyTorch implementation of neural long-term memory architectures from:

1. **Titans**: [Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) (arXiv:2501.00663)
2. **Atlas**: [Learning to Optimally Memorize the Context at Test Time](https://arxiv.org/abs/2505.23735) (arXiv:2505.23735)

## Overview

These architectures introduce neural memory modules that learn to memorize context at test time through gradient-based optimization. Key innovations:

- **Surprise Metric**: Gradient-based importance signal with momentum
- **Memory Update Rule**: `M_t = (1 - α_t) * M_{t-1} + S_t`
- **Deep Memory MLPs**: Superlinear capacity O(d_k * d_v)
- **Omega Rule** (Atlas): Sliding window context optimization
- **Polynomial Features** (Atlas): O(d_k^p) capacity scaling

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Titans Language Model

```python
from titans_atlas import TitansLM
from titans_atlas.configs import TitansConfig

# Create config
config = TitansConfig(
    d_model=512,
    num_layers=12,
    variant="MAG",  # or "MAC", "MAL"
    vocab_size=32000,
)

# Create model
model = TitansLM(config, variant="MAG")

# Forward pass
outputs = model(input_ids=tokens, labels=labels)
loss = outputs["loss"]

# Generation
generated = model.generate(prompt_tokens, max_new_tokens=100)
```

### Atlas with Omega Rule

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
```

## Architecture Variants

### Titans Variants

| Variant | Description | Best For |
|---------|-------------|----------|
| **MAC** (Memory as Context) | Memory provides additional context for attention | Complex reasoning |
| **MAG** (Memory as Gating) | Parallel branches combined via gating | Efficiency |
| **MAL** (Memory as Layer) | Sequential memory → attention | Long sequences |

### Atlas Enhancements

- **Omega Rule**: Optimizes memory over sliding window instead of per-token
- **Polynomial Features**: φ_p(x) expansion for increased capacity
- **Muon Optimizer**: Second-order approximation for stable memory updates
- **Taylor Kernel**: Learnable softmax approximation

## Key Components

### Neural Memory Module

```python
from titans_atlas.layers import NeuralMemory

memory = NeuralMemory(
    d_model=512,
    d_key=64,
    d_value=64,
    num_memory_layers=2,
    use_momentum=True,
    use_forget_gate=True,
)

output, new_state = memory(x, memory_state=prev_state)
```

### Polynomial Features

```python
from titans_atlas.models.atlas import PolynomialFeatures

phi = PolynomialFeatures(input_dim=64, degree=2)
# Maps 64-dim input to O(64²) dimensional features
features = phi(keys)  # Superlinear capacity
```

## Training

```bash
# Train Titans on your data
python examples/train_titans.py \
    --variant MAG \
    --d_model 512 \
    --num_layers 12 \
    --epochs 100 \
    --batch_size 32
```

## Memory Equations

### Titans Memory Update
```
S_t = η_t * S_{t-1} - θ_t * ∇ℓ(M_{t-1}; x_t)  # Surprise
M_t = (1 - α_t) * M_{t-1} + S_t                # Update
```

### Atlas Omega Rule
```
min_M Σ_{i=t-c+1}^{t} γ_i^(t) · ||M(k_i) - v_i||²
```

## Performance

From the papers:
- Titans handles sequences >2M tokens
- Atlas achieves +80% accuracy improvement on 10M context BABILong
- Linear-time inference vs quadratic for Transformers
- Maintains competitive performance on standard benchmarks

## Configuration Presets

```python
from titans_atlas.configs import titans_small, titans_medium, titans_large, atlas_small

# Pre-configured models
config = titans_medium()  # ~400M params
config = atlas_small()    # With Omega rule
```

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

```bibtex
@article{behrouz2025titans,
  title={Titans: Learning to Memorize at Test Time},
  author={Behrouz, Ali and Zhong, Peilin and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2501.00663},
  year={2025}
}

@article{behrouz2025atlas,
  title={Atlas: Learning to Optimally Memorize the Context at Test Time},
  author={Behrouz, Ali and others},
  journal={arXiv preprint arXiv:2505.23735},
  year={2025}
}
```

## License

MIT

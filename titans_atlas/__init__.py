"""
Titans & Atlas: Learning to Memorize at Test Time

Implementation of:
- Titans: Learning to Memorize at Test Time (arXiv:2501.00663)
- Atlas: Learning to Optimally Memorize the Context at Test Time (arXiv:2505.23735)

This package provides PyTorch implementations of neural long-term memory modules
that learn to memorize context at test time through gradient-based optimization.
"""

from titans_atlas.models.titans import (
    TitansMAC,
    TitansMAG,
    TitansMAL,
)
from titans_atlas.models.atlas import (
    Atlas,
    OmegaNet,
    DeepTransformer,
)
from titans_atlas.layers.memory import (
    NeuralMemory,
    DeepMemory,
)
from titans_atlas.layers.attention import (
    SlidingWindowAttention,
    PersistentMemoryAttention,
)

__version__ = "0.1.0"
__all__ = [
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "Atlas",
    "OmegaNet",
    "DeepTransformer",
    "NeuralMemory",
    "DeepMemory",
    "SlidingWindowAttention",
    "PersistentMemoryAttention",
]

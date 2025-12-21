#!/usr/bin/env python3
"""
Unit tests for Atlas models.

Run with: python -m pytest tests/test_models.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_atlas.configs import AtlasConfig, MemoryConfig, AttentionConfig
from titans_atlas.layers.memory import DeepMemory, NeuralMemory, NeuralMemoryParallel
from titans_atlas.layers.attention import SlidingWindowAttention, PersistentMemoryAttention
from titans_atlas.models.atlas import (
    PolynomialFeatures,
    LearnableTaylorKernel,
    OmegaRule,
    Atlas,
    DeepTransformer,
)
from titans_atlas.utils import parallel_scan, RotaryEmbedding


class TestDeepMemory:
    """Tests for DeepMemory module."""

    def test_forward_shape(self):
        """Test output shape is correct."""
        memory = DeepMemory(d_key=64, d_value=64, num_layers=2)
        keys = torch.randn(2, 10, 64)
        output = memory(keys)
        assert output.shape == (2, 10, 64)

    def test_different_dims(self):
        """Test with different key/value dimensions."""
        memory = DeepMemory(d_key=32, d_value=64, num_layers=3)
        keys = torch.randn(2, 10, 32)
        output = memory(keys)
        assert output.shape == (2, 10, 64)


class TestNeuralMemory:
    """Tests for NeuralMemory module."""

    def test_forward_shape(self):
        """Test output shape."""
        memory = NeuralMemory(d_model=256, d_key=64, d_value=64)
        x = torch.randn(2, 10, 256)
        output, state = memory(x)
        assert output.shape == (2, 10, 256)
        assert state is not None

    def test_with_memory_state(self):
        """Test passing previous memory state."""
        memory = NeuralMemory(d_model=256, d_key=64, d_value=64)
        x1 = torch.randn(2, 10, 256)
        _, state1 = memory(x1)

        x2 = torch.randn(2, 10, 256)
        output, state2 = memory(x2, memory_state=state1)
        assert output.shape == (2, 10, 256)


class TestNeuralMemoryParallel:
    """Tests for parallel memory implementation."""

    def test_forward_shape(self):
        """Test output shape."""
        memory = NeuralMemoryParallel(d_model=256, d_key=64, d_value=64)
        x = torch.randn(2, 32, 256)
        output, new_mem = memory(x)
        assert output.shape == (2, 32, 256)


class TestSlidingWindowAttention:
    """Tests for SlidingWindowAttention."""

    def test_forward_shape(self):
        """Test output shape."""
        attn = SlidingWindowAttention(
            d_model=256, num_heads=4, d_head=64, window_size=16,
            use_flash=False
        )
        x = torch.randn(2, 32, 256)
        output, cache = attn(x)
        assert output.shape == (2, 32, 256)

    def test_with_mask(self):
        """Test with attention mask."""
        attn = SlidingWindowAttention(
            d_model=256, num_heads=4, d_head=64, window_size=16,
            use_flash=False
        )
        x = torch.randn(2, 32, 256)
        mask = torch.ones(2, 32)
        mask[:, 16:] = 0
        output, _ = attn(x, attention_mask=mask)
        assert output.shape == (2, 32, 256)


class TestPersistentMemoryAttention:
    """Tests for PersistentMemoryAttention."""

    def test_forward_shape(self):
        """Test output shape."""
        attn = PersistentMemoryAttention(
            d_model=256, num_heads=4, d_head=64, num_persistent=8
        )
        x = torch.randn(2, 32, 256)
        output = attn(x)
        assert output.shape == (2, 32, 256)


class TestPolynomialFeatures:
    """Tests for PolynomialFeatures."""

    def test_degree_1(self):
        """Test degree 1 (linear features)."""
        phi = PolynomialFeatures(input_dim=4, degree=1, include_bias=True)
        x = torch.randn(2, 10, 4)
        output = phi(x)
        # Output should be 1 + 4 = 5 features
        assert output.shape == (2, 10, 5)

    def test_degree_2(self):
        """Test degree 2 (quadratic features)."""
        phi = PolynomialFeatures(input_dim=3, degree=2, include_bias=True)
        x = torch.randn(2, 10, 3)
        output = phi(x)
        # Features: 1 + 3 + 6 = 10 (constant + linear + quadratic)
        assert output.shape == (2, 10, 10)


class TestLearnableTaylorKernel:
    """Tests for LearnableTaylorKernel."""

    def test_forward(self):
        """Test Taylor approximation."""
        kernel = LearnableTaylorKernel(order=4, learnable=True)
        x = torch.randn(2, 10)
        output = kernel(x)
        assert output.shape == (2, 10)

    def test_approximates_exp(self):
        """Test that it approximates exp(x) for small x."""
        kernel = LearnableTaylorKernel(order=6, learnable=False)
        x = torch.tensor([0.0, 0.1, 0.5])
        output = kernel(x)
        expected = torch.exp(x)
        assert torch.allclose(output, expected, atol=0.01)


class TestOmegaRule:
    """Tests for OmegaRule."""

    def test_forward_shape(self):
        """Test output shape."""
        omega = OmegaRule(d_key=32, d_value=32, context_window=16)
        keys = torch.randn(2, 20, 32)
        values = torch.randn(2, 20, 32)
        queries = torch.randn(2, 20, 32)
        output, state = omega(keys, values, queries)
        assert output.shape == (2, 20, 32)


class TestAtlas:
    """Tests for Atlas model."""

    @pytest.fixture
    def small_config(self):
        """Create small Atlas config."""
        return AtlasConfig(
            d_model=64,
            num_layers=2,
            context_window=16,
            polynomial_degree=2,
            memory=MemoryConfig(d_model=64, d_key=16, d_value=16, num_memory_layers=2),
            attention=AttentionConfig(d_model=64, num_heads=2, d_head=16, window_size=16,
                                     use_flash_attention=False),
            vocab_size=100,
            max_seq_len=64,
        )

    def test_deep_transformer(self, small_config):
        """Test DeepTransformer forward pass."""
        model = DeepTransformer(small_config)
        x = torch.randn(2, 32, 64)
        output, states = model(x)
        assert output.shape == (2, 32, 64)

    def test_atlas_lm(self, small_config):
        """Test Atlas language model."""
        model = Atlas(small_config)
        input_ids = torch.randint(0, 100, (2, 32))
        outputs = model(input_ids=input_ids, labels=input_ids)
        assert "logits" in outputs
        assert "loss" in outputs
        assert outputs["logits"].shape == (2, 32, 100)


class TestUtils:
    """Tests for utility functions."""

    def test_parallel_scan(self):
        """Test parallel scan implementation."""
        batch, seq_len, dim = 2, 16, 8
        gates = torch.rand(batch, seq_len, dim) * 0.5 + 0.5
        values = torch.randn(batch, seq_len, dim)
        initial = torch.zeros(batch, dim)

        output = parallel_scan(gates, values, initial)
        assert output.shape == (batch, seq_len, dim)

    def test_rotary_embedding(self):
        """Test rotary embeddings."""
        rotary = RotaryEmbedding(dim=32, max_seq_len=128)
        q = torch.randn(2, 4, 16, 32)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 16, 32)
        q_rot, k_rot = rotary(q, k, 16)
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

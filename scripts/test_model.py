#!/usr/bin/env python3
"""
Quick test script for Atlas model.

Tests:
- Model instantiation
- Forward pass shapes
- Memory state updates
- Gradient flow
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.atlas import Atlas, AtlasConfig, create_atlas_50m
from src.model.memory import MatrixMemory
from src.model.retention import RetentionGate
from src.model.attention import SlidingWindowAttention, GatingMechanism


def test_memory_module():
    """Test matrix memory module."""
    print("\n=== Testing MatrixMemory ===")

    memory = MatrixMemory(d_key=64, d_value=64)

    batch_size = 2
    seq_len = 16
    d_model = 64

    x = torch.randn(batch_size, seq_len, d_model)

    # Initialize state
    W, m = memory.init_state(batch_size, x.device)
    print(f"Initial W shape: {W.shape}")
    print(f"Initial m shape: {m.shape}")

    # Forward pass
    output, (W_new, m_new), metrics = memory(x, return_metrics=True)

    print(f"Output shape: {output.shape}")
    print(f"W_new shape: {W_new.shape}")
    print(f"Metrics: {metrics}")

    # Check shapes
    assert output.shape == (batch_size, seq_len, 64)
    assert W_new.shape == W.shape
    assert m_new.shape == m.shape

    # Check memory updated
    assert not torch.allclose(W, W_new), "Memory should have updated"

    print("Memory module: PASSED")


def test_retention_gate():
    """Test retention gate."""
    print("\n=== Testing RetentionGate ===")

    gate = RetentionGate(d_key=64, d_value=64)

    batch_size = 2
    W = torch.randn(batch_size, 64, 64)
    W_prev = torch.randn(batch_size, 64, 64)

    grad, metrics = gate(W, W_prev)

    print(f"Grad shape: {grad.shape}")
    print(f"Metrics: {metrics}")

    assert grad.shape == W.shape
    assert "lambda_local" in metrics
    assert "lambda_global" in metrics

    print("Retention gate: PASSED")


def test_attention():
    """Test sliding window attention."""
    print("\n=== Testing SlidingWindowAttention ===")

    attn = SlidingWindowAttention(
        d_model=64,
        n_heads=4,
        window_size=8,
    )

    batch_size = 2
    seq_len = 32
    x = torch.randn(batch_size, seq_len, 64)

    output, weights = attn(x, return_weights=True)

    print(f"Output shape: {output.shape}")
    print(f"Weights shape: {weights.shape}")

    assert output.shape == x.shape
    assert weights.shape == (batch_size, 4, seq_len, seq_len)

    print("Attention: PASSED")


def test_gating():
    """Test gating mechanism."""
    print("\n=== Testing GatingMechanism ===")

    gate = GatingMechanism(d_model=64)

    batch_size = 2
    seq_len = 16
    mem_out = torch.randn(batch_size, seq_len, 64)
    attn_out = torch.randn(batch_size, seq_len, 64)

    output, gate_values = gate(mem_out, attn_out, return_gate=True)

    print(f"Output shape: {output.shape}")
    print(f"Gate values shape: {gate_values.shape}")
    print(f"Gate mean: {gate_values.mean():.3f}")

    assert output.shape == mem_out.shape
    assert gate_values.shape == mem_out.shape
    assert (gate_values >= 0).all() and (gate_values <= 1).all()

    print("Gating: PASSED")


def test_atlas_model():
    """Test full Atlas model."""
    print("\n=== Testing Atlas Model ===")

    # Small config for testing
    config = AtlasConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        vocab_size=1000,
        max_seq_len=64,
        d_key=64,
        d_value=64,
        window_size=16,
        dropout=0.0,
    )

    model = Atlas(config)
    print(f"Parameters: {model.n_params:,}")

    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))

    # Forward pass
    logits, memory_states, metrics = model(input_ids, return_metrics=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Memory states: {len(memory_states)} layers")
    print(f"Metrics keys: {list(metrics.keys())}")

    assert logits.shape == (batch_size, seq_len, 1000)
    assert len(memory_states) == 2

    # Test loss computation
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    loss, _, _ = model.compute_loss(input_ids, labels)
    print(f"Loss: {loss.item():.4f}")

    # Test gradient flow
    loss.backward()
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"All gradients computed: {has_grad}")

    assert has_grad, "Some parameters have no gradient"

    print("Atlas model: PASSED")


def test_atlas_50m():
    """Test 50M parameter model configuration (paper specs)."""
    print("\n=== Testing 50M Configuration (Paper Specs) ===")

    model = create_atlas_50m()
    print(f"Parameters: {model.n_params:,}")
    print(f"Vocab size: {model.config.vocab_size}")

    # Verify vocab size matches T5 (32100)
    assert model.config.vocab_size == 32100, f"Expected vocab 32100, got {model.config.vocab_size}"

    # Should be around 50M (with 32K vocab)
    assert 40_000_000 < model.n_params < 70_000_000, f"Expected ~50M, got {model.n_params:,}"

    # Quick forward pass (CPU, small batch)
    input_ids = torch.randint(0, 32100, (1, 64))
    logits, _, _ = model(input_ids)

    print(f"Logits shape: {logits.shape}")
    assert logits.shape == (1, 64, 32100)

    print("50M configuration: PASSED")


def test_generation():
    """Test autoregressive generation."""
    print("\n=== Testing Generation ===")

    config = AtlasConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        vocab_size=1000,
        max_seq_len=64,
        d_key=64,
        d_value=64,
    )

    model = Atlas(config)
    model.eval()

    prompt = torch.randint(0, 1000, (1, 5))
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)

    print(f"Prompt length: {prompt.shape[1]}")
    print(f"Generated length: {generated.shape[1]}")

    assert generated.shape[1] == 15
    assert (generated[:, :5] == prompt).all()

    print("Generation: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("Atlas Model Tests")
    print("=" * 60)

    test_memory_module()
    test_retention_gate()
    test_attention()
    test_gating()
    test_atlas_model()
    test_atlas_50m()
    test_generation()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)

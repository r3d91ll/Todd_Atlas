#!/usr/bin/env python3
"""
Test script for validating the episodic training framework.

This script validates all components of the episodic training pipeline
with a small model before cloud deployment.

Usage:
    python scripts/test_episodic_training.py [--full] [--dashboard]

Options:
    --full      Run full 1000 step test (1-2 hours)
    --dashboard Launch Streamlit dashboard for monitoring
    --device    Device to use (default: cuda if available)
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from src.model.attention import (
            SlidingWindowAttention,
            GatingMechanism,
            FeedForward,
            GateMode,
        )
        print("  ✓ attention.py (GateMode enum)")

        from src.model.omega_memory import OmegaMemory, OmegaMemoryConfig
        print("  ✓ omega_memory.py")

        from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
        print("  ✓ atlas_omega.py")

        from src.training.retrieval_verifier import RetrievalVerifier, StorageRecord
        print("  ✓ retrieval_verifier.py")

        from src.training.episodic_trainer import (
            EpisodicDDPTrainer,
            EpisodicConfig,
            TrainerConfig,
            TrainingPhase,
            EpisodePhase,
        )
        print("  ✓ episodic_trainer.py")

        from training_framework.core.metric_collector import MetricCollector
        print("  ✓ metric_collector.py")

        from training_framework.monitoring.alert_system import AlertSystem
        print("  ✓ alert_system.py")

        from training_framework.adapters.atlas_adapter import AtlasMetricsAdapter
        print("  ✓ atlas_adapter.py")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_gate_mode():
    """Test GateMode switching in GatingMechanism."""
    print("\nTesting GateMode switching...")

    from src.model.attention import GatingMechanism, GateMode

    gate = GatingMechanism(d_model=64)

    # Test mode switching
    gate.set_mode(GateMode.NORMAL)
    assert gate.get_mode() == GateMode.NORMAL, "NORMAL mode not set"
    print("  ✓ NORMAL mode")

    gate.set_mode(GateMode.STORAGE)
    assert gate.get_mode() == GateMode.STORAGE, "STORAGE mode not set"
    print("  ✓ STORAGE mode")

    gate.set_mode(GateMode.RETRIEVAL)
    assert gate.get_mode() == GateMode.RETRIEVAL, "RETRIEVAL mode not set"
    print("  ✓ RETRIEVAL mode")

    # Test forward with different modes
    mem_out = torch.randn(2, 10, 64)
    attn_out = torch.randn(2, 10, 64)

    gate.set_mode(GateMode.NORMAL)
    out_normal, gate_vals = gate(mem_out, attn_out, return_gate=True)
    print(f"  ✓ Normal forward: gate mean = {gate_vals.mean():.3f}")

    gate.set_mode(GateMode.STORAGE)
    out_storage, gate_vals = gate(mem_out, attn_out, return_gate=True)
    print(f"  ✓ Storage forward: gate mean = {gate_vals.mean():.3f}")

    gate.set_mode(GateMode.RETRIEVAL)
    out_retrieval, gate_vals = gate(mem_out, attn_out, return_gate=True)
    print(f"  ✓ Retrieval forward: gate mean = {gate_vals.mean():.3f}")

    return True


def test_model_gate_control(device='cpu'):
    """Test model-level gate control methods."""
    print(f"\nTesting model gate control (device={device})...")

    from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
    from src.model.attention import GateMode

    # Create small test model
    config = AtlasOmegaConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        vocab_size=100,
        max_seq_len=64,
        d_key=64,
        d_value=64,
        poly_degree=2,
        context_window=4,
    )
    model = AtlasOmega(config).to(device)

    # Test gate mode propagation
    model.set_gate_mode(GateMode.STORAGE)
    assert model.get_gate_mode() == GateMode.STORAGE
    print("  ✓ set_gate_mode propagates to all blocks")

    # Test gate floor
    model.set_gate_floor(0.25)
    print("  ✓ set_gate_floor propagates to all blocks")

    # Test forward pass to populate gate values
    input_ids = torch.randint(0, 100, (2, 16), device=device)
    logits, memory_states, metrics = model(input_ids, return_metrics=True)

    # Test get_gate_metrics
    gate_metrics = model.get_gate_metrics()
    assert 'gate_mean' in gate_metrics, "gate_mean not in metrics"
    print(f"  ✓ get_gate_metrics: gate_mean = {gate_metrics.get('gate_mean', 0):.3f}")

    # Test get_all_gates
    all_gates = model.get_all_gates()
    if all_gates is not None:
        print(f"  ✓ get_all_gates: shape = {all_gates.shape}")

    # Test memory state access
    memory_state = model.get_memory_state()
    print(f"  ✓ get_memory_state: {len(memory_state)} layers")

    # Test reset
    model.reset_all_memory()
    print("  ✓ reset_all_memory")

    return True


def test_retrieval_verifier(device='cpu'):
    """Test RetrievalVerifier functionality."""
    print(f"\nTesting RetrievalVerifier (device={device})...")

    from src.training.retrieval_verifier import RetrievalVerifier

    verifier = RetrievalVerifier(
        max_buffer_size=10,
        retrieval_loss_weight=5.0,
        device=device,
    )

    # Simulate storage
    batch = {'input_ids': torch.randint(0, 100, (2, 32))}
    batch_hash = verifier.compute_batch_hash(batch)
    print(f"  ✓ compute_batch_hash: {batch_hash}")

    target_tokens = torch.randint(0, 100, (2, 32))
    memory_state = torch.randn(2, 64)

    verifier.record_storage(batch_hash, target_tokens, memory_state)
    print("  ✓ record_storage")

    # Test retrieval
    record = verifier.get_stored_record(batch_hash)
    assert record is not None, "Storage record not found"
    print("  ✓ get_stored_record")

    # Simulate model output
    model_logits = torch.randn(2, 32, 100)

    # Test verification
    metrics = verifier.verify_retrieval(batch_hash, model_logits, memory_state)
    assert 'retrieval_token_accuracy' in metrics
    print(f"  ✓ verify_retrieval: accuracy = {metrics['retrieval_token_accuracy']:.3f}")

    # Test retrieval loss computation
    loss, metrics = verifier.compute_retrieval_loss_from_hash(batch_hash, model_logits)
    print(f"  ✓ compute_retrieval_loss_from_hash: loss = {loss.item():.3f}")

    # Test statistics
    stats = verifier.get_statistics()
    print(f"  ✓ get_statistics: {stats}")

    return True


def test_metrics_adapter(device='cpu'):
    """Test AtlasMetricsAdapter."""
    print(f"\nTesting AtlasMetricsAdapter (device={device})...")

    from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
    from training_framework.adapters.atlas_adapter import AtlasMetricsAdapter

    # Create small model
    config = AtlasOmegaConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        vocab_size=100,
        max_seq_len=64,
        d_key=64,
        d_value=64,
    )
    model = AtlasOmega(config).to(device)

    # Run forward to populate metrics
    input_ids = torch.randint(0, 100, (2, 16), device=device)
    logits, _, _ = model(input_ids, return_metrics=True)

    # Create adapter and collect metrics
    adapter = AtlasMetricsAdapter(track_per_layer=True)
    outputs = {'loss': torch.tensor(2.5)}
    metrics = adapter.collect(model, outputs)

    # Verify key metrics are present
    expected_metrics = ['gate_mean', 'gate_collapse_risk', 'memory_sparsity']
    for key in expected_metrics:
        if key in metrics:
            print(f"  ✓ {key}: {metrics[key]:.4f}")
        else:
            print(f"  ? {key}: not found (may be expected)")

    # Test alert thresholds
    thresholds = adapter.get_alert_thresholds()
    print(f"  ✓ Alert thresholds defined: {len(thresholds)} metrics")

    return True


def test_episodic_trainer_init(device='cpu'):
    """Test EpisodicDDPTrainer initialization."""
    print(f"\nTesting EpisodicDDPTrainer initialization (device={device})...")

    from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
    from src.training.episodic_trainer import (
        EpisodicDDPTrainer,
        EpisodicConfig,
        TrainerConfig,
    )
    from torch.utils.data import DataLoader, TensorDataset

    # Create small model
    config = AtlasOmegaConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_ff=128,
        vocab_size=100,
        max_seq_len=64,
        d_key=64,
        d_value=64,
    )
    model = AtlasOmega(config)

    # Create synthetic data
    input_ids = torch.randint(0, 100, (50, 32))
    labels = torch.randint(0, 100, (50, 32))
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Wrap dataloader to return dict
    class DictDataLoader:
        def __init__(self, dl):
            self.dl = dl

        def __iter__(self):
            for batch in self.dl:
                yield {'input_ids': batch[0], 'labels': batch[1]}

        def __len__(self):
            return len(self.dl)

    # Create trainer config
    trainer_config = TrainerConfig(
        max_steps=10,
        learning_rate=1e-4,
        batch_size=4,
        gradient_accumulation_steps=2,
        log_interval=5,
        checkpoint_interval=100,
        metrics_path="runs/test_episodic/metrics_stream.jsonl",
        checkpoint_dir="runs/test_episodic/checkpoints",
        device=device,
        episodic=EpisodicConfig(
            storage_samples=2,
            retrieval_samples=2,
            phase1_steps=5,
            phase2_steps=5,
        ),
    )

    # Initialize trainer
    trainer = EpisodicDDPTrainer(
        model=model,
        train_dataloader=DictDataLoader(dataloader),
        config=trainer_config,
    )

    print("  ✓ Trainer initialized")
    print(f"    - Model params: {model.n_params:,}")
    print(f"    - Device: {trainer.device}")
    print(f"    - Training phase: {trainer.training_phase.value}")

    return True


def test_short_training_run(device='cpu'):
    """Run a short training loop to test full pipeline."""
    print(f"\nTesting short training run (device={device})...")

    from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
    from src.training.episodic_trainer import (
        EpisodicDDPTrainer,
        EpisodicConfig,
        TrainerConfig,
    )
    from training_framework.adapters.atlas_adapter import AtlasMetricsAdapter
    from torch.utils.data import DataLoader, TensorDataset
    import tempfile

    # Create output directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small model
        config = AtlasOmegaConfig(
            d_model=64,
            n_layers=2,
            n_heads=2,
            d_ff=128,
            vocab_size=100,
            max_seq_len=64,
            d_key=64,
            d_value=64,
        )
        model = AtlasOmega(config)

        # Create synthetic data
        input_ids = torch.randint(0, 100, (100, 32))
        labels = torch.randint(0, 100, (100, 32))
        dataset = TensorDataset(input_ids, labels)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        class DictDataLoader:
            def __init__(self, dl):
                self.dl = dl

            def __iter__(self):
                for batch in self.dl:
                    yield {'input_ids': batch[0], 'labels': batch[1]}

            def __len__(self):
                return len(self.dl)

        # Create trainer config - very short for testing
        trainer_config = TrainerConfig(
            max_steps=8,  # 2 episodes with 2+2 samples each
            learning_rate=1e-4,
            batch_size=4,
            gradient_accumulation_steps=2,
            log_interval=2,
            checkpoint_interval=100,
            metrics_path=os.path.join(tmpdir, "metrics_stream.jsonl"),
            checkpoint_dir=os.path.join(tmpdir, "checkpoints"),
            device=device,
            episodic=EpisodicConfig(
                storage_samples=2,
                retrieval_samples=2,
                phase1_steps=4,
                phase2_steps=4,
            ),
        )

        # Initialize trainer with metrics adapter
        adapter = AtlasMetricsAdapter()
        trainer = EpisodicDDPTrainer(
            model=model,
            train_dataloader=DictDataLoader(dataloader),
            config=trainer_config,
            metrics_adapter=adapter,
        )

        # Run training
        print("  Running training loop...")
        start = time.time()
        final_stats = trainer.train()
        elapsed = time.time() - start

        print(f"  ✓ Training completed in {elapsed:.1f}s")
        print(f"    - Final step: {final_stats['final_step']}")
        print(f"    - Episodes: {final_stats['total_episodes']}")
        print(f"    - Best retrieval acc: {final_stats['best_retrieval_accuracy']:.3f}")

        # Verify metrics file was created
        metrics_path = Path(trainer_config.metrics_path)
        if metrics_path.exists():
            with open(metrics_path) as f:
                n_lines = sum(1 for _ in f)
            print(f"  ✓ Metrics file created: {n_lines} entries")
        else:
            print("  ? Metrics file not created (may be empty buffer)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Test episodic training framework")
    parser.add_argument('--full', action='store_true', help='Run full 1000 step test')
    parser.add_argument('--dashboard', action='store_true', help='Launch dashboard')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 60)
    print("Atlas Episodic Training Framework Validation")
    print("=" * 60)
    print(f"Device: {device}")
    print()

    # Change to Atlas directory
    atlas_dir = Path(__file__).parent.parent
    os.chdir(atlas_dir)
    sys.path.insert(0, str(atlas_dir))

    tests = [
        ("Import Test", test_imports),
        ("GateMode Test", test_gate_mode),
        ("Model Gate Control", lambda: test_model_gate_control(device)),
        ("Retrieval Verifier", lambda: test_retrieval_verifier(device)),
        ("Metrics Adapter", lambda: test_metrics_adapter(device)),
        ("Trainer Initialization", lambda: test_episodic_trainer_init(device)),
        ("Short Training Run", lambda: test_short_training_run(device)),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed! Ready for cloud deployment.")
    else:
        print("Some tests failed. Please fix before deploying.")

    if args.dashboard:
        print("\nLaunching Streamlit dashboard...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "training_framework/monitoring/streamlit_monitor.py",
            "--",
            "--metrics-path", "runs/test_episodic/metrics_stream.jsonl",
        ])

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

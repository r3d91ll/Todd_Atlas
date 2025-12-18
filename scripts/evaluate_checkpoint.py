#!/usr/bin/env python3
"""
Re-evaluate checkpoints with corrected validation (accumulated memory states).

This script loads a checkpoint and runs validation WITH memory accumulation,
which matches how training actually works. The original validation code
incorrectly used fresh memory per batch.

Usage:
    python scripts/evaluate_checkpoint.py --checkpoint runs/atlas_50m_v4/checkpoints/checkpoint_5000.pt
    python scripts/evaluate_checkpoint.py --checkpoint runs/atlas_50m_v4/checkpoints/checkpoint_10000.pt
"""

import argparse
import yaml
import torch
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.atlas import Atlas, AtlasConfig
from src.data.loader import create_dataloaders
from torch.amp import autocast


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> Atlas:
    """Create Atlas model from config."""
    model_cfg = config["model"]
    model_config = AtlasConfig(
        d_model=model_cfg["d_model"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_key=model_cfg["d_key"],
        d_value=model_cfg["d_value"],
        momentum_beta=model_cfg["momentum_beta"],
        memory_lr_init=model_cfg["memory_lr_init"],
        learn_memory_lr=model_cfg["learn_memory_lr"],
        retention_local_init=model_cfg["retention_local_init"],
        retention_global_init=model_cfg["retention_global_init"],
        adaptive_retention=model_cfg["adaptive_retention"],
        window_size=model_cfg["window_size"],
        dropout=model_cfg["dropout"],
    )
    return Atlas(model_config)


@torch.no_grad()
def evaluate_with_accumulated_memory(model, val_loader, device, n_batches=100):
    """
    Run validation with accumulated memory states (CORRECTED approach).

    This matches how training works - memory accumulates across batches.
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    # Start with fresh memory (like start of training epoch)
    memory_states = None

    print(f"\nRunning validation with ACCUMULATED memory ({n_batches} batches)...")

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # Pass AND accumulate memory states (this is the fix!)
            loss, memory_states, _ = model.compute_loss(
                input_ids,
                labels,
                memory_states=memory_states,
            )

        total_loss += loss.item()
        batch_count += 1

        if (i + 1) % 20 == 0:
            avg_loss = total_loss / batch_count
            avg_ppl = math.exp(min(avg_loss, 20))
            print(f"  Batch {i+1}/{n_batches}: Loss={avg_loss:.4f}, PPL={avg_ppl:.2f}")

    avg_loss = total_loss / batch_count
    avg_ppl = math.exp(min(avg_loss, 20))

    return avg_loss, avg_ppl


@torch.no_grad()
def evaluate_without_memory(model, val_loader, device, n_batches=100):
    """
    Run validation WITHOUT memory accumulation (ORIGINAL buggy approach).

    This starts fresh each batch, which doesn't match training behavior.
    """
    model.eval()
    total_loss = 0.0
    batch_count = 0

    print(f"\nRunning validation WITHOUT memory accumulation ({n_batches} batches)...")

    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", input_ids).to(device)

        with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            # No memory states passed - fresh each batch (old buggy way)
            loss, _, _ = model.compute_loss(
                input_ids,
                labels,
                memory_states=None,
            )

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    avg_ppl = math.exp(min(avg_loss, 20))

    return avg_loss, avg_ppl


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoint with corrected validation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=Path, default=Path("configs/atlas_50m.yaml"))
    parser.add_argument("--n_batches", type=int, default=100, help="Number of validation batches")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    print("=" * 60)
    print("Atlas Checkpoint Re-Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print()

    # Load config
    config = load_config(args.config)

    # Create model
    model = create_model(config)
    print(f"Model parameters: {model.n_params:,}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Handle DDP wrapped state dict
    state_dict = checkpoint["model_state_dict"]
    # Remove "module." prefix if present (from DDP)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(args.device).bfloat16()

    step = checkpoint.get("step", "unknown")
    stage = checkpoint.get("stage", "unknown")
    print(f"Checkpoint step: {step}")
    print(f"Training stage: {stage}")

    # Create validation dataloader
    data_cfg = config["data"]
    train_cfg = config["training"]
    data_dir = Path(__file__).parent.parent / data_cfg["data_dir"]

    print(f"\nLoading validation data from: {data_dir}")
    _, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        tokenizer_name=data_cfg["tokenizer"],
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_split=data_cfg.get("val_split", 0.20),
    )

    # Run BOTH evaluation methods for comparison
    print("\n" + "=" * 60)
    print("COMPARISON: Old (buggy) vs New (corrected) validation")
    print("=" * 60)

    # Method 1: WITHOUT memory accumulation (old buggy way)
    old_loss, old_ppl = evaluate_without_memory(
        model, val_loader, args.device, args.n_batches
    )

    # Recreate dataloader to reset iterator
    _, val_loader, _ = create_dataloaders(
        data_dir=data_dir,
        tokenizer_name=data_cfg["tokenizer"],
        max_seq_len=data_cfg["max_seq_len"],
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        val_split=data_cfg.get("val_split", 0.20),
    )

    # Method 2: WITH memory accumulation (corrected way)
    new_loss, new_ppl = evaluate_with_accumulated_memory(
        model, val_loader, args.device, args.n_batches
    )

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint.name}")
    print(f"Step: {step}")
    print(f"Training Stage: {stage}")
    print()
    print(f"OLD (buggy - no memory):     Val Loss={old_loss:.4f}, Val PPL={old_ppl:.2f}")
    print(f"NEW (corrected - with mem):  Val Loss={new_loss:.4f}, Val PPL={new_ppl:.2f}")
    print()

    diff_loss = old_loss - new_loss
    diff_ppl = old_ppl - new_ppl
    print(f"Difference: Loss={diff_loss:+.4f}, PPL={diff_ppl:+.2f}")

    if new_ppl < old_ppl:
        improvement = (old_ppl - new_ppl) / old_ppl * 100
        print(f"Corrected validation shows {improvement:.1f}% better perplexity!")

    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Atlas training script.

Usage:
    python scripts/train.py --config configs/atlas_50m.yaml
    python scripts/train.py --config configs/atlas_50m.yaml --device cuda:1

Reference: Behrouz et al. Miras framework
"""

import argparse
import yaml
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.atlas import Atlas, AtlasConfig
from src.training.trainer import AtlasTrainer, TNTTrainer
from src.data.loader import create_dataloaders, create_test_dataloader


def load_config(config_path: Path) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict) -> Atlas:
    """Create Atlas model from config."""
    model_config = AtlasConfig(
        d_model=config["model"]["d_model"],
        n_layers=config["model"]["n_layers"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        vocab_size=config["model"]["vocab_size"],
        max_seq_len=config["model"]["max_seq_len"],
        d_key=config["model"]["d_key"],
        d_value=config["model"]["d_value"],
        momentum_beta=config["model"]["momentum_beta"],
        memory_lr_init=config["model"]["memory_lr_init"],
        learn_memory_lr=config["model"]["learn_memory_lr"],
        retention_local_init=config["model"]["retention_local_init"],
        retention_global_init=config["model"]["retention_global_init"],
        adaptive_retention=config["model"]["adaptive_retention"],
        window_size=config["model"]["window_size"],
        dropout=config["model"]["dropout"],
    )

    model = Atlas(model_config)
    print(f"Created Atlas model with {model.n_params:,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Atlas model")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/atlas_50m.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use synthetic data for testing",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Device
    device_str = args.device or config["hardware"]["device"]
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Check GPU availability
    if device.type == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"GPU: {torch.cuda.get_device_name(device)} ({gpu_mem:.1f} GB)")

    # Create model
    model = create_model(config)

    # Create dataloaders
    if args.test:
        print("Using synthetic data for testing")
        train_loader = create_test_dataloader(
            vocab_size=config["model"]["vocab_size"],
            seq_len=config["data"]["max_seq_len"],
            n_samples=1000,
            batch_size=config["training"]["batch_size"],
        )
        val_loader = None
        tokenizer = None
    else:
        # Real data
        data_dir = Path(__file__).parent.parent / config["data"]["data_dir"]
        print(f"Loading data from: {data_dir}")

        train_loader, val_loader, tokenizer = create_dataloaders(
            data_dir=data_dir,
            tokenizer_name=config["data"]["tokenizer"],
            max_seq_len=config["data"]["max_seq_len"],
            batch_size=config["training"]["batch_size"],
            num_workers=config["data"]["num_workers"],
        )

    # Output directory
    output_dir = Path(__file__).parent.parent / config["output"]["run_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Create trainer
    training_config = config["training"]

    if training_config.get("use_tnt", False):
        print("Using TNT two-stage training")
        trainer = TNTTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            stage1_chunk_size=training_config["stage1_chunk_size"],
            stage1_steps=training_config["stage1_steps"],
            stage2_chunk_size=training_config["stage2_chunk_size"],
            stage2_steps=training_config["stage2_steps"],
            grad_accum_steps=training_config["grad_accum_steps"],
            output_dir=output_dir,
            device=device,
            log_every=training_config["log_every"],
            val_every=training_config["val_every"],
            save_every=training_config["save_every"],
        )
    else:
        print("Using standard training")
        total_steps = training_config.get(
            "total_steps",
            training_config["stage1_steps"] + training_config["stage2_steps"]
        )
        trainer = AtlasTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=training_config["learning_rate"],
            warmup_steps=training_config["warmup_steps"],
            total_steps=total_steps,
            grad_accum_steps=training_config["grad_accum_steps"],
            output_dir=output_dir,
            device=device,
            log_every=training_config["log_every"],
            val_every=training_config["val_every"],
            save_every=training_config["save_every"],
        )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60 + "\n")

    trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Output saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

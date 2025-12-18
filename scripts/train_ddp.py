#!/usr/bin/env python3
"""
Distributed Data Parallel training script for Atlas.

Launch with torchrun:
    torchrun --nproc_per_node=2 scripts/train_ddp.py --config configs/atlas_50m.yaml

Or with a specific master port (if default is in use):
    torchrun --nproc_per_node=2 --master_port=12356 scripts/train_ddp.py --config configs/atlas_50m.yaml

Single GPU fallback:
    python scripts/train_ddp.py --config configs/atlas_50m.yaml --single-gpu

Reference: Behrouz et al. Titans/TNT papers
"""

import argparse
import yaml
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.atlas import Atlas, AtlasConfig
from src.training.ddp_trainer import DDPTrainer, setup_ddp, cleanup_ddp, is_main_process
from src.training.alerts import create_alerts_from_config
from src.data.loader import create_dataloaders, create_test_dataloader


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


def main():
    parser = argparse.ArgumentParser(description="Train Atlas with DDP")
    parser.add_argument("--config", type=Path, default=Path("configs/atlas_50m.yaml"))
    parser.add_argument("--single-gpu", action="store_true", help="Force single GPU mode")
    parser.add_argument("--test", action="store_true", help="Use synthetic data")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # DDP setup
    if args.single_gpu or not torch.cuda.is_available():
        rank = 0
        world_size = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        # Get rank and world_size from torchrun environment
        rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        if world_size > 1:
            setup_ddp(rank, world_size, backend=config["hardware"].get("backend", "nccl"))

        device = torch.device(f"cuda:{rank}")

    is_main = (rank == 0)

    if is_main:
        print("=" * 60)
        print("Atlas DDP Training")
        print("=" * 60)
        print(f"World size: {world_size}")
        print(f"Rank: {rank}")
        print(f"Device: {device}")
        print()

    # Create model
    model = create_model(config)
    if is_main:
        print(f"Model parameters: {model.n_params:,}")
        print(f"Vocab size: {config['model']['vocab_size']}")
        print()

    # Create dataloaders
    train_cfg = config["training"]
    data_cfg = config["data"]

    if args.test:
        if is_main:
            print("Using synthetic data for testing")
        train_loader = create_test_dataloader(
            vocab_size=config["model"]["vocab_size"],
            seq_len=data_cfg["max_seq_len"],
            n_samples=10000,
            batch_size=train_cfg["batch_size"],
        )
        val_loader = create_test_dataloader(
            vocab_size=config["model"]["vocab_size"],
            seq_len=data_cfg["max_seq_len"],
            n_samples=1000,
            batch_size=train_cfg["batch_size"],
        )
    else:
        data_dir = Path(__file__).parent.parent / data_cfg["data_dir"]
        if is_main:
            print(f"Loading data from: {data_dir}")

        train_loader, val_loader, tokenizer = create_dataloaders(
            data_dir=data_dir,
            tokenizer_name=data_cfg["tokenizer"],
            max_seq_len=data_cfg["max_seq_len"],
            batch_size=train_cfg["batch_size"],
            num_workers=data_cfg["num_workers"],
            val_split=data_cfg.get("val_split", 0.20),
        )

    # Wrap with DistributedSampler for DDP
    if world_size > 1 and not args.test:
        # For IterableDataset, we handle distribution differently
        # The dataset itself should handle rank-based sharding
        pass  # Our streaming dataset handles this internally

    # Output directory
    output_dir = Path(__file__).parent.parent / config["output"]["run_dir"]
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(output_dir / "config.yaml", "w") as f:
            yaml.dump(config, f)

    # Calculate effective batch size
    effective_batch = (
        train_cfg["batch_size"] *
        data_cfg["max_seq_len"] *
        train_cfg["grad_accum_steps"] *
        world_size
    )
    if is_main:
        print(f"Effective batch size: {effective_batch:,} tokens")
        print(f"  = {train_cfg['batch_size']} batch × {data_cfg['max_seq_len']} seq × {train_cfg['grad_accum_steps']} accum × {world_size} GPUs")
        print()

    # Create trainer
    trainer = DDPTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        # Optimization
        lr=train_cfg["learning_rate"],
        betas=tuple(train_cfg.get("betas", [0.9, 0.95])),
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_steps"],
        grad_accum_steps=train_cfg["grad_accum_steps"],
        grad_clip=train_cfg["grad_clip"],
        # DDP
        rank=rank,
        world_size=world_size,
        device=device,
        # Mixed precision
        use_amp=config["hardware"].get("mixed_precision", True),
        # TNT
        use_tnt=train_cfg.get("use_tnt", True),
        stage1_chunk_size=train_cfg["stage1_chunk_size"],
        stage1_steps=train_cfg["stage1_steps"],
        stage2_chunk_size=train_cfg["stage2_chunk_size"],
        stage2_steps=train_cfg["stage2_steps"],
        # Logging
        output_dir=output_dir,
        log_every=train_cfg["log_every"],
        val_every=train_cfg["val_every"],
        save_every=train_cfg["save_every"],
        # Early stopping
        val_patience=train_cfg.get("val_patience", 10),
        # Memory reset
        memory_reset_every=train_cfg.get("memory_reset_every", 5000),
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Setup alerts (only on main process)
    if is_main:
        alerts = create_alerts_from_config(config)
        if alerts:
            trainer.set_alerts(alerts)
            print(f"SMS alerts enabled: {alerts.to_address}")
            print(f"  Quiet hours: {alerts.quiet_start.strftime('%H:%M')} - {alerts.quiet_end.strftime('%H:%M')}")
        else:
            print("SMS alerts disabled")
        print()

    # Train
    if is_main:
        print("Starting training...")
        print("=" * 60)

    try:
        trainer.train()
    finally:
        if world_size > 1:
            cleanup_ddp()

    if is_main:
        print("=" * 60)
        print("Training complete!")
        print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()

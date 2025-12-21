#!/usr/bin/env python3
"""
Train Atlas with Episodic Memory.

Usage:
    # 40M local test on A6000
    python scripts/train_episodic.py --config configs/atlas_40m_local_test.yaml

    # 389M cloud training
    python scripts/train_episodic.py --config configs/atlas_389m_episodic.yaml

    # Resume from checkpoint
    python scripts/train_episodic.py --config configs/atlas_40m_local_test.yaml --resume runs/atlas_40m_local/checkpoints/checkpoint_1000.pt
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.atlas_omega import (
    AtlasOmega,
    AtlasOmegaConfig,
    create_atlas_omega_40m,
    create_atlas_omega_389m,
)
from src.training.episodic_trainer import (
    EpisodicDDPTrainer,
    EpisodicConfig,
    TrainerConfig,
)
from training_framework.adapters.atlas_adapter import AtlasMetricsAdapter


class DictDataLoader:
    """Wrapper that converts batches to dict format with input_ids/labels keys."""

    def __init__(self, dl):
        self.dl = dl

    def __iter__(self):
        for batch in self.dl:
            # Handle different batch formats
            if isinstance(batch, dict):
                yield batch
            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                yield {'input_ids': batch[0], 'labels': batch[1]}
            else:
                yield {'input_ids': batch, 'labels': batch}

    def __len__(self):
        return len(self.dl)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load secrets if available
    secrets_path = Path(config_path).parent / 'secrets.yaml'
    if secrets_path.exists():
        print(f"Loading secrets from: {secrets_path}")
        with open(secrets_path) as f:
            secrets = yaml.safe_load(f)

        # Merge telegram secrets into config
        if secrets and 'telegram' in secrets:
            if 'monitoring' not in config:
                config['monitoring'] = {}
            if 'telegram' not in config['monitoring']:
                config['monitoring']['telegram'] = {}
            config['monitoring']['telegram']['bot_token'] = secrets['telegram'].get('bot_token', '')
            config['monitoring']['telegram']['chat_id'] = secrets['telegram'].get('chat_id', '')

    return config


def create_model_from_config(config: dict) -> AtlasOmega:
    """Create model from configuration.

    Uses explicit config values when provided, falling back to factory
    functions only for specific parameter counts that match standard configs.
    """
    model_cfg = config.get('model', {})

    # Check for explicit custom configuration
    d_model = model_cfg.get('d_model', 384)
    n_layers = model_cfg.get('n_layers', 8)

    # Only use factory functions for exact standard configs
    # Otherwise always use explicit config (allows custom 10M models etc)
    if d_model == 384 and n_layers == 8 and model_cfg.get('use_factory', False):
        print("Using 40M model configuration (factory)")
        return create_atlas_omega_40m()
    elif d_model == 1024 and n_layers == 16 and model_cfg.get('use_factory', False):
        print("Using 389M model configuration (factory)")
        return create_atlas_omega_389m()

    # Create from explicit config (default path for custom sizes)
    print(f"Using custom model configuration: d={d_model}, L={n_layers}")
    atlas_config = AtlasOmegaConfig(
        d_model=model_cfg.get('d_model', 384),
        n_layers=model_cfg.get('n_layers', 8),
        n_heads=model_cfg.get('n_heads', 6),
        d_ff=model_cfg.get('d_ff', 1536),
        vocab_size=model_cfg.get('vocab_size', 32000),
        max_seq_len=model_cfg.get('max_seq_len', 2048),
        d_key=model_cfg.get('d_key', model_cfg.get('d_model', 384)),
        d_value=model_cfg.get('d_value', model_cfg.get('d_model', 384)),
        poly_degree=model_cfg.get('poly_degree', 2),
        context_window=model_cfg.get('context_window', 16),
        init_alpha=model_cfg.get('init_alpha', 0.99),
        init_theta=model_cfg.get('init_theta', 0.9),
        init_eta=model_cfg.get('init_eta', 0.1),
        window_size=model_cfg.get('window_size', 256),
        dropout=model_cfg.get('dropout', 0.1),
    )
    return AtlasOmega(atlas_config)


def create_dataloader(config: dict) -> DataLoader:
    """Create training dataloader."""
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    # Check if using synthetic data
    if data_cfg.get('use_synthetic', False):
        print("Using synthetic data for testing")
        return create_synthetic_dataloader(
            vocab_size=data_cfg.get('synthetic_vocab_size', 32000),
            seq_len=data_cfg.get('synthetic_seq_len', 2048),
            batch_size=training_cfg.get('batch_size', 16),
            num_samples=10000,
        )

    # Use real data - check environment variable first (for Docker)
    dataset_path = os.environ.get('DATA_PATH', data_cfg.get('dataset_path'))

    # If DATA_PATH env var is a directory, use it directly
    if dataset_path and os.path.isdir(dataset_path):
        # Check if it's a HuggingFace dataset (has dataset_info.json or state.json)
        hf_markers = ['dataset_info.json', 'state.json', 'data-00000-of-00001.arrow']
        is_hf_dataset = any(
            os.path.exists(os.path.join(dataset_path, m))
            for m in hf_markers
        )

        if is_hf_dataset:
            print(f"Loading HuggingFace dataset from: {dataset_path}")
            return create_hf_dataloader(
                dataset_path=dataset_path,
                batch_size=training_cfg.get('batch_size', 16),
                max_seq_len=data_cfg.get('max_seq_len', 2048),
                tokenizer_name=data_cfg.get('tokenizer', 't5-base'),
            )

        # Check if it contains subdirectories (data root) or is a specific dataset
        subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        if subdirs and any('ingredient' in d or 'common_crawl' in d for d in subdirs):
            # It's a data root, pick a subdirectory
            dataset_path = os.path.join(dataset_path, subdirs[0])
            print(f"Using first subdirectory: {dataset_path}")

    if dataset_path is None:
        print("No dataset_path specified, falling back to synthetic data")
        return create_synthetic_dataloader(
            vocab_size=32000,
            seq_len=data_cfg.get('max_seq_len', 2048),
            batch_size=training_cfg.get('batch_size', 16),
            num_samples=10000,
        )

    # Import data loader for JSONL format
    from src.data.loader import create_dataloaders

    max_seq_len = data_cfg.get('max_seq_len', 2048)
    batch_size = training_cfg.get('batch_size', 16)

    train_loader, _, _ = create_dataloaders(
        data_dir=Path(dataset_path),
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=data_cfg.get('num_workers', 4),
    )

    return DictDataLoader(train_loader)


def create_hf_dataloader(
    dataset_path: str,
    batch_size: int,
    max_seq_len: int,
    tokenizer_name: str = "t5-base",
) -> DataLoader:
    """Create dataloader from HuggingFace dataset saved to disk."""
    from datasets import load_from_disk
    from transformers import AutoTokenizer

    # Load dataset
    dataset = load_from_disk(dataset_path)
    print(f"Loaded HF dataset with {len(dataset)} examples")

    # Load tokenizer from config
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    class HFDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, tokenizer, max_len):
            self.dataset = hf_dataset
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            item = self.dataset[idx]
            # Get text from common fields
            text = item.get('text', item.get('content', str(item)))

            # Tokenize
            encoded = self.tokenizer(
                text,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )

            input_ids = encoded['input_ids'].squeeze(0)
            return {'input_ids': input_ids, 'labels': input_ids.clone()}

    torch_dataset = HFDataset(dataset, tokenizer, max_seq_len)

    return DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


def create_synthetic_dataloader(
    vocab_size: int,
    seq_len: int,
    batch_size: int,
    num_samples: int = 10000,
) -> DataLoader:
    """Create synthetic dataloader for testing."""
    from torch.utils.data import TensorDataset

    input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))
    labels = input_ids.clone()

    dataset = TensorDataset(input_ids, labels)

    base_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    return DictDataLoader(base_loader)


def create_trainer_config(config: dict) -> TrainerConfig:
    """Create trainer config from YAML config."""
    training_cfg = config.get('training', {})
    episodic_cfg = config.get('episodic', {})

    ep_config = EpisodicConfig(
        storage_samples=int(episodic_cfg.get('storage_samples', 10)),
        retrieval_samples=int(episodic_cfg.get('retrieval_samples', 10)),
        phase1_steps=int(episodic_cfg.get('phase1_steps', 15000)),
        phase2_steps=int(episodic_cfg.get('phase2_steps', 25000)),
        phase1_gate_floor=float(episodic_cfg.get('phase1_gate_floor', 0.30)),
        phase2_gate_floor=float(episodic_cfg.get('phase2_gate_floor', 0.10)),
        phase3_gate_floor=float(episodic_cfg.get('phase3_gate_floor', 0.05)),
        storage_gate_target=float(episodic_cfg.get('storage_gate_target', 0.80)),
        retrieval_gate_floor=float(episodic_cfg.get('retrieval_gate_floor', 0.30)),
        retrieval_loss_weight=float(episodic_cfg.get('retrieval_loss_weight', 5.0)),
        storage_loss_weight=float(episodic_cfg.get('storage_loss_weight', 1.0)),
        reset_memory_between_episodes=bool(episodic_cfg.get('reset_memory_between_episodes', False)),
    )

    # Get telegram config
    monitoring_cfg = config.get('monitoring', {})
    telegram_cfg = monitoring_cfg.get('telegram', {})

    # Get grokking config
    grokking_cfg = monitoring_cfg.get('grokking', {})

    return TrainerConfig(
        max_steps=int(training_cfg.get('max_steps', 5000)),
        learning_rate=float(training_cfg.get('learning_rate', 3e-4)),
        weight_decay=float(training_cfg.get('weight_decay', 0.1)),
        warmup_steps=int(training_cfg.get('warmup_steps', 500)),
        grad_clip=float(training_cfg.get('grad_clip', 1.0)),
        batch_size=int(training_cfg.get('batch_size', 16)),
        gradient_accumulation_steps=int(training_cfg.get('gradient_accumulation_steps', 2)),
        log_interval=int(training_cfg.get('log_interval', 25)),
        checkpoint_interval=int(training_cfg.get('checkpoint_interval', 1000)),
        metrics_path=str(training_cfg.get('metrics_path', 'runs/experiment/metrics_stream.jsonl')),
        checkpoint_dir=str(training_cfg.get('checkpoint_dir', 'runs/experiment/checkpoints')),
        device=str(training_cfg.get('device', 'cuda')),
        dtype=str(training_cfg.get('dtype', 'bfloat16')),
        compile_model=bool(training_cfg.get('compile_model', False)),
        use_ddp=bool(training_cfg.get('use_ddp', False)),
        episodic=ep_config,
        telegram_bot_token=str(telegram_cfg.get('bot_token', '')),
        telegram_chat_id=str(telegram_cfg.get('chat_id', '')),
        grokking_enabled=bool(grokking_cfg.get('enabled', False)),
        grokking_interval=int(grokking_cfg.get('metrics_interval', 500)),
    )


def main():
    parser = argparse.ArgumentParser(description="Train Atlas with Episodic Memory")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume')
    parser.add_argument('--test-alert', action='store_true', help='Send test Telegram alert and exit')
    parser.add_argument('--device', type=str, default=None, help='Override device (e.g., cuda:0)')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Apply overrides: config file < env var < CLI argument
    # Check CUDA_DEVICE environment variable first (lower priority)
    if os.environ.get('CUDA_DEVICE'):
        config.setdefault('training', {})['device'] = os.environ['CUDA_DEVICE']
        print(f"Device from env CUDA_DEVICE: {os.environ['CUDA_DEVICE']}")

    # CLI argument has highest priority
    if args.device:
        config.setdefault('training', {})['device'] = args.device
        print(f"Device override from CLI: {args.device}")

    # Test alert mode
    if args.test_alert:
        print("Testing Telegram alert...")
        from training_framework.monitoring.alert_system import AlertSystem, TelegramConfig

        telegram_cfg = config.get('monitoring', {}).get('telegram', {})
        if telegram_cfg.get('enabled', False):
            tg_config = TelegramConfig(
                bot_token=os.environ.get('TELEGRAM_BOT_TOKEN', telegram_cfg.get('bot_token', '')),
                chat_id=os.environ.get('TELEGRAM_CHAT_ID', telegram_cfg.get('chat_id', '')),
            )
            alert_system = AlertSystem(tg_config)
            alert_system.send_alert('INFO', 'Test alert from Atlas training script')
            print("Test alert sent!")
        else:
            print("Telegram not enabled in config")
        return

    # Create output directories
    trainer_config = create_trainer_config(config)
    os.makedirs(os.path.dirname(trainer_config.metrics_path), exist_ok=True)
    os.makedirs(trainer_config.checkpoint_dir, exist_ok=True)

    # Create model
    print("Creating model...")
    model = create_model_from_config(config)
    print(f"Model parameters: {model.n_params:,}")

    # Create dataloader
    print("Creating dataloader...")
    train_loader = create_dataloader(config)

    # Create metrics adapter
    adapter = AtlasMetricsAdapter(track_per_layer=True)

    # Create trainer
    print("Creating trainer...")
    trainer = EpisodicDDPTrainer(
        model=model,
        train_dataloader=train_loader,
        config=trainer_config,
        metrics_adapter=adapter,
    )

    # Resume if specified
    if args.resume:
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Print training info
    print("\n" + "=" * 60)
    print("ATLAS EPISODIC MEMORY TRAINING")
    print("=" * 60)
    print(f"Model:          {model.n_params:,} parameters")
    print(f"Device:         {trainer_config.device}")
    print(f"Max steps:      {trainer_config.max_steps:,}")
    print(f"Batch size:     {trainer_config.batch_size} x {trainer_config.gradient_accumulation_steps}")
    print(f"Episodic:       {trainer_config.episodic.storage_samples}S + {trainer_config.episodic.retrieval_samples}R")
    print(f"Metrics:        {trainer_config.metrics_path}")
    print(f"Checkpoints:    {trainer_config.checkpoint_dir}")
    print("=" * 60 + "\n")

    # Train
    final_stats = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final step:           {final_stats['final_step']}")
    print(f"Total episodes:       {final_stats['total_episodes']}")
    print(f"Best retrieval acc:   {final_stats['best_retrieval_accuracy']:.2%}")
    print(f"Verifier success:     {final_stats['verifier_success_rate']:.2%}")
    print(f"Training time:        {final_stats['training_time_hours']:.2f} hours")
    print("=" * 60)


if __name__ == '__main__':
    main()

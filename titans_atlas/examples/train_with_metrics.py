#!/usr/bin/env python3
"""
Atlas Training Script with Full Weaver Space Metrics.

Trains Atlas with:
- Convergence metrics (loss, perplexity, gradient health)
- Weaver space measurements (memory dynamics, attention geometry, manifold evolution)
- Real-time JSONL metrics output for dashboard integration
- YAML config file support with argparse overrides

Usage:
    # With config file (recommended):
    python train_with_metrics.py --config config.yaml

    # With command-line arguments (for quick testing):
    python train_with_metrics.py --data-path ./data/train.bin --max-steps 10000

    # Dashboard (separate terminal):
    streamlit run dashboard.py --server.port 8050

Paper-standard training:
    LR=1e-4 with 10% warmup (5000 steps), cosine decay to 1e-5
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_atlas.models.atlas import Atlas
from titans_atlas.configs import AtlasConfig, MemoryConfig, AttentionConfig
from titans_atlas.metrics.convergence import ConvergenceMetrics
from titans_atlas.metrics.weaver_space import WeaverSpaceMetrics


class TokenizedDataset(Dataset):
    """
    Dataset for pre-tokenized data stored as memory-mapped numpy array.

    Supports train/val splitting for proper evaluation.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        train_split: float = 0.9,
        is_train: bool = True,
    ):
        """
        Initialize tokenized dataset.

        Args:
            data_path: Path to binary file containing uint16 token IDs.
            seq_length: Sequence length for each sample.
            train_split: Fraction of data for training (default: 0.9).
            is_train: If True, use training split; else validation split.
        """
        self.seq_length = seq_length
        data_file = Path(data_path)

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        # Memory-map the data
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)

        # Split into train/val
        split_idx = int(self.num_tokens * train_split)
        if is_train:
            self.start_idx = 0
            self.end_idx = split_idx
        else:
            self.start_idx = split_idx
            self.end_idx = self.num_tokens

        self.length = (self.end_idx - self.start_idx) // seq_length
        print(f"{'Train' if is_train else 'Val'} dataset: {self.length:,} samples ({self.end_idx - self.start_idx:,} tokens)")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        start = self.start_idx + idx * self.seq_length
        end = start + self.seq_length + 1
        tokens = torch.from_numpy(self.data[start:end].astype('int64'))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset with random tokens for throughput testing.
    """

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        tokens = torch.randint(0, self.vocab_size, (self.seq_length + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if max_steps <= warmup_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        print("PyYAML not installed. Install with: pip install pyyaml")
        sys.exit(1)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge YAML config with command-line argument overrides."""
    # Command-line args override config file values
    if args.data_path:
        config.setdefault('data', {})['tokenized_path'] = args.data_path
    if args.batch_size != 64:  # Non-default value
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.seq_length != 256:
        config.setdefault('training', {})['seq_length'] = args.seq_length
    if args.max_steps != 50000:
        config.setdefault('training', {})['max_steps'] = args.max_steps
    if args.lr != 1e-4:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.warmup_steps != 5000:
        config.setdefault('training', {})['warmup_steps'] = args.warmup_steps
    if args.device:
        config.setdefault('training', {})['device'] = args.device
    if args.checkpoint_dir != "./runs/atlas":
        config['output_dir'] = args.checkpoint_dir
    return config


def get_default_config() -> dict:
    """Return default configuration (paper-standard settings)."""
    return {
        'experiment': 'atlas_training',
        'version': '1.0',
        'hypothesis': 'Train Atlas with paper-standard hyperparameters',
        'model': {
            'd_model': 512,
            'num_layers': 6,
            'context_window': 64,
            'polynomial_degree': 2,
            'vocab_size': 32000,
            'max_seq_len': 1024,
            'ffn_hidden_dim': 2048,
            'dropout': 0.0,
            'init_std': 0.02,
        },
        'memory': {
            'd_key': 64,
            'd_value': 64,
            'num_memory_layers': 2,
            'use_momentum': True,
            'use_forget_gate': True,
            'learnable_lr': True,
            'learnable_momentum': True,
            'learnable_forget': True,
        },
        'attention': {
            'num_heads': 8,
            'd_head': 64,
            'window_size': 512,
            'num_persistent_tokens': 8,
            'use_flash_attention': False,
            'use_rotary_embeddings': True,
            'use_qkv_conv': True,
        },
        'training': {
            'batch_size': 64,
            'seq_length': 256,
            'learning_rate': 1e-4,  # Paper standard
            'min_lr': 1e-5,  # Cosine decay to 10% of peak
            'warmup_steps': 5000,  # 10% warmup
            'max_steps': 50000,
            'weight_decay': 0.1,
            'grad_clip': 1.0,
            'beta1': 0.9,
            'beta2': 0.95,
            'device': 'cuda:0',
            'dtype': 'bfloat16',
        },
        'data': {
            'train_split': 0.9,
        },
        'weaver_metrics': {
            'enabled': True,
            'sample_rate': 100,
        },
        'logging': {
            'log_interval': 10,
            'save_interval': 5000,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Atlas Training with Metrics")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--data-path", type=str, default=None, help="Path to tokenized data (.bin file)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length (default: 256)")
    parser.add_argument("--max-steps", type=int, default=50000, help="Max training steps (default: 50000)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate (default: 1e-4, paper standard)")
    parser.add_argument("--warmup-steps", type=int, default=5000, help="Warmup steps (default: 5000, 10%% of 50k)")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="./runs/atlas", help="Output directory")
    parser.add_argument("--save-interval", type=int, default=5000, help="Save checkpoint every N steps")
    parser.add_argument("--metrics-sample-rate", type=int, default=100, help="Capture weaver metrics every N steps")
    parser.add_argument("--enable-weaver-metrics", action="store_true", default=True, help="Enable weaver space metrics")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda:0, cuda:1, etc)")
    parser.add_argument("--clean-start", action="store_true", help="Remove previous runs in output directory")
    args = parser.parse_args()

    # Load config (YAML or defaults)
    if args.config:
        config = load_config(Path(args.config))
    else:
        config = get_default_config()

    # Merge with command-line overrides
    config = merge_config_with_args(config, args)

    # Extract config sections
    model_cfg = config.get('model', get_default_config()['model'])
    memory_cfg = config.get('memory', get_default_config()['memory'])
    attn_cfg = config.get('attention', get_default_config()['attention'])
    train_cfg = config.get('training', get_default_config()['training'])
    data_cfg = config.get('data', get_default_config()['data'])
    weaver_cfg = config.get('weaver_metrics', get_default_config()['weaver_metrics'])
    logging_cfg = config.get('logging', get_default_config()['logging'])

    # Create run directory
    base_output_dir = config.get('output_dir', args.checkpoint_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Handle clean start
    if args.clean_start:
        import shutil
        parent_dir = Path(base_output_dir)
        if parent_dir.exists():
            for item in parent_dir.iterdir():
                if item.is_dir() and item != run_dir:
                    shutil.rmtree(item)
                    print(f"Removed: {item}")

    # Device setup
    device_str = args.device or train_cfg.get('device', 'cuda:0')
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device(device_str)
    device_idx = device.index if device.index is not None else 0

    print("=" * 70)
    print("Atlas Training with Weaver Space Metrics")
    print("=" * 70)
    print()
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device_idx)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(device_idx).total_memory / 1e9:.1f} GB")
    print(f"Run directory: {run_dir}")
    print()

    # Create Atlas config
    atlas_config = AtlasConfig(
        d_model=model_cfg['d_model'],
        num_layers=model_cfg['num_layers'],
        context_window=model_cfg['context_window'],
        polynomial_degree=model_cfg['polynomial_degree'],
        vocab_size=model_cfg['vocab_size'],
        max_seq_len=model_cfg['max_seq_len'],
        ffn_hidden_dim=model_cfg['ffn_hidden_dim'],
        dropout=model_cfg.get('dropout', 0.0),
        init_std=model_cfg.get('init_std', 0.02),
        memory=MemoryConfig(
            d_model=model_cfg['d_model'],
            d_key=memory_cfg['d_key'],
            d_value=memory_cfg['d_value'],
            num_memory_layers=memory_cfg['num_memory_layers'],
            use_momentum=memory_cfg.get('use_momentum', True),
            use_forget_gate=memory_cfg.get('use_forget_gate', True),
            learnable_lr=memory_cfg.get('learnable_lr', True),
            learnable_momentum=memory_cfg.get('learnable_momentum', True),
            learnable_forget=memory_cfg.get('learnable_forget', True),
        ),
        attention=AttentionConfig(
            d_model=model_cfg['d_model'],
            num_heads=attn_cfg['num_heads'],
            d_head=attn_cfg['d_head'],
            window_size=attn_cfg.get('window_size', 512),
            num_persistent_tokens=attn_cfg.get('num_persistent_tokens', 8),
            use_flash_attention=attn_cfg.get('use_flash_attention', False),
            use_rotary_embeddings=attn_cfg.get('use_rotary_embeddings', True),
            use_qkv_conv=attn_cfg.get('use_qkv_conv', True),
        ),
    )

    # Create model
    print("Creating Atlas model...")
    model = Atlas(atlas_config)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Architecture: {model_cfg['d_model']}D x {model_cfg['num_layers']}L x {attn_cfg['num_heads']}H")
    print(f"Vocab size: {model_cfg['vocab_size']}")
    print()

    # Dataset
    data_path = data_cfg.get('tokenized_path')
    seq_length = train_cfg['seq_length']

    if data_path:
        print(f"Loading tokenized data from {data_path}...")
        dataset = TokenizedDataset(
            data_path,
            seq_length=seq_length,
            train_split=data_cfg.get('train_split', 0.9),
            is_train=True,
        )
    else:
        print("Using synthetic data (random tokens) for throughput testing...")
        dataset = SyntheticDataset(model_cfg['vocab_size'], seq_length, num_samples=100000)

    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Training parameters (convert scientific notation if needed)
    learning_rate = float(train_cfg['learning_rate'])
    min_lr = float(train_cfg.get('min_lr', learning_rate * 0.1))
    beta1 = float(train_cfg.get('beta1', 0.9))
    beta2 = float(train_cfg.get('beta2', 0.95))
    weight_decay = float(train_cfg.get('weight_decay', 0.1))
    grad_clip = float(train_cfg.get('grad_clip', 1.0))
    warmup_steps = int(train_cfg.get('warmup_steps', args.warmup_steps))
    max_steps = int(train_cfg.get('max_steps', args.max_steps))

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )

    # Mixed precision
    dtype_str = train_cfg.get('dtype', args.precision)
    if dtype_str in ("bf16", "bfloat16"):
        dtype = torch.bfloat16
        scaler = GradScaler("cuda", enabled=False)
    elif dtype_str in ("fp16", "float16"):
        dtype = torch.float16
        scaler = GradScaler("cuda", enabled=True)
    else:
        dtype = torch.float32
        scaler = GradScaler("cuda", enabled=False)

    # Initialize metrics
    sample_rate = weaver_cfg.get('sample_rate', args.metrics_sample_rate)

    convergence_metrics = ConvergenceMetrics(
        output_dir=str(metrics_dir / "convergence"),
        patience=max_steps // 10,
    )

    weaver_metrics = WeaverSpaceMetrics(
        output_dir=str(metrics_dir / "weaver"),
        sample_rate=sample_rate,
        enabled=weaver_cfg.get('enabled', args.enable_weaver_metrics),
    )

    # Register hooks for hidden state capture
    weaver_metrics.register_hooks(model)

    # Resume from checkpoint if specified
    start_step = 1
    if args.resume_from:
        checkpoint_path = Path(args.resume_from)
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_step = checkpoint.get("step", 0) + 1
            print(f"  Loaded model state from step {start_step - 1}")
        else:
            print(f"WARNING: Checkpoint not found: {checkpoint_path}")
            print("  Starting from scratch...")

    # Save run config (for dashboard auto-detection)
    run_config = {
        "experiment": config.get('experiment', 'atlas_training'),
        "version": config.get('version', '1.0'),
        "hypothesis": config.get('hypothesis', ''),
        "model_config": asdict(atlas_config),
        "training_config": {
            'batch_size': train_cfg['batch_size'],
            'seq_length': train_cfg['seq_length'],
            'learning_rate': learning_rate,
            'min_lr': min_lr,
            'warmup_steps': warmup_steps,
            'max_steps': max_steps,
        },
        "num_params": num_params,
        "start_time": datetime.now().isoformat(),
        "device": str(device),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    # JSONL metrics file (for dashboard)
    metrics_file = run_dir / "metrics.jsonl"

    log_interval = logging_cfg.get('log_interval', args.log_interval)
    save_interval = logging_cfg.get('save_interval', args.save_interval)

    print(f"Training config:")
    print(f"  Batch size: {train_cfg['batch_size']}")
    print(f"  Sequence length: {train_cfg['seq_length']}")
    print(f"  Tokens per batch: {train_cfg['batch_size'] * train_cfg['seq_length']:,}")
    print(f"  Max steps: {max_steps}")
    print(f"  Learning rate: {learning_rate} -> {min_lr} (cosine decay)")
    print(f"  Warmup steps: {warmup_steps} ({100*warmup_steps/max_steps:.0f}%)")
    print(f"  Weaver metrics: every {sample_rate} steps")
    print()

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    running_loss = 0.0
    tokens_processed = 0
    start_time = time.time()

    # Convergence tracking
    best_convergence_score = 0.0
    convergence_threshold = 0.95
    prev_converging = False

    print("=" * 70)
    if start_step > 1:
        print(f"Resuming training from step {start_step}...")
    else:
        print("Starting training with Weaver Space metrics capture...")
    print("=" * 70)
    print()
    sys.stdout.flush()

    for step in range(start_step, max_steps + 1):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        optimizer.zero_grad()

        with autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

        # Backward pass
        if dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        loss_val = loss.item()
        running_loss += loss_val
        tokens_processed += input_ids.numel()

        # Update convergence metrics
        conv_state = convergence_metrics.update(
            step=step,
            loss=loss_val,
            grad_norm=grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            learning_rate=lr,
            tokens_seen=tokens_processed,
            batch_size=train_cfg['batch_size'],
        )

        # Update weaver space metrics
        weaver_metrics.step(
            step=step,
            epoch=0,
            loss=loss_val,
            memory_state=outputs.get("memory_states", [{}])[0] if outputs.get("memory_states") else None,
        )

        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
            avg_loss = running_loss / log_interval

            # GPU metrics
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
            try:
                gpu_util = torch.cuda.utilization(device_idx)
            except (ModuleNotFoundError, RuntimeError):
                gpu_util = 0

            # Convergence indicators
            conv_indicator = "↓" if conv_state.is_converging else ("→" if conv_state.is_plateaued else "~")
            ppl = conv_state.perplexity

            print(
                f"Step {step:6d}/{max_steps} | "
                f"Loss: {avg_loss:.4f} {conv_indicator} | "
                f"PPL: {ppl:8.2f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"GPU: {gpu_mem:.1f}GB"
            )
            sys.stdout.flush()

            # Write to JSONL for dashboard
            metrics_entry = {
                "iteration": step,
                "timestamp": time.time(),
                "loss": float(avg_loss),
                "perplexity": float(ppl),
                "learning_rate": float(lr),
                "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                "tokens_per_sec": float(tokens_per_sec),
                "gpu_mem_gb": float(gpu_mem),
                "gpu_util": int(gpu_util) if gpu_util else 0,
                "convergence_score": float(conv_state.convergence_score),
                "is_converging": bool(conv_state.is_converging),
            }
            with open(metrics_file, 'a') as f:
                json.dump(metrics_entry, f)
                f.write('\n')

            running_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # Checkpointing
        if step % save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(atlas_config),
                "loss": loss_val,
                "convergence_summary": convergence_metrics.get_summary(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save weaver metrics snapshot
            weaver_metrics.save_checkpoint(step)
            convergence_metrics.save_full_history()

        # Convergence-based checkpoint
        if conv_state.convergence_score > best_convergence_score:
            best_convergence_score = conv_state.convergence_score

            if conv_state.convergence_score >= convergence_threshold:
                checkpoint_path = checkpoints_dir / f"checkpoint_converged_{step:06d}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(atlas_config),
                    "loss": loss_val,
                    "convergence_score": conv_state.convergence_score,
                    "convergence_summary": convergence_metrics.get_summary(),
                }, checkpoint_path)
                print(f"Convergence checkpoint saved (score={conv_state.convergence_score:.3f}): {checkpoint_path}")

        # Track convergence state transitions
        if conv_state.is_converging and not prev_converging:
            print(f"Model started converging at step {step}")
        prev_converging = conv_state.is_converging

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)

    # Final stats
    final_gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"Peak GPU memory: {final_gpu_mem:.2f} GB")

    # Print convergence summary
    summary = convergence_metrics.get_summary()
    print(f"\nConvergence Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Best loss: {summary['best_loss']:.4f} (step {summary['best_step']})")
    print(f"  Final loss: {summary['current_loss']:.4f}")
    print(f"  Loss reduction: {summary['loss_reduction_pct']:.1f}%")
    print(f"  Final perplexity: {summary['current_perplexity']:.2f}")

    # Print weaver space summary
    weaver_summary = weaver_metrics.get_summary()
    if weaver_summary.get("manifold"):
        print(f"\nWeaver Space Summary:")
        print(f"  Mean D_eff preservation: {weaver_summary['manifold'].get('mean_d_eff_preservation', 0):.2f}")
        print(f"  Mean min D_eff: {weaver_summary['manifold'].get('mean_min_d_eff', 0):.1f}")

    # Save final checkpoint
    final_path = checkpoints_dir / "checkpoint_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(atlas_config),
        "convergence_summary": summary,
        "weaver_summary": weaver_summary,
    }, final_path)
    print(f"\nSaved final checkpoint: {final_path}")

    # Save final metrics
    convergence_metrics.save_full_history()
    weaver_metrics.save_checkpoint(step)

    # Cleanup hooks
    weaver_metrics.remove_hooks()

    print(f"\nMetrics saved to: {metrics_dir}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

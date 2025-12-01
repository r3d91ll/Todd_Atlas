#!/usr/bin/env python3
"""
Atlas 36M Training with Full Weaver Space Metrics.

Training script that captures:
- Convergence metrics (loss, perplexity, gradient health)
- Weaver space measurements (memory dynamics, attention geometry, manifold evolution)
- Real-time metrics output for dashboard integration

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_with_metrics.py --data-path ./data/

    # With dashboard (separate terminal):
    streamlit run dashboard.py --server.port 8050

    # Full training run to convergence:
    CUDA_VISIBLE_DEVICES=0 python train_with_metrics.py \
        --data-path ./data/ \
        --max-steps 10000 \
        --batch-size 64 \
        --seq-length 512 \
        --save-interval 1000 \
        --metrics-sample-rate 50
"""

import os
import sys
import math
import time
import argparse
from pathlib import Path
from dataclasses import asdict
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_atlas.models.atlas import Atlas
from titans_atlas.configs import atlas_36m, TrainingConfig
from titans_atlas.metrics.convergence import ConvergenceMetrics, GradientMonitor
from titans_atlas.metrics.weaver_space import WeaverSpaceMetrics


class TokenizedDataset(Dataset):
    """
    Dataset for pre-tokenized data stored as memory-mapped numpy array.

    Reads from a binary file (train.bin) containing uint16 token IDs.
    """

    def __init__(self, data_path: str, seq_length: int):
        """
        Initialize tokenized dataset.

        Args:
            data_path: Path to directory containing train.bin file.
            seq_length: Sequence length for each sample.
        """
        self.seq_length = seq_length
        self.data_path = Path(data_path)

        # Try to load tokenized data
        data_file = self.data_path / "train.bin"
        if data_file.exists():
            import numpy as np
            self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
            self.num_tokens = len(self.data)
            print(f"Loaded {self.num_tokens:,} tokens from {data_file}")
        else:
            raise FileNotFoundError(f"Data file not found: {data_file}")

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_tokens // self.seq_length

    def __getitem__(self, idx: int):
        """
        Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dict with input_ids and labels tensors.
        """
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        tokens = torch.from_numpy(self.data[start:end].astype('int64'))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset with random tokens for throughput testing.

    Generates random token sequences on-the-fly without requiring data files.
    """

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        """
        Initialize synthetic dataset.

        Args:
            vocab_size: Size of vocabulary for random token generation.
            seq_length: Sequence length for each sample.
            num_samples: Number of samples in the dataset.
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return self.num_samples

    def __getitem__(self, idx: int):
        """
        Get a random sample by index.

        Args:
            idx: Sample index (ignored, generates random data).

        Returns:
            Dict with random input_ids and labels tensors.
        """
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
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    parser = argparse.ArgumentParser(description="Atlas 36M Training with Metrics")
    parser.add_argument("--data-path", type=str, default=None, help="Path to tokenized data")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-steps", type=int, default=10000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="./runs/atlas_36m", help="Output directory")
    parser.add_argument("--save-interval", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--metrics-sample-rate", type=int, default=50, help="Capture metrics every N steps")
    parser.add_argument("--enable-weaver-metrics", action="store_true", default=True, help="Enable weaver space metrics")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    # Setup output directory
    run_dir = Path(args.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Create model
    print("Creating Atlas 36M model...")
    config = atlas_36m()
    config.max_seq_len = args.seq_length

    model = Atlas(config)
    model = model.to(device)

    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Config: d_model={config.d_model}, layers={config.num_layers}, heads={config.attention.num_heads}")
    print()

    # Dataset
    if args.data_path:
        print(f"Loading tokenized data from {args.data_path}...")
        dataset = TokenizedDataset(args.data_path, args.seq_length)
    else:
        print("Using synthetic data (random tokens) for throughput testing...")
        dataset = SyntheticDataset(config.vocab_size, args.seq_length, num_samples=100000)

    num_workers = 0 if args.data_path else 4
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # Mixed precision
    if args.precision == "fp16":
        dtype = torch.float16
        scaler = GradScaler("cuda", enabled=True)
    elif args.precision == "bf16":
        dtype = torch.bfloat16
        scaler = GradScaler("cuda", enabled=False)
    else:
        dtype = torch.float32
        scaler = GradScaler("cuda", enabled=False)

    # Initialize metrics
    convergence_metrics = ConvergenceMetrics(
        output_dir=str(metrics_dir / "convergence"),
        patience=args.max_steps // 10,  # 10% of total steps
    )

    weaver_metrics = WeaverSpaceMetrics(
        output_dir=str(metrics_dir / "weaver"),
        sample_rate=args.metrics_sample_rate,
        enabled=args.enable_weaver_metrics,
    )

    # Note: GradientMonitor available for detailed per-layer analysis if needed
    # grad_monitor = GradientMonitor(model)

    # Register hooks for hidden state capture
    weaver_metrics.register_hooks(model)

    # Resume from checkpoint if specified
    start_step = 1
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint.get("step", 0) + 1
        print(f"Resumed from step {start_step - 1}")

    # Save run config
    run_config = {
        "model_config": asdict(config),
        "training_config": vars(args),
        "num_params": num_params,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"Training with {args.precision} precision")
    print(f"Batch size: {args.batch_size}, Seq length: {args.seq_length}")
    print(f"Tokens per batch: {args.batch_size * args.seq_length:,}")
    print(f"Output directory: {run_dir}")
    print()

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    running_loss = 0.0
    tokens_processed = 0
    start_time = time.time()

    print("=" * 70)
    print("Starting training with metrics collection...")
    print("=" * 70)
    sys.stdout.flush()

    for step in range(start_step, args.max_steps + 1):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr, args.lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward pass
        optimizer.zero_grad()

        with autocast("cuda", dtype=dtype, enabled=(args.precision != "fp32")):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

        # Backward pass
        if args.precision == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            batch_size=args.batch_size,
        )

        # Update weaver space metrics (captures based on sample_rate)
        weaver_metrics.step(
            step=step,
            epoch=0,
            loss=loss_val,
            memory_state=outputs.get("memory_states", [{}])[0] if outputs.get("memory_states") else None,
        )

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            avg_loss = running_loss / args.log_interval

            gpu_mem = torch.cuda.max_memory_allocated() / 1e9

            # Convergence indicators
            conv_indicator = "↓" if conv_state.is_converging else ("→" if conv_state.is_plateaued else "~")
            ppl = conv_state.perplexity

            print(
                f"Step {step:6d}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} {conv_indicator} | "
                f"PPL: {ppl:8.2f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"GPU: {gpu_mem:.1f}GB | "
                f"Conv: {conv_state.convergence_score:.2f}"
            )
            sys.stdout.flush()

            # Write metrics to JSONL for dashboard
            convergence_metrics.log_to_jsonl()
            weaver_metrics.log_to_jsonl()

            running_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # Checkpointing
        if step % args.save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "loss": loss_val,
                "convergence_summary": convergence_metrics.get_summary(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

            # Save weaver metrics snapshot
            weaver_metrics.save_checkpoint(step)

            # Save convergence history
            convergence_metrics.save_full_history()

        # Early stopping check
        should_stop, reason = convergence_metrics.should_stop_early(
            convergence_threshold=0.98,  # High threshold
        )
        if should_stop and step > args.warmup_steps * 2:
            print(f"\nEarly stopping: {reason}")
            break

    print()
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)

    # Final stats
    final_gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory: {final_gpu_mem:.2f} GB")

    # Print convergence summary
    summary = convergence_metrics.get_summary()
    print(f"\nConvergence Summary:")
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Best loss: {summary['best_loss']:.4f} (step {summary['best_step']})")
    print(f"  Final loss: {summary['current_loss']:.4f}")
    print(f"  Loss reduction: {summary['loss_reduction_pct']:.1f}%")
    print(f"  Final perplexity: {summary['current_perplexity']:.2f}")
    print(f"  Convergence score: {summary['convergence_score']:.2f}")

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
        "config": asdict(config),
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
    print(f"Checkpoints saved to: {checkpoints_dir}")


if __name__ == "__main__":
    main()

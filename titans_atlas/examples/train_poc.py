#!/usr/bin/env python3
"""
Proof-of-Concept Training Script for Atlas 36M model.

Single-GPU training on GPU0 to validate the implementation works.
Uses synthetic data (random tokens) to test throughput before real data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python train_poc.py

    # Or with real data:
    CUDA_VISIBLE_DEVICES=0 python train_poc.py --data-path /path/to/data
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


class SyntheticDataset(Dataset):
    """Synthetic dataset with random tokens for throughput testing."""

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int = 10000):
        """
        Initialize the dataset.
        
        Parameters:
            vocab_size (int): Number of distinct token IDs in the vocabulary.
            seq_length (int): Number of input tokens per sample (each example will use a contiguous sequence of this length).
            num_samples (int): Total number of samples the dataset exposes (default 10000).
        """
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self) -> int:
        """
        Return the number of samples available in the dataset.
        
        Returns:
            int: The total number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx: int):
        # Generate random tokens
        """
        Return a randomly generated token example containing input tokens and next-token labels.
        
        Parameters:
        	idx (int): Index of the sample (ignored; kept for Dataset compatibility).
        
        Returns:
        	dict: A mapping with:
        		- "input_ids" (torch.LongTensor): 1-D tensor of length `self.seq_length` with random token ids.
        		- "labels" (torch.LongTensor): 1-D tensor of length `self.seq_length` containing the next-token targets (input_ids shifted left by one).
        """
        tokens = torch.randint(0, self.vocab_size, (self.seq_length + 1,))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized data stored as memory-mapped numpy array."""

    def __init__(self, data_path: str, seq_length: int):
        """
        Initialize the dataset by memory-mapping a `train.bin` file of token IDs and storing sequence configuration.
        
        Parameters:
            data_path (str): Path to a directory containing a `train.bin` file of token IDs.
            seq_length (int): Sequence length (number of tokens per example) used by the dataset.
        
        Notes:
            - Expects `train.bin` to be a contiguous array of unsigned 16-bit token IDs; the file is loaded via numpy.memmap and assigned to `self.data`.
            - Sets `self.num_tokens` to the total number of tokens found in `train.bin` and prints the loaded token count.
        
        Raises:
            FileNotFoundError: If `train.bin` is not present at `data_path`.
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
        """
        Compute the number of token sequences available for sampling.
        
        Returns:
            int: Number of sequences (total tokens divided by sequence length, floored).
        """
        return self.num_tokens // self.seq_length

    def __getitem__(self, idx: int):
        """
        Retrieve the input-target token pair for a given dataset index.
        
        Parameters:
        	idx (int): Sequence index; selects a contiguous block of tokens starting at `idx * seq_length`.
        
        Returns:
        	dict: A mapping with:
        		- "input_ids" (torch.Tensor): 1-D `torch.int64` tensor of length `seq_length` containing the input tokens.
        		- "labels" (torch.Tensor): 1-D `torch.int64` tensor of length `seq_length` containing the target tokens (input shifted left by one).
        """
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        tokens = torch.from_numpy(self.data[start:end].astype('int64'))
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """
    Compute the learning rate using a linear warmup followed by cosine decay.
    
    Returns:
        The learning rate for the given `step` as a `float`. During the first `warmup_steps` it linearly increases from 0 to `max_lr`; after warmup it follows a cosine decay from `max_lr` down toward `min_lr`, and does not fall below `min_lr`.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def count_parameters(model: nn.Module) -> int:
    """
    Compute the number of trainable parameters in a model.
    
    Parameters:
        model (nn.Module): The model to inspect.
    
    Returns:
        int: Total number of parameters with `requires_grad` set to True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    """
    Run a single-GPU proof-of-concept training run for the Atlas 36M model driven by command-line arguments.
    
    Parses CLI options to configure model, data source (synthetic tokens by default or pre-tokenized memory-mapped data), optimizer, precision (fp32/fp16/bf16), and training hyperparameters; builds the model and dataloader, runs the training loop with per-step learning-rate scheduling, mixed-precision/autocast handling, gradient clipping, periodic logging, and periodic checkpoint saves to the specified checkpoint directory, then writes a final checkpoint and prints peak GPU memory before exiting.
    """
    parser = argparse.ArgumentParser(description="Atlas 36M Proof-of-Concept Training")
    parser.add_argument("--data-path", type=str, default=None, help="Path to tokenized data (if None, uses synthetic)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--log-interval", type=int, default=10, help="Log every N steps")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_poc", help="Checkpoint directory")
    parser.add_argument("--save-interval", type=int, default=500, help="Save checkpoint every N steps")
    args = parser.parse_args()

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

    # Use num_workers=0 for memmap datasets to avoid multiprocessing issues
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
        scaler = GradScaler("cuda", enabled=False)  # BF16 doesn't need scaling
    else:
        dtype = torch.float32
        scaler = GradScaler("cuda", enabled=False)

    print(f"Training with {args.precision} precision")
    print(f"Batch size: {args.batch_size}, Seq length: {args.seq_length}")
    print(f"Tokens per batch: {args.batch_size * args.seq_length:,}")
    print()

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    running_loss = 0.0
    tokens_processed = 0
    start_time = time.time()

    print("="*60)
    print("Starting training...")
    print("="*60)
    sys.stdout.flush()

    for step in range(1, args.max_steps + 1):
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

        running_loss += loss.item()
        tokens_processed += input_ids.numel()

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed
            avg_loss = running_loss / args.log_interval

            gpu_mem = torch.cuda.max_memory_allocated() / 1e9

            print(
                f"Step {step:5d}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr:.2e} | "
                f"Tok/s: {tokens_per_sec:,.0f} | "
                f"GPU Mem: {gpu_mem:.1f}GB"
            )

            running_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # Checkpointing
        if step % args.save_interval == 0:
            checkpoint_dir = Path(args.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": asdict(config),
                "loss": loss.item(),
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    print()
    print("="*60)
    print("Training complete!")
    print("="*60)

    # Final stats
    final_gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"Peak GPU memory: {final_gpu_mem:.2f} GB")

    # Save final checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_path = checkpoint_dir / "checkpoint_final.pt"
    torch.save({
        "step": args.max_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
    }, final_path)
    print(f"Saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
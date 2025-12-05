#!/usr/bin/env python3
"""
Training script for Atlas models with mixed precision and distributed training support.

Features:
- FP16/BF16 mixed precision via torch.cuda.amp
- Distributed Data Parallel (DDP) for multi-GPU training
- Gradient accumulation for effective large batch sizes
- Cosine learning rate schedule with warmup
- Gradient clipping
- Checkpointing and resume capability
- WandB logging (optional)

Usage:
    # Single GPU
    python train_atlas.py --config config.yaml

    # Multi-GPU (2 GPUs)
    torchrun --nproc_per_node=2 train_atlas.py --config config.yaml

    # With specific GPU assignment
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_atlas.py
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torch.distributed as dist

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_atlas.models.atlas import Atlas
from titans_atlas.configs import AtlasConfig, TrainingConfig, atlas_500m


class TextDataset(Dataset):
    """Simple text dataset for pre-training.

    Expects pre-tokenized data as memory-mapped numpy arrays.
    For real training, use HuggingFace datasets or similar.
    """

    def __init__(
        self,
        data_path: str,
        seq_length: int,
        split: str = "train",
    ):
        """
        Initialize the dataset for pretraining by loading a memory-mapped token file or falling back to generated test data.
        
        Parameters:
            data_path (str): Path to the directory containing pre-tokenized binary files.
            seq_length (int): Sequence length (number of tokens per example).
            split (str): Dataset split name; expects a file named `<split>.bin` containing uint16 token values.
        
        Notes:
            - If `<split>.bin` exists, it is memory-mapped as a numpy array of dtype `uint16` and exposed as `self.data`; `self.num_tokens` is set to the file's token count.
            - If the file is missing, `self.data` is set to `None` and `self.num_tokens` is set to 1_000_000 for testing purposes.
        """
        self.seq_length = seq_length
        self.data_path = Path(data_path)

        # Load pre-tokenized data (expects .bin file with uint16 tokens)
        data_file = self.data_path / f"{split}.bin"
        if data_file.exists():
            import numpy as np
            self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
            self.num_tokens = len(self.data)
        else:
            # Fallback: generate random data for testing
            print(f"Warning: {data_file} not found. Using random data for testing.")
            self.data = None
            self.num_tokens = 1_000_000  # 1M tokens for testing

    def __len__(self) -> int:
        """
        Number of sequences available in the dataset.
        
        Returns:
            int: The number of sequences computed as total token count divided by sequence length.
        """
        return self.num_tokens // self.seq_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieve a token sequence pair (inputs and next-token labels) for the given sequence index.
        
        Parameters:
            idx (int): Zero-based sequence index; the sequence spans tokens [idx * seq_length : idx * seq_length + seq_length + 1].
        
        Returns:
            dict: A dictionary with:
                - "input_ids" (torch.Tensor): Token sequence of length `seq_length` representing model inputs.
                - "labels" (torch.Tensor): Token sequence of length `seq_length` representing next-token targets (inputs shifted left by one).
            Both tensors are integer tensors (dtype int64). If no memmapped data is available, tokens are randomly sampled in range [0, 50257).
        """
        start = idx * self.seq_length
        end = start + self.seq_length + 1  # +1 for labels shift

        if self.data is not None:
            tokens = torch.from_numpy(self.data[start:end].astype('int64'))
        else:
            # Random tokens for testing
            tokens = torch.randint(0, 50257, (self.seq_length + 1,))

        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


def get_lr(step: int, config: TrainingConfig) -> float:
    """
    Compute the learning rate for a given training step using linear warmup followed by cosine decay to a minimum learning rate.
    
    Parameters:
        step (int): Current training step index (0-based).
        config (TrainingConfig): Configuration providing `warmup_steps`, `max_steps`, `learning_rate`, and `min_lr`.
    
    Returns:
        float: Learning rate for the given step. Rises linearly from 0 to `learning_rate` over `warmup_steps`, then follows a cosine decay from `learning_rate` to `min_lr` between `warmup_steps` and `max_steps` (clamped at `min_lr` after `max_steps`).
    """
    if step < config.warmup_steps:
        # Linear warmup
        return config.learning_rate * step / config.warmup_steps

    # Cosine decay
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def setup_distributed() -> tuple[int, int, int]:
    """
    Set up the PyTorch distributed training environment for the current process.
    
    Returns:
        rank (int): Global rank of this process.
        local_rank (int): Local GPU index for this process on the current node.
        world_size (int): Total number of processes participating in the job.
    """
    if "RANK" in os.environ:
        # Launched with torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
    else:
        # Single GPU
        rank = 0
        local_rank = 0
        world_size = 1

    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def count_parameters(model: nn.Module) -> int:
    """
    Compute the total number of trainable parameters in a PyTorch module.
    
    Returns:
        int: Total count of parameters where `requires_grad` is True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    step: int,
    config: TrainingConfig,
    model_config: AtlasConfig,
    metrics: Dict[str, float],
    path: Path,
):
    """
    Save the current training state to a checkpoint file.
    
    Parameters:
        model (nn.Module): Model to save; may be a DistributedDataParallel-wrapped module.
        optimizer (torch.optim.Optimizer): Optimizer whose state will be stored.
        scaler (GradScaler): GradScaler for mixed-precision training whose state will be stored.
        step (int): Current training step number to record in the checkpoint.
        config (TrainingConfig): Training configuration to include in the checkpoint.
        model_config (AtlasConfig): Model configuration to include in the checkpoint.
        metrics (Dict[str, float]): Latest training metrics to include in the checkpoint (e.g., loss, throughput).
        path (Path): Filesystem path where the checkpoint file will be written.
    
    The checkpoint file contains the following keys: "step", "model_state_dict", "optimizer_state_dict",
    "scaler_state_dict", "training_config", "model_config", and "metrics".
    """
    # Get raw model from DDP wrapper
    raw_model = model.module if isinstance(model, DDP) else model

    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "training_config": asdict(config),
        "model_config": asdict(model_config),
        "metrics": metrics,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, Any]:
    """
    Load a training checkpoint from disk and restore model, optimizer, and scaler states.
    
    Parameters:
    	path (Path): Filesystem path to the checkpoint file.
    	model (nn.Module): Model to restore; if wrapped in DistributedDataParallel, the underlying module will be loaded.
    	optimizer (Optional[torch.optim.Optimizer]): Optimizer to restore state into, if provided.
    	scaler (Optional[GradScaler]): GradScaler to restore state into, if provided.
    
    Returns:
    	checkpoint (Dict[str, Any]): The loaded checkpoint dictionary (contains at least `model_state_dict` and may include `optimizer_state_dict`, `scaler_state_dict`, training/config metadata, and metrics).
    """
    checkpoint = torch.load(path, map_location="cpu")

    # Get raw model from DDP wrapper
    raw_model = model.module if isinstance(model, DDP) else model
    raw_model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint


def train(
    model_config: AtlasConfig,
    train_config: TrainingConfig,
    resume_from: Optional[str] = None,
):
    """
    Run the end-to-end training loop for an Atlas model, including distributed/mixed-precision support, logging, and checkpointing.
    
    This function trains the provided Atlas model according to the settings in train_config: it configures distributed training and device placement, prepares the optimizer with selective weight decay, enables mixed precision (FP16/BF16/FP32) and gradient scaling, loads the training dataset and dataloader (with optional DistributedSampler), supports gradient accumulation and gradient clipping, updates a cosine-with-warmup learning rate schedule, periodically saves checkpoints (including a persistent "latest" and a final checkpoint), and optionally logs metrics to Weights & Biases. The process will resume from a checkpoint if resume_from is a valid path. Training runs until train_config.max_steps and performs distributed cleanup on completion.
    
    Parameters:
        model_config (AtlasConfig): Configuration for constructing the Atlas model.
        train_config (TrainingConfig): Training hyperparameters and runtime settings (data paths, batch sizes, learning rate schedule, precision, checkpointing, logging, etc.). The function will update train_config.world_size with the detected world size.
        resume_from (Optional[str]): Path to a checkpoint file to resume training from; if None or the path does not exist, training starts from step 0.
    """

    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Update world size in config
    train_config.world_size = world_size

    if is_main:
        print(f"Training Atlas model")
        print(f"  World size: {world_size}")
        print(f"  Precision: {train_config.precision}")
        print(f"  Effective batch size: {train_config.effective_batch_size}")
        print(f"  Tokens per step: {train_config.tokens_per_step:,}")

    # Set seed for reproducibility
    torch.manual_seed(train_config.seed + rank)
    torch.cuda.manual_seed(train_config.seed + rank)

    # Create model
    model = Atlas(model_config)
    model = model.to(device)

    if is_main:
        num_params = count_parameters(model)
        print(f"  Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Wrap in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer
    # Separate weight decay for different parameter groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "embedding" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": train_config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=train_config.learning_rate, betas=(train_config.beta1, train_config.beta2))

    # Mixed precision setup
    if train_config.precision == "fp16":
        dtype = torch.float16
    elif train_config.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    scaler = GradScaler("cuda", enabled=(train_config.precision == "fp16"))

    # Dataset and dataloader
    train_dataset = TextDataset(
        train_config.data_path,
        train_config.seq_length,
        split="train",
    )

    if world_size > 1:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    else:
        sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=train_config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Resume from checkpoint
    start_step = 0
    if resume_from:
        checkpoint_path = Path(resume_from)
        if checkpoint_path.exists():
            if is_main:
                print(f"Resuming from {checkpoint_path}")
            checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scaler)
            start_step = checkpoint["step"] + 1

    # WandB logging
    if is_main and train_config.wandb_project:
        try:
            import wandb
            wandb.init(
                project=train_config.wandb_project,
                name=train_config.wandb_run_name,
                config={
                    "model": asdict(model_config),
                    "training": asdict(train_config),
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            train_config.wandb_project = None

    # Training loop
    model.train()
    optimizer.zero_grad()

    data_iter = iter(train_loader)
    step = start_step
    running_loss = 0.0
    tokens_processed = 0
    start_time = time.time()

    if is_main:
        print(f"\nStarting training from step {start_step}")
        print("-" * 60)

    while step < train_config.max_steps:
        # Update learning rate
        lr = get_lr(step, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation
        for micro_step in range(train_config.gradient_accumulation_steps):
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                if sampler is not None:
                    sampler.set_epoch(step)
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with mixed precision
            with autocast(device_type="cuda", dtype=dtype, enabled=train_config.use_amp):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                loss = loss / train_config.gradient_accumulation_steps

            # Backward pass
            scaler.scale(loss).backward()

            running_loss += loss.item()
            tokens_processed += input_ids.numel()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.grad_clip
        )

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1

        # Logging
        if step % train_config.log_interval == 0 and is_main:
            elapsed = time.time() - start_time
            tokens_per_sec = tokens_processed / elapsed

            metrics = {
                "step": step,
                "loss": running_loss / train_config.log_interval,
                "lr": lr,
                "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                "tokens_per_sec": tokens_per_sec,
            }

            print(
                f"Step {step:6d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"LR: {lr:.2e} | "
                f"Grad: {metrics['grad_norm']:.2f} | "
                f"Tok/s: {tokens_per_sec:,.0f}"
            )

            if train_config.wandb_project:
                import wandb
                wandb.log(metrics)

            running_loss = 0.0
            tokens_processed = 0
            start_time = time.time()

        # Checkpointing
        if step % train_config.save_interval == 0 and is_main:
            checkpoint_dir = Path(train_config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f"checkpoint_{step:07d}.pt"
            save_checkpoint(
                model, optimizer, scaler, step,
                train_config, model_config,
                {"loss": running_loss},
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

            # Also save latest
            latest_path = checkpoint_dir / "checkpoint_latest.pt"
            save_checkpoint(
                model, optimizer, scaler, step,
                train_config, model_config,
                {"loss": running_loss},
                latest_path,
            )

    # Final checkpoint
    if is_main:
        checkpoint_dir = Path(train_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_path = checkpoint_dir / "checkpoint_final.pt"
        save_checkpoint(
            model, optimizer, scaler, step,
            train_config, model_config,
            {"loss": running_loss},
            final_path,
        )
        print(f"\nTraining complete! Final checkpoint: {final_path}")

    cleanup_distributed()


def main():
    """
    Parse command-line arguments, build the chosen model and training configurations, and launch training.
    
    Accepts flags for model size, precision, micro-batch size, gradient accumulation, sequence length, maximum steps, learning rate, checkpoint directory, data path, optional WandB project, and an optional checkpoint to resume from. Selects the corresponding Atlas model preset, constructs a TrainingConfig from the provided arguments, and calls train(...) with the assembled configurations.
    """
    parser = argparse.ArgumentParser(description="Train Atlas model")
    parser.add_argument(
        "--model-size",
        type=str,
        default="500m",
        choices=["small", "medium", "large", "500m"],
        help="Model size preset",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Training precision",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Micro batch size per GPU",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100000,
        help="Maximum training steps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=6e-4,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to training data",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="WandB project name (optional)",
    )

    args = parser.parse_args()

    # Model config
    if args.model_size == "500m":
        model_config = atlas_500m()
    elif args.model_size == "small":
        from titans_atlas.configs import atlas_small
        model_config = atlas_small()
    elif args.model_size == "medium":
        from titans_atlas.configs import atlas_medium
        model_config = atlas_medium()
    else:
        from titans_atlas.configs import atlas_large
        model_config = atlas_large()

    # Training config
    train_config = TrainingConfig(
        precision=args.precision,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        seq_length=args.seq_length,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        data_path=args.data_path,
        wandb_project=args.wandb_project,
    )

    train(model_config, train_config, resume_from=args.resume)


if __name__ == "__main__":
    main()
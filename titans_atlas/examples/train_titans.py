#!/usr/bin/env python3
"""
Example training script for Titans language model.

This demonstrates how to train a Titans model on a language modeling task.
Supports all three variants: MAC, MAG, and MAL.

Usage:
    python train_titans.py --variant MAG --epochs 10
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from titans_atlas.models.titans import TitansLM
from titans_atlas.configs import TitansConfig, MemoryConfig, AttentionConfig


class RandomTextDataset(Dataset):
    """Simple random dataset for demonstration."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": tokens, "labels": tokens}


def create_config(
    variant: str = "MAG",
    d_model: int = 256,
    num_layers: int = 4,
    vocab_size: int = 1000,
    max_seq_len: int = 512,
) -> TitansConfig:
    """Create a small config for testing."""
    return TitansConfig(
        d_model=d_model,
        num_layers=num_layers,
        variant=variant,
        chunk_size=64,
        memory=MemoryConfig(
            d_model=d_model,
            d_key=32,
            d_value=32,
            num_memory_layers=2,
            use_momentum=True,
            use_forget_gate=True,
        ),
        attention=AttentionConfig(
            d_model=d_model,
            num_heads=4,
            d_head=32,
            window_size=64,
            num_persistent_tokens=4,
            use_flash_attention=False,  # Disable for compatibility
        ),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=0.1,
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": loss.item()})

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Titans model")
    parser.add_argument("--variant", type=str, default="MAG",
                       choices=["MAC", "MAG", "MAL"],
                       help="Architecture variant")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Sequence length")
    parser.add_argument("--d_model", type=int, default=256,
                       help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of layers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--save_path", type=str, default=None,
                       help="Path to save model")
    args = parser.parse_args()

    print(f"Training Titans {args.variant} on {args.device}")

    # Create config and model
    config = create_config(
        variant=args.variant,
        d_model=args.d_model,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
    )
    model = TitansLM(config, variant=args.variant)
    model = model.to(args.device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create datasets
    train_dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        num_samples=1000,
    )
    val_dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_len=args.seq_len,
        num_samples=100,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.1,
    )

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, args.device, epoch)
        val_loss = evaluate(model, val_loader, args.device)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, ppl={math.exp(val_loss):.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if args.save_path:
                torch.save(model.state_dict(), args.save_path)
                print(f"Saved best model to {args.save_path}")

    # Test generation
    print("\nTesting generation...")
    model.eval()
    prompt = torch.randint(0, config.vocab_size, (1, 10)).to(args.device)
    generated = model.generate(prompt, max_new_tokens=20, temperature=0.8, top_k=50)
    print(f"Generated tokens: {generated[0].tolist()}")


if __name__ == "__main__":
    main()

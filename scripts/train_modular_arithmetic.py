#!/usr/bin/env python3
"""
Train Atlas on Modular Arithmetic for Grokking Detection.

This script trains Atlas on a deterministic task (modular arithmetic)
where retrieval accuracy is meaningful and grokking can be properly detected.

Task: Predict (a + b) mod p given "a + b ="
- Each input has exactly ONE correct answer
- Validation accuracy jumping from low to high = grokking detected

Usage:
    python scripts/train_modular_arithmetic.py --config configs/atlas_modular_arithmetic.yaml

Expected grokking behavior:
1. Train accuracy reaches ~100% quickly (memorization)
2. Val accuracy stays low for many steps
3. Suddenly, val accuracy jumps to ~100% (grokking!)
"""

import os
import sys
import yaml
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
from src.data.modular_arithmetic import (
    ModularArithmeticDataset,
    create_modular_arithmetic_dataloaders,
)
from src.training.stablemax import StableCrossEntropyLoss
from src.training.orthogonal_grad import apply_orthogonal_projection
from training_framework.monitoring.grokking_metrics import (
    GrokkingDetector,
    GrokkingConfig,
)


@dataclass
class TrainingState:
    """Track training progress."""
    step: int = 0
    epoch: int = 0
    best_val_acc: float = 0.0
    grokking_detected: bool = False
    grokking_step: Optional[int] = None


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config: dict, vocab_size: int) -> AtlasOmega:
    """Create Atlas model from config."""
    model_cfg = config.get('model', {})

    atlas_config = AtlasOmegaConfig(
        d_model=model_cfg.get('d_model', 128),
        n_layers=model_cfg.get('n_layers', 2),
        n_heads=model_cfg.get('n_heads', 4),
        d_ff=model_cfg.get('d_ff', 512),
        vocab_size=vocab_size,  # Use actual vocab size from dataset
        max_seq_len=model_cfg.get('max_seq_len', 6),
        d_key=model_cfg.get('d_key', 128),
        d_value=model_cfg.get('d_value', 128),
        poly_degree=model_cfg.get('poly_degree', 2),
        context_window=model_cfg.get('context_window', 4),
        init_alpha=model_cfg.get('init_alpha', 0.99),
        init_theta=model_cfg.get('init_theta', 0.9),
        init_eta=model_cfg.get('init_eta', 0.1),
        window_size=model_cfg.get('window_size', 6),
        dropout=model_cfg.get('dropout', 0.1),
    )

    model = AtlasOmega(atlas_config)
    print(f"Created Atlas model: {model.n_params:,} parameters")

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    For modular arithmetic, we measure:
    - Accuracy: fraction of correct answers (exact match)
    - Loss: cross-entropy loss
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        answers = batch['answer'].to(device)

        # Forward pass
        logits, _, _ = model(input_ids)

        # Get prediction for answer position (last token)
        answer_logits = logits[:, -1, :]  # [batch, vocab]
        predictions = answer_logits.argmax(dim=-1)  # [batch]

        # Accuracy (exact match)
        correct = (predictions == answers.squeeze(-1)).sum().item()
        total_correct += correct
        total_samples += answers.size(0)

        # Loss on answer token only
        loss = F.cross_entropy(answer_logits, answers.squeeze(-1))
        total_loss += loss.item()
        n_batches += 1

    model.train()

    return {
        'val_accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        'val_loss': total_loss / n_batches if n_batches > 0 else 0.0,
        'val_samples': total_samples,
    }


def train_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    memory_states: Optional[list] = None,
    use_orthogonal_grad: bool = False,
    orthogonal_grad_strength: float = 1.0,
    grad_clip: float = 1.0,
) -> Tuple[Dict[str, float], list]:
    """Execute single training step."""
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    answers = batch['answer'].to(device)

    # Forward
    logits, memory_states, _ = model(
        input_ids,
        memory_states=memory_states,
        return_metrics=False,
    )

    # Loss on answer position only
    answer_logits = logits[:, -1, :]  # [batch, vocab]
    loss = criterion(answer_logits, answers.squeeze(-1))

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Orthogonal gradient projection (PerpGrad)
    if use_orthogonal_grad:
        apply_orthogonal_projection(model, strength=orthogonal_grad_strength)

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    optimizer.step()

    # Compute training accuracy
    predictions = answer_logits.argmax(dim=-1)
    train_acc = (predictions == answers.squeeze(-1)).float().mean().item()

    # Get gate metrics
    gate_metrics = model.get_gate_metrics()

    metrics = {
        'train_loss': loss.item(),
        'train_accuracy': train_acc,
        'grad_norm': grad_norm.item(),
        **gate_metrics,
    }

    return metrics, memory_states


def main():
    parser = argparse.ArgumentParser(description='Train Atlas on Modular Arithmetic')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    training_cfg = config.get('training', {})
    task_cfg = config.get('task', {})
    stability_cfg = config.get('monitoring', {}).get('stability', {})

    # Device
    device = torch.device(training_cfg.get('device', 'cuda:0'))
    print(f"Using device: {device}")

    # Create data loaders
    train_loader, val_loader, vocab_size = create_modular_arithmetic_dataloaders(
        prime=task_cfg.get('prime', 97),
        operation=task_cfg.get('operation', 'add'),
        batch_size=training_cfg.get('batch_size', 512),
        train_fraction=task_cfg.get('train_fraction', 0.5),
    )

    # Create model
    model = create_model(config, vocab_size)
    model = model.to(device)

    # Optimizer with weight decay (critical for grokking!)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg.get('learning_rate', 1e-3),
        weight_decay=training_cfg.get('weight_decay', 1.0),  # HIGH weight decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_cfg.get('max_steps', 100000),
        eta_min=training_cfg.get('learning_rate', 1e-3) * 0.1,
    )

    # Loss function
    if stability_cfg.get('use_stablemax', True):
        criterion = StableCrossEntropyLoss()
        print("Using StableMax loss")
    else:
        criterion = nn.CrossEntropyLoss()

    # Orthogonal gradient settings
    use_orthogonal_grad = stability_cfg.get('use_orthogonal_grad', True)
    orthogonal_grad_strength = stability_cfg.get('orthogonal_grad_strength', 1.0)

    # Grokking detector
    grok_cfg = config.get('monitoring', {}).get('grokking', {})
    grokking_detector = GrokkingDetector(GrokkingConfig(
        enabled=grok_cfg.get('enabled', True),
        metrics_interval=grok_cfg.get('metrics_interval', 500),
    ))

    # Setup output paths
    metrics_path = Path(training_cfg.get('metrics_path', 'runs/atlas_modular_arithmetic/metrics_stream.jsonl'))
    checkpoint_dir = Path(training_cfg.get('checkpoint_dir', 'runs/atlas_modular_arithmetic/checkpoints'))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training state
    state = TrainingState()
    memory_states = None

    # Metrics file
    metrics_file = open(metrics_path, 'a')

    print(f"\nStarting training...")
    print(f"  Max steps: {training_cfg.get('max_steps', 100000)}")
    print(f"  Weight decay: {training_cfg.get('weight_decay', 1.0)}")
    print(f"  Learning rate: {training_cfg.get('learning_rate', 1e-3)}")
    print(f"  Orthogonal grad: {use_orthogonal_grad}")
    print(f"  StableMax: {stability_cfg.get('use_stablemax', True)}")
    print()

    start_time = time.time()

    while state.step < training_cfg.get('max_steps', 100000):
        for batch in train_loader:
            state.step += 1

            # Training step
            metrics, memory_states = train_step(
                model=model,
                batch=batch,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                memory_states=memory_states,
                use_orthogonal_grad=use_orthogonal_grad,
                orthogonal_grad_strength=orthogonal_grad_strength,
                grad_clip=training_cfg.get('grad_clip', 1.0),
            )

            scheduler.step()

            # Evaluate periodically
            if state.step % training_cfg.get('eval_interval', 500) == 0:
                val_metrics = evaluate(model, val_loader, device)
                metrics.update(val_metrics)

                # Check for grokking
                if val_metrics['val_accuracy'] > state.best_val_acc:
                    state.best_val_acc = val_metrics['val_accuracy']

                if val_metrics['val_accuracy'] > 0.95 and not state.grokking_detected:
                    state.grokking_detected = True
                    state.grokking_step = state.step
                    print(f"\n{'='*60}")
                    print(f"GROKKING DETECTED at step {state.step}!")
                    print(f"  Train accuracy: {metrics['train_accuracy']:.1%}")
                    print(f"  Val accuracy:   {val_metrics['val_accuracy']:.1%}")
                    print(f"  Elapsed time:   {(time.time() - start_time) / 3600:.2f} hours")
                    print(f"{'='*60}\n")

            # Grokking metrics
            if state.step % grok_cfg.get('metrics_interval', 500) == 0:
                grok_metrics = grokking_detector.compute_metrics(
                    model=model,
                    step=state.step,
                    gate_mean=metrics.get('gate_mean', 1.0),
                )
                metrics.update(grok_metrics.to_dict())

            # Add metadata
            metrics['step'] = state.step
            metrics['epoch'] = state.epoch
            metrics['learning_rate'] = scheduler.get_last_lr()[0]
            metrics['elapsed_hours'] = (time.time() - start_time) / 3600
            metrics['best_val_acc'] = state.best_val_acc
            metrics['grokking_detected'] = state.grokking_detected

            # Log to file
            metrics_file.write(json.dumps(metrics) + '\n')
            metrics_file.flush()

            # Console output
            if state.step % training_cfg.get('log_interval', 100) == 0:
                val_acc = metrics.get('val_accuracy', 0)
                gate_mean = metrics.get('gate_mean', 0)
                print(
                    f"[Step {state.step:>6}] "
                    f"Train: {metrics['train_accuracy']:.1%} | "
                    f"Val: {val_acc:.1%} | "
                    f"Loss: {metrics['train_loss']:.4f} | "
                    f"Gate: {gate_mean:.1%} | "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            # Checkpoint
            if state.step % training_cfg.get('checkpoint_interval', 5000) == 0:
                ckpt_path = checkpoint_dir / f'checkpoint_{state.step}.pt'
                torch.save({
                    'step': state.step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_acc': state.best_val_acc,
                    'grokking_detected': state.grokking_detected,
                    'grokking_step': state.grokking_step,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # Early stopping if grokking achieved and confirmed
            early_cfg = config.get('early_stopping', {})
            if early_cfg.get('enabled', True):
                if state.grokking_detected:
                    steps_since_grok = state.step - state.grokking_step
                    if steps_since_grok >= early_cfg.get('patience', 1000):
                        # Verify it's sustained
                        val_metrics = evaluate(model, val_loader, device)
                        if val_metrics['val_accuracy'] > early_cfg.get('threshold', 0.99):
                            print(f"\nGrokking confirmed! Val accuracy: {val_metrics['val_accuracy']:.1%}")
                            print(f"Training complete at step {state.step}")
                            break

            if state.step >= training_cfg.get('max_steps', 100000):
                break

        state.epoch += 1

    # Final evaluation
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    val_metrics = evaluate(model, val_loader, device)
    print(f"Final validation accuracy: {val_metrics['val_accuracy']:.1%}")
    print(f"Best validation accuracy:  {state.best_val_acc:.1%}")
    print(f"Grokking detected: {state.grokking_detected}")
    if state.grokking_step:
        print(f"Grokking step: {state.grokking_step}")
    print(f"Total steps: {state.step}")
    print(f"Total time: {(time.time() - start_time) / 3600:.2f} hours")

    metrics_file.close()

    # Save final model
    final_path = checkpoint_dir / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'final_metrics': val_metrics,
        'grokking_detected': state.grokking_detected,
        'grokking_step': state.grokking_step,
    }, final_path)
    print(f"Saved final model: {final_path}")


if __name__ == '__main__':
    main()

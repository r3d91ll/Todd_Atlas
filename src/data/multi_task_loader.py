"""
Multi-Task Data Loader for Domain Universality Testing.

Provides SEPARATE dataloaders for language and math tasks to avoid
wasteful padding (math samples are 5 tokens, language ~256 tokens).

Key Features:
- Separate dataloaders for each task type (no padding waste)
- 50-50 split (Phase 1) controlled at episode level
- Random alternation between loaders during training

Design Decision: Separate batches by task type.
- Math batches: 32 × 5 = 160 tokens (efficient)
- Lang batches: 32 × 256 = 8,192 tokens (natural length)
- NOT mixed batches: 32 × 256 = 8,192 tokens (wasteful for math)
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Any
from dataclasses import dataclass


@dataclass
class DualDataLoaders:
    """Container for math and language dataloaders."""
    math: DataLoader
    language: DataLoader
    math_fraction: float = 0.5

    def __post_init__(self):
        self.rng = random.Random(42)

    def should_use_math(self) -> bool:
        """Randomly decide whether to use math or language for next batch."""
        return self.rng.random() < self.math_fraction

    def get_task_info(self) -> Dict[str, Any]:
        """Get info about the dataloaders."""
        return {
            'math_batches': len(self.math),
            'language_batches': len(self.language),
            'math_fraction': self.math_fraction,
        }


def create_dual_dataloaders(
    language_dataset: Dataset,
    math_prime: int = 97,
    math_operation: str = 'add',
    math_fraction: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> DualDataLoaders:
    """
    Create separate dataloaders for math and language tasks.

    This avoids the wasteful padding that occurs when mixing 5-token
    math samples with 256-token language samples in the same batch.

    Args:
        language_dataset: Language corpus dataset
        math_prime: Prime modulus for modular arithmetic
        math_operation: Operation ('add', 'sub', 'mul', 'div')
        math_fraction: Fraction of batches that should be math (0.5 = 50-50)
        batch_size: Batch size for both loaders
        seed: Random seed
        num_workers: DataLoader workers

    Returns:
        DualDataLoaders with separate math and language loaders
    """
    from src.data.modular_arithmetic import ModularArithmeticDataset

    # Create math dataset
    math_dataset = ModularArithmeticDataset(
        prime=math_prime,
        operation=math_operation,
        split='train',
        seed=seed,
    )

    # Math dataloader - simple stacking, no padding needed
    math_loader = DataLoader(
        math_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_math,
    )

    # Language dataloader - uses existing collation
    language_loader = DataLoader(
        language_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_language,
    )

    print("DualDataLoaders initialized:")
    print(f"  Math samples: {len(math_dataset)} ({len(math_loader)} batches)")
    print(f"  Language samples: {len(language_dataset)} ({len(language_loader)} batches)")
    print(f"  Math fraction: {math_fraction:.0%}")
    print(f"  Batch size: {batch_size}")
    print(f"  Math sequence length: 5 tokens (fixed)")
    print(f"  Language sequence length: ~{language_dataset.seq_length if hasattr(language_dataset, 'seq_length') else 256} tokens")

    return DualDataLoaders(
        math=math_loader,
        language=language_loader,
        math_fraction=math_fraction,
    )


def collate_math(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for math batches.

    Math samples are fixed-length (5 tokens), so just stack them.
    No padding needed!

    Returns:
        Dict with:
            - input_ids: [batch, 5]
            - answer: [batch, 1]
            - task_type: 'math'
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    answers = torch.stack([item['answer'] for item in batch])

    return {
        'input_ids': input_ids,
        'answer': answers,
        'task_type': 'math',  # Single string, not a list
    }


def collate_language(batch: list) -> Dict[str, torch.Tensor]:
    """
    Collate function for language batches.

    Language samples may vary slightly in length, so pad to max in batch.

    Returns:
        Dict with:
            - input_ids: [batch, seq_len]
            - labels: [batch, seq_len] (if present)
            - task_type: 'language'
    """
    input_ids_list = [item['input_ids'] if isinstance(item, dict) else item for item in batch]

    # Handle case where dataset returns just tensors
    if not isinstance(batch[0], dict):
        input_ids = torch.stack(input_ids_list)
        return {
            'input_ids': input_ids,
            'task_type': 'language',
        }

    # Find max length in this batch
    max_len = max(ids.size(-1) for ids in input_ids_list)

    # Pad to max length
    padded_ids = []
    for ids in input_ids_list:
        if ids.size(-1) < max_len:
            padding = torch.zeros(max_len - ids.size(-1), dtype=ids.dtype)
            ids = torch.cat([ids, padding])
        padded_ids.append(ids)

    input_ids = torch.stack(padded_ids)

    result = {
        'input_ids': input_ids,
        'task_type': 'language',
    }

    # Handle labels if present
    if isinstance(batch[0], dict) and 'labels' in batch[0]:
        labels_list = [item.get('labels', item['input_ids']) for item in batch]
        padded_labels = []
        for labels in labels_list:
            if labels.size(-1) < max_len:
                padding = torch.full((max_len - labels.size(-1),), -100, dtype=labels.dtype)
                labels = torch.cat([labels, padding])
            padded_labels.append(labels)
        result['labels'] = torch.stack(padded_labels)

    return result


# Keep old functions for backwards compatibility but mark as deprecated
def create_multi_task_dataloader(*args, **kwargs):
    """DEPRECATED: Use create_dual_dataloaders instead."""
    raise NotImplementedError(
        "create_multi_task_dataloader is deprecated. "
        "Use create_dual_dataloaders() which returns separate loaders "
        "to avoid wasteful padding of 5-token math samples to 256 tokens."
    )


def collate_multi_task(*args, **kwargs):
    """DEPRECATED: Use collate_math or collate_language instead."""
    raise NotImplementedError(
        "collate_multi_task is deprecated. "
        "Use separate collate_math() and collate_language() functions."
    )

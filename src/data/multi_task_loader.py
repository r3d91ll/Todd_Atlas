"""
Multi-Task Data Loader for Domain Universality Testing.

Combines language corpus with modular arithmetic to test whether
grokking is domain-universal or domain-specific.

Key Features:
- 50-50 split (Phase 1) between language and math
- Random ordering within episodes (NOT blocks)
- Task type tagging for metric routing

Design Decision: Random ordering prevents pattern exploitation.
NOT: [Lit, Lit, Lit, Lit, Lit, Math, Math, Math, Math, Math]
YES: [Math, Lit, Lit, Math, Lit, Math, Math, Lit, Math, Lit]
"""

import torch
import random
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any


class MultiTaskDataset(Dataset):
    """
    Combines language corpus with modular arithmetic.
    Tags each batch with task type for metric routing.

    CRITICAL: Uses random ordering within episodes, NOT blocks.
    """

    def __init__(
        self,
        language_dataset: Dataset,
        math_dataset: Dataset,
        math_fraction: float = 0.5,  # 50-50 for Phase 1
        random_ordering: bool = True,  # Random within episodes
        seed: int = 42,
    ):
        """
        Initialize multi-task dataset.

        Args:
            language_dataset: Dataset for language (Shakespeare/Dumas)
            math_dataset: Dataset for modular arithmetic
            math_fraction: Fraction of data that is math (0.5 = 50-50)
            random_ordering: If True, randomly interleave. If False, block ordering.
            seed: Random seed for reproducibility
        """
        self.language_dataset = language_dataset
        self.math_dataset = math_dataset
        self.math_fraction = math_fraction
        self.random_ordering = random_ordering
        self.rng = random.Random(seed)

        # Create indices with proper mixing
        self.indices = self._create_mixed_indices()

        print("MultiTaskDataset initialized:")
        print(f"  Language samples: {len(language_dataset)}")
        print(f"  Math samples: {len(math_dataset)}")
        print(f"  Math fraction: {math_fraction:.0%}")
        print(f"  Random ordering: {random_ordering}")
        print(f"  Total mixed samples: {len(self.indices)}")

    def _create_mixed_indices(self) -> List[Tuple[str, int]]:
        """
        Create indices that mix language and math at desired ratio.

        If random_ordering=True: Randomly interleave (no predictable pattern)
        If random_ordering=False: Block ordering (language then math)
        """
        lang_len = len(self.language_dataset)
        math_len = len(self.math_dataset)

        # Calculate how many of each based on fraction
        # Use the smaller dataset to determine total, then scale
        if self.math_fraction == 0.5:
            # 50-50 split: use minimum of both, doubled
            min_len = min(lang_len, math_len)
            target_lang = min_len
            target_math = min_len
        else:
            # General case: scale based on fraction
            total_available = lang_len + math_len
            target_math = int(total_available * self.math_fraction)
            target_lang = total_available - target_math
            # Clamp to available
            target_math = min(target_math, math_len)
            target_lang = min(target_lang, lang_len)

        # Create index lists (cycle if needed)
        lang_indices = [('language', i % lang_len) for i in range(target_lang)]
        math_indices = [('math', i % math_len) for i in range(target_math)]

        if self.random_ordering:
            # Random interleaving - no predictable pattern
            indices = lang_indices + math_indices
            self.rng.shuffle(indices)
        else:
            # Block ordering (NOT recommended - allows pattern exploitation)
            indices = lang_indices + math_indices

        return indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task_type, dataset_idx = self.indices[idx]

        if task_type == 'language':
            item = self.language_dataset[dataset_idx]
            # Ensure we return a dict copy to avoid modifying original
            if isinstance(item, dict):
                item = dict(item)
            else:
                # Handle case where dataset returns tensors directly
                item = {'input_ids': item}
        else:
            item = self.math_dataset[dataset_idx]
            if isinstance(item, dict):
                item = dict(item)
            else:
                item = {'input_ids': item}

        # Tag with task type for metric routing
        item['task_type'] = task_type
        return item

    def get_task_counts(self) -> Dict[str, int]:
        """Get count of each task type in the dataset."""
        counts = {'language': 0, 'math': 0}
        for task_type, _ in self.indices:
            counts[task_type] += 1
        return counts


def create_multi_task_dataloader(
    language_dataset: Dataset,
    math_prime: int = 97,
    math_operation: str = 'add',
    math_fraction: float = 0.5,
    random_ordering: bool = True,
    batch_size: int = 32,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create dataloader that mixes language and math tasks.

    Args:
        language_dataset: Language corpus dataset
        math_prime: Prime modulus for modular arithmetic
        math_operation: Operation ('add', 'sub', 'mul', 'div')
        math_fraction: Fraction of math samples (0.5 = 50-50)
        random_ordering: Random interleaving within epochs
        batch_size: Batch size
        seed: Random seed
        num_workers: DataLoader workers

    Returns:
        DataLoader with mixed language and math samples
    """
    from src.data.modular_arithmetic import ModularArithmeticDataset

    math_dataset = ModularArithmeticDataset(
        prime=math_prime,
        operation=math_operation,
        split='train',
        seed=seed,
    )

    multi_dataset = MultiTaskDataset(
        language_dataset=language_dataset,
        math_dataset=math_dataset,
        math_fraction=math_fraction,
        random_ordering=random_ordering,
        seed=seed,
    )

    # shuffle=False because we pre-shuffled with random_ordering
    # This preserves the random interleaving we set up
    return DataLoader(
        multi_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def collate_multi_task(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for multi-task batches.

    Handles the case where language and math have different tensor shapes
    by padding to the longest sequence in the batch.

    Args:
        batch: List of sample dicts from MultiTaskDataset

    Returns:
        Collated batch with padded tensors and task_types list
    """
    # Separate by task type
    task_types = [item['task_type'] for item in batch]

    # Get all input_ids
    input_ids_list = [item['input_ids'] for item in batch]

    # Find max length
    max_len = max(ids.size(-1) for ids in input_ids_list)

    # Pad all to max length
    padded_ids = []
    for ids in input_ids_list:
        if ids.size(-1) < max_len:
            padding = torch.zeros(max_len - ids.size(-1), dtype=ids.dtype)
            ids = torch.cat([ids, padding])
        padded_ids.append(ids)

    # Stack
    input_ids = torch.stack(padded_ids)

    result = {
        'input_ids': input_ids,
        'task_types': task_types,
    }

    # Handle optional fields (labels, answer, etc.)
    if 'labels' in batch[0]:
        labels_list = [item.get('labels', item['input_ids']) for item in batch]
        padded_labels = []
        for labels in labels_list:
            if labels.size(-1) < max_len:
                # Pad with -100 (ignore index)
                padding = torch.full((max_len - labels.size(-1),), -100, dtype=labels.dtype)
                labels = torch.cat([labels, padding])
            padded_labels.append(labels)
        result['labels'] = torch.stack(padded_labels)

    if 'answer' in batch[0]:
        # Only math batches have 'answer'
        # Get dtype from first available answer for consistency
        answer_dtype = next(
            (item['answer'].dtype for item in batch if 'answer' in item),
            torch.long
        )
        answers = [
            item.get('answer', torch.tensor([-1], dtype=answer_dtype))
            for item in batch
        ]
        result['answer'] = torch.stack(answers)

    return result

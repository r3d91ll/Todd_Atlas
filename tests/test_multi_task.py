"""
Tests for Multi-Task Episodic Memory Training.

Tests the masking infrastructure and multi-task data loader.
"""

import pytest
import torch
import random
from typing import Dict

# Import modules to test
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.masking import (
    create_masked_batch,
    create_math_masked_batch,
    compute_masked_accuracy,
)
from src.data.multi_task_loader import (
    MultiTaskDataset,
    collate_multi_task,
)


class TestMasking:
    """Tests for pure random masking infrastructure."""

    def test_create_masked_batch_basic(self):
        """Test basic masked batch creation."""
        batch = {
            'input_ids': torch.tensor([
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ])
        }

        masked_batch, mask_positions, original_tokens = create_masked_batch(
            batch,
            mask_token_id=99,
            num_masks=1,
            pad_token_id=0,
            seed=42,
        )

        # Check shapes
        assert masked_batch['input_ids'].shape == batch['input_ids'].shape
        assert mask_positions.shape == (2, 1)
        assert original_tokens.shape == (2, 1)

        # Check that masks were applied
        for i in range(2):
            pos = mask_positions[i, 0].item()
            assert masked_batch['input_ids'][i, pos] == 99
            assert original_tokens[i, 0] == batch['input_ids'][i, pos]

    def test_create_masked_batch_skips_padding(self):
        """Test that padding tokens are never masked."""
        batch = {
            'input_ids': torch.tensor([
                [1, 2, 0, 0, 0],  # Only 2 valid tokens
                [6, 7, 8, 0, 0],  # 3 valid tokens
            ])
        }

        masked_batch, mask_positions, original_tokens = create_masked_batch(
            batch,
            mask_token_id=99,
            num_masks=1,
            pad_token_id=0,
            seed=42,
        )

        # Check that mask positions are in valid range
        for i in range(2):
            pos = mask_positions[i, 0].item()
            if pos >= 0:  # Valid position
                # Should not be a padding position
                assert batch['input_ids'][i, pos] != 0

    def test_create_masked_batch_multiple_masks(self):
        """Test multiple masks per sequence."""
        batch = {
            'input_ids': torch.tensor([
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            ])
        }

        masked_batch, mask_positions, original_tokens = create_masked_batch(
            batch,
            mask_token_id=99,
            num_masks=3,
            pad_token_id=0,
            seed=42,
        )

        # Check shapes
        assert mask_positions.shape == (1, 3)
        assert original_tokens.shape == (1, 3)

        # Check all positions are unique
        positions = mask_positions[0].tolist()
        assert len(set(positions)) == len(positions)

        # Check all masks applied
        for k in range(3):
            pos = mask_positions[0, k].item()
            assert masked_batch['input_ids'][0, pos] == 99

    def test_create_masked_batch_reproducible(self):
        """Test that seed makes masking reproducible."""
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]])
        }

        _, pos1, tok1 = create_masked_batch(batch, 99, 1, 0, seed=42)
        _, pos2, tok2 = create_masked_batch(batch, 99, 1, 0, seed=42)

        assert torch.equal(pos1, pos2)
        assert torch.equal(tok1, tok2)

    def test_create_math_masked_batch(self):
        """Test math batch masking (answer at last position)."""
        batch = {
            'input_ids': torch.tensor([
                [23, 97, 45, 101, 68],  # "23 + 45 = 68"
                [10, 97, 20, 101, 30],  # "10 + 20 = 30"
            ]),
            'answer': torch.tensor([[68], [30]]),
        }

        masked_batch, mask_positions, original_tokens = create_math_masked_batch(
            batch,
            mask_token_id=99,
        )

        # Answer should be masked (last position)
        assert masked_batch['input_ids'][0, 4] == 99
        assert masked_batch['input_ids'][1, 4] == 99

        # Mask positions should be last position
        assert (mask_positions == 4).all()

    def test_compute_masked_accuracy_perfect(self):
        """Test accuracy computation with perfect predictions."""
        predictions = torch.tensor([[5, 10], [15, 20]])
        original_tokens = torch.tensor([[5, 10], [15, 20]])
        mask_positions = torch.tensor([[0, 1], [0, 1]])

        accuracy = compute_masked_accuracy(predictions, original_tokens, mask_positions)
        assert accuracy == 1.0

    def test_compute_masked_accuracy_partial(self):
        """Test accuracy computation with partial correctness."""
        predictions = torch.tensor([[5, 99], [15, 99]])  # Half wrong
        original_tokens = torch.tensor([[5, 10], [15, 20]])
        mask_positions = torch.tensor([[0, 1], [0, 1]])

        accuracy = compute_masked_accuracy(predictions, original_tokens, mask_positions)
        assert accuracy == 0.5

    def test_compute_masked_accuracy_handles_invalid_positions(self):
        """Test that -1 positions are ignored in accuracy."""
        predictions = torch.tensor([[5, 99]])
        original_tokens = torch.tensor([[5, -1]])  # Second position invalid
        mask_positions = torch.tensor([[0, -1]])

        accuracy = compute_masked_accuracy(predictions, original_tokens, mask_positions)
        assert accuracy == 1.0  # Only first position counted


class TestMultiTaskDataset:
    """Tests for multi-task data loader."""

    def create_mock_dataset(self, size: int, prefix: str = "lang"):
        """Create a simple mock dataset."""
        class MockDataset(torch.utils.data.Dataset):
            def __init__(self, size, prefix):
                self.size = size
                self.prefix = prefix

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'input_ids': torch.ones(10) * idx,
                    'name': f'{self.prefix}_{idx}',
                }

        return MockDataset(size, prefix)

    def test_multi_task_dataset_50_50_split(self):
        """Test that 50-50 split is respected."""
        lang_ds = self.create_mock_dataset(100, "lang")
        math_ds = self.create_mock_dataset(100, "math")

        multi_ds = MultiTaskDataset(
            language_dataset=lang_ds,
            math_dataset=math_ds,
            math_fraction=0.5,
            random_ordering=True,
            seed=42,
        )

        counts = multi_ds.get_task_counts()
        assert counts['language'] == 100
        assert counts['math'] == 100

    def test_multi_task_dataset_task_type_tagging(self):
        """Test that items are tagged with task type."""
        lang_ds = self.create_mock_dataset(10, "lang")
        math_ds = self.create_mock_dataset(10, "math")

        multi_ds = MultiTaskDataset(
            language_dataset=lang_ds,
            math_dataset=math_ds,
            math_fraction=0.5,
            random_ordering=True,
            seed=42,
        )

        for i in range(len(multi_ds)):
            item = multi_ds[i]
            assert 'task_type' in item
            assert item['task_type'] in ('language', 'math')

    def test_multi_task_dataset_random_ordering(self):
        """Test that random ordering mixes task types."""
        lang_ds = self.create_mock_dataset(50, "lang")
        math_ds = self.create_mock_dataset(50, "math")

        multi_ds = MultiTaskDataset(
            language_dataset=lang_ds,
            math_dataset=math_ds,
            math_fraction=0.5,
            random_ordering=True,
            seed=42,
        )

        # Get first 20 items and check for mixing
        task_types = [multi_ds[i]['task_type'] for i in range(20)]

        # Should have both types in first 20 (very unlikely to be all one type)
        assert 'language' in task_types
        assert 'math' in task_types

    def test_multi_task_dataset_block_ordering(self):
        """Test that block ordering keeps types together."""
        lang_ds = self.create_mock_dataset(50, "lang")
        math_ds = self.create_mock_dataset(50, "math")

        multi_ds = MultiTaskDataset(
            language_dataset=lang_ds,
            math_dataset=math_ds,
            math_fraction=0.5,
            random_ordering=False,  # Block ordering
            seed=42,
        )

        # First 50 should all be language
        first_50_types = [multi_ds[i]['task_type'] for i in range(50)]
        assert all(t == 'language' for t in first_50_types)

        # Last 50 should all be math
        last_50_types = [multi_ds[i]['task_type'] for i in range(50, 100)]
        assert all(t == 'math' for t in last_50_types)

    def test_collate_multi_task_basic(self):
        """Test custom collate function."""
        batch = [
            {'input_ids': torch.tensor([1, 2, 3]), 'task_type': 'language'},
            {'input_ids': torch.tensor([4, 5, 6, 7]), 'task_type': 'math'},
        ]

        collated = collate_multi_task(batch)

        assert 'input_ids' in collated
        assert 'task_types' in collated
        assert collated['input_ids'].shape == (2, 4)  # Padded to max length
        assert collated['task_types'] == ['language', 'math']


class TestIntegration:
    """Integration tests for the full multi-task pipeline."""

    def test_mask_then_unmask_recovers_original(self):
        """Test that we can recover original tokens from mask info."""
        batch = {
            'input_ids': torch.tensor([
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
            ])
        }

        masked_batch, mask_positions, original_tokens = create_masked_batch(
            batch.copy(),
            mask_token_id=99,
            num_masks=2,
            pad_token_id=0,
            seed=42,
        )

        # Recover original
        recovered = masked_batch['input_ids'].clone()
        for i in range(2):
            for k in range(2):
                pos = mask_positions[i, k].item()
                if pos >= 0:
                    recovered[i, pos] = original_tokens[i, k]

        # Should match original
        assert torch.equal(recovered, batch['input_ids'])

    def test_pure_random_no_heuristics(self):
        """Test that masking is truly random (no content-word bias)."""
        # Create a batch with common and rare tokens
        # If there were heuristics, rare tokens would be masked more often
        batch = {
            'input_ids': torch.tensor([
                [1, 1, 1, 1, 1, 100, 100, 100, 100, 100],  # 1=common, 100=rare
            ] * 100)  # 100 samples
        }

        common_masked = 0
        rare_masked = 0

        for seed in range(100):
            _, positions, tokens = create_masked_batch(
                batch,
                mask_token_id=99,
                num_masks=1,
                pad_token_id=0,
                seed=seed,
            )

            for i in range(100):
                tok = tokens[i, 0].item()
                if tok == 1:
                    common_masked += 1
                elif tok == 100:
                    rare_masked += 1

        # Should be roughly 50-50 (allow 40-60% range)
        total = common_masked + rare_masked
        common_ratio = common_masked / total

        assert 0.40 < common_ratio < 0.60, f"Common ratio {common_ratio} suggests heuristic bias"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

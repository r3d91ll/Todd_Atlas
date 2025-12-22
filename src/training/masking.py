"""
Pure Random Masking for Multi-Task Episodic Memory Training.

This module implements PURE RANDOM masking with NO heuristics.

Design Decision: Any heuristic becomes a learnable shortcut:
- "Always mask 4+ char words" -> Model learns phrase completion
- "Never mask function words" -> Model learns grammatical patterns
- "Pure random masking" -> Model MUST use actual memory

The only token we skip is padding. Everything else is fair game.
"""

import torch
import random
from typing import Dict, Tuple, Optional


def create_masked_batch(
    batch: Dict[str, torch.Tensor],
    mask_token_id: int,
    num_masks: int = 1,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Create masked version of batch with PURE RANDOM masking.

    No heuristics. No filtering. Mask ANY token (except padding).
    Could mask "the", "I", "compare", "summer" - anything goes.

    Args:
        batch: Original batch with 'input_ids' tensor [batch_size, seq_len]
        mask_token_id: Token ID to use for [MASK]
        num_masks: Number of tokens to mask per sequence
        pad_token_id: Only token to avoid masking
        seed: Optional random seed for reproducibility

    Returns:
        masked_batch: Batch with masks applied
        mask_positions: [batch_size, num_masks] positions that were masked
        original_tokens: [batch_size, num_masks] original tokens at masked positions
    """
    # Use local Random instance to avoid mutating global random state
    rng = random.Random(seed) if seed is not None else random.Random()

    input_ids = batch['input_ids'].clone()
    batch_size, seq_len = input_ids.shape

    mask_positions = torch.zeros(batch_size, num_masks, dtype=torch.long)
    original_tokens = torch.zeros(batch_size, num_masks, dtype=torch.long)

    for i in range(batch_size):
        # Only avoid padding - everything else is fair game
        valid_positions = [
            j for j in range(seq_len)
            if input_ids[i, j].item() != pad_token_id
        ]

        if len(valid_positions) >= num_masks:
            # Randomly select positions (no heuristics!)
            selected = rng.sample(valid_positions, num_masks)
            for k, pos in enumerate(selected):
                mask_positions[i, k] = pos
                original_tokens[i, k] = input_ids[i, pos].clone()
                input_ids[i, pos] = mask_token_id
        else:
            # Not enough valid positions - mask what we can
            for k, pos in enumerate(valid_positions):
                mask_positions[i, k] = pos
                original_tokens[i, k] = input_ids[i, pos].clone()
                input_ids[i, pos] = mask_token_id
            # Fill remaining with -1 to indicate no mask
            for k in range(len(valid_positions), num_masks):
                mask_positions[i, k] = -1
                original_tokens[i, k] = -1

    masked_batch = {**batch, 'input_ids': input_ids}
    return masked_batch, mask_positions, original_tokens


def create_math_masked_batch(
    batch: Dict[str, torch.Tensor],
    mask_token_id: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Create masked version for modular arithmetic.

    For math, we always mask the answer position (last token before padding).
    Input:  [a, op, b, =, answer]
    Masked: [a, op, b, =, MASK]

    Args:
        batch: Batch with 'input_ids' and 'answer' tensors
        mask_token_id: Token ID to use for [MASK]

    Returns:
        masked_batch: Batch with answer masked
        mask_positions: [batch_size, 1] position of the answer
        original_tokens: [batch_size, 1] the correct answer
    """
    input_ids = batch['input_ids'].clone()
    batch_size, seq_len = input_ids.shape

    # Answer is always at the last position for math
    answer_position = seq_len - 1

    mask_positions = torch.full((batch_size, 1), answer_position, dtype=torch.long)
    original_tokens = batch['answer'].clone()

    # Mask the answer position
    input_ids[:, answer_position] = mask_token_id

    masked_batch = {**batch, 'input_ids': input_ids}
    return masked_batch, mask_positions, original_tokens


def compute_masked_accuracy(
    predictions: torch.Tensor,
    original_tokens: torch.Tensor,
    mask_positions: torch.Tensor,
) -> float:
    """
    Compute exact match accuracy for masked predictions.

    Args:
        predictions: [batch_size, num_masks] predicted token IDs
        original_tokens: [batch_size, num_masks] ground truth token IDs
        mask_positions: [batch_size, num_masks] positions (-1 means no mask)

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    # Create mask for valid positions (not -1)
    valid_mask = mask_positions >= 0

    if valid_mask.sum() == 0:
        return 0.0

    # Compare only valid positions
    correct = (predictions == original_tokens) & valid_mask
    accuracy = correct.float().sum() / valid_mask.float().sum()

    return accuracy.item()

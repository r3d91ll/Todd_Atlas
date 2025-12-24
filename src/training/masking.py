"""
Word-Level Masking for Episodic Memory Training.

This module implements WORD-LEVEL masking where entire words are replaced
with a single [MASK] token, regardless of how many subword tokens they contain.

Design Decision: Word-level masking tests semantic memory more directly:
- Token masking: "Did you predict '▁sum' + 'mer' correctly?" (morphology)
- Word masking: "Did you remember the word 'summer'?" (memory)

For episodic memory, we want to test whether the model stored and retrieved
the actual words, not whether it can guess subword patterns.
"""

import torch
import random
from typing import Dict, Tuple, Optional, List
from transformers import AutoTokenizer

# Cache tokenizer for word boundary detection
_tokenizer = None

def _get_tokenizer():
    """Get cached tokenizer for word boundary detection."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")
    return _tokenizer


def _identify_words(token_ids: torch.Tensor, pad_token_id: int = 0) -> List[List[int]]:
    """
    Identify word boundaries in a tokenized sequence.

    T5 tokenizer uses '▁' (U+2581) prefix to mark word starts.
    Groups consecutive tokens into words.

    Args:
        token_ids: 1D tensor of token IDs
        pad_token_id: Padding token to skip

    Returns:
        List of words, where each word is a list of token positions
    """
    tokenizer = _get_tokenizer()
    words = []
    current_word = []

    # Get the raw token strings (preserves ▁ marker)
    token_strs = tokenizer.convert_ids_to_tokens(token_ids.tolist())

    for pos, (tid, token_str) in enumerate(zip(token_ids.tolist(), token_strs)):
        if tid == pad_token_id:
            # End current word if any, skip padding
            if current_word:
                words.append(current_word)
                current_word = []
            continue

        # Skip special tokens like </s>, <pad>
        if token_str in ['</s>', '<pad>', '<unk>', '<s>']:
            if current_word:
                words.append(current_word)
                current_word = []
            continue

        # Check if this starts a new word
        # T5/SentencePiece uses ▁ prefix for word-initial tokens
        is_word_start = token_str.startswith('▁')

        if is_word_start and current_word:
            # Save previous word, start new one
            words.append(current_word)
            current_word = [pos]
        elif is_word_start:
            # First word in sequence
            current_word = [pos]
        else:
            # Continue current word (subword token)
            current_word.append(pos)

    # Don't forget last word
    if current_word:
        words.append(current_word)

    return words


def create_masked_batch(
    batch: Dict[str, torch.Tensor],
    mask_token_id: int,
    num_masks: int = 1,
    pad_token_id: int = 0,
    seed: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Create masked version of batch with WORD-LEVEL masking.

    Each selected word (regardless of subword token count) is replaced
    with a SINGLE [MASK] token. The sequence is shifted and padded
    to maintain original length.

    Args:
        batch: Original batch with 'input_ids' tensor [batch_size, seq_len]
        mask_token_id: Token ID to use for [MASK]
        num_masks: Number of WORDS to mask per sequence
        pad_token_id: Padding token ID
        seed: Optional random seed for reproducibility

    Returns:
        masked_batch: Batch with masks applied
        mask_positions: [batch_size, num_masks] positions of [MASK] tokens
        original_tokens: [batch_size, num_masks] FIRST token of each masked word
                        (used for accuracy - predicting first token of word)
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    input_ids = batch['input_ids'].clone()
    batch_size, seq_len = input_ids.shape

    mask_positions = torch.full((batch_size, num_masks), -1, dtype=torch.long)
    original_tokens = torch.full((batch_size, num_masks), -1, dtype=torch.long)

    new_input_ids = torch.full_like(input_ids, pad_token_id)

    for i in range(batch_size):
        # Identify words in this sequence
        words = _identify_words(input_ids[i], pad_token_id)

        if len(words) < num_masks:
            # Not enough words - mask what we can
            words_to_mask = list(range(len(words)))
        else:
            # Randomly select words to mask
            words_to_mask = rng.sample(range(len(words)), num_masks)

        # Sort by position so we process left-to-right
        words_to_mask.sort()

        # Build new sequence with masks
        new_seq = []
        masked_word_idx = 0

        for word_idx, word_positions in enumerate(words):
            if word_idx in words_to_mask and masked_word_idx < num_masks:
                # Replace entire word with single [MASK]
                mask_pos = len(new_seq)
                new_seq.append(mask_token_id)

                # Record mask position and original first token
                mask_positions[i, masked_word_idx] = mask_pos
                original_tokens[i, masked_word_idx] = input_ids[i, word_positions[0]].item()
                masked_word_idx += 1
            else:
                # Keep all tokens of this word
                for pos in word_positions:
                    new_seq.append(input_ids[i, pos].item())

        # Copy to output tensor (truncate or pad as needed)
        actual_len = min(len(new_seq), seq_len)
        new_input_ids[i, :actual_len] = torch.tensor(new_seq[:actual_len], dtype=torch.long)

    masked_batch = {**batch, 'input_ids': new_input_ids}
    return masked_batch, mask_positions, original_tokens


def create_math_masked_batch(
    batch: Dict[str, torch.Tensor],
    mask_token_id: Optional[int] = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Create masked version for modular arithmetic.

    For math, we always mask the answer position (position 4 in the sequence).
    Input:  [a, op, b, =, answer, pad, pad, ...]
    Masked: [a, op, b, =, MASK,   pad, pad, ...]

    The math sequence format is fixed:
        Position 0: a (first operand, offset+0 to offset+prime-1)
        Position 1: op token (offset+prime to offset+prime+3)
        Position 2: b (second operand, offset+0 to offset+prime-1)
        Position 3: = token (offset+prime+4)
        Position 4: answer (offset+0 to offset+prime-1)

    IMPORTANT: Uses math-specific mask token (offset+prime+5) to keep
    the entire sequence in the math token range. Using a language mask
    token (like 3) confuses the model by mixing token domains.

    Args:
        batch: Batch with 'input_ids' and 'answer' tensors
        mask_token_id: Token ID to use for [MASK]. If None, uses math-specific
                       mask token (31102 = offset + prime + 5)

    Returns:
        masked_batch: Batch with answer masked
        mask_positions: [batch_size, 1] position of the answer
        original_tokens: [batch_size, 1] the correct answer
    """
    # Use math-specific mask token by default
    if mask_token_id is None:
        from src.data.modular_arithmetic import ModularArithmeticDataset
        mask_token_id = ModularArithmeticDataset.get_mask_token_id()
    input_ids = batch['input_ids'].clone()
    batch_size, seq_len = input_ids.shape

    # Answer is always at position 4 for math (fixed format)
    # NOT seq_len - 1, which would be wrong after padding
    answer_position = 4

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

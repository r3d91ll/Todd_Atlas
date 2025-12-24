"""
Modular Arithmetic Dataset for Grokking Experiments.

Generates examples of modular arithmetic (a op b = c mod p) for testing
whether grokking mechanism works with meaningful geometric metrics.

Based on Power et al. (2022): "Grokking: Generalization Beyond Overfitting
on Small Algorithmic Datasets"
"""

import torch
from torch.utils.data import Dataset
from typing import Literal, Optional
import random


class ModularArithmeticDataset(Dataset):
    """
    Dataset of modular arithmetic examples.

    Each example is: (a, op_token, b, eq_token, answer)
    where answer = (a op b) mod prime

    For multi-task training, this provides examples where:
    - Fourier concentration metrics are meaningful
    - Circular fit metrics track the mod-p structure
    - Grokking phase detection works correctly

    Token format (with offset to avoid T5 vocab collision):
        - Numbers 0 to prime-1 map to token IDs (offset + 0) to (offset + prime-1)
        - Operation tokens: + = offset+prime, - = offset+prime+1, etc.
        - Equals token: = = offset+prime+4

    IMPORTANT: Default offset=31000 places math tokens at end of 32K vocab,
    avoiding collision with T5 tokenizer which uses tokens 0-101 for common
    words like "the", "and", "I", "but", etc.
    """

    # Operation token offsets (added to prime + token_offset)
    OP_ADD = 0
    OP_SUB = 1
    OP_MUL = 2
    OP_DIV = 3
    OP_EQ = 4
    OP_MASK = 5  # Math-specific mask token

    # Default token offset to avoid T5 vocab collision
    # T5 uses tokens 0-101 for common words; we use 31000+ to stay safe
    DEFAULT_TOKEN_OFFSET = 31000

    @classmethod
    def get_mask_token_id(cls, prime: int = 97, token_offset: int = DEFAULT_TOKEN_OFFSET) -> int:
        """Get the math-specific mask token ID."""
        return token_offset + prime + cls.OP_MASK

    def __init__(
        self,
        prime: int = 97,
        operation: Literal["add", "sub", "mul", "div"] = "add",
        split: Literal["train", "val", "test"] = "train",
        train_fraction: float = 0.5,
        seed: int = 42,
        token_offset: int = DEFAULT_TOKEN_OFFSET,
    ):
        """
        Initialize modular arithmetic dataset.

        Args:
            prime: The modulus for arithmetic (default 97, a prime)
            operation: Which operation to use
            split: train/val/test split
            train_fraction: Fraction of examples for training (rest split 50-50 val/test)
            seed: Random seed for reproducibility
            token_offset: Offset added to all tokens to avoid vocab collision (default 31000)
        """
        self.prime = prime
        self.operation = operation
        self.split = split
        self.token_offset = token_offset

        # Token IDs (with offset to avoid T5 vocab collision)
        # Numbers: 0-96 → 31000-31096
        # Operators: + → 31097, - → 31098, * → 31099, / → 31100, = → 31101
        self.op_token = token_offset + prime + {
            "add": self.OP_ADD,
            "sub": self.OP_SUB,
            "mul": self.OP_MUL,
            "div": self.OP_DIV,
        }[operation]
        self.eq_token = token_offset + prime + self.OP_EQ

        # Generate all possible (a, b) pairs
        all_pairs = [(a, b) for a in range(prime) for b in range(prime)]

        # For division, exclude b=0
        if operation == "div":
            all_pairs = [(a, b) for a, b in all_pairs if b != 0]

        # Shuffle deterministically
        rng = random.Random(seed)
        rng.shuffle(all_pairs)

        # Split into train/val/test
        n_total = len(all_pairs)
        n_train = int(n_total * train_fraction)
        n_remaining = n_total - n_train
        n_val = n_remaining // 2

        if split == "train":
            self.pairs = all_pairs[:n_train]
        elif split == "val":
            self.pairs = all_pairs[n_train:n_train + n_val]
        else:  # test
            self.pairs = all_pairs[n_train + n_val:]

        print(f"ModularArithmeticDataset ({operation} mod {prime}):")
        print(f"  Split: {split}, Examples: {len(self.pairs)}")
        print(f"  Token offset: {token_offset} (numbers use {token_offset}-{token_offset + prime - 1})")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single example.

        Returns:
            dict with:
                - input_ids: [a, op, b, eq, answer] tensor (all with offset applied)
                - answer: [answer] tensor (with offset, for easy loss computation)
                - task_type: "math" (for metric routing)
        """
        a, b = self.pairs[idx]

        # Compute answer based on operation (raw value 0 to prime-1)
        if self.operation == "add":
            answer_raw = (a + b) % self.prime
        elif self.operation == "sub":
            answer_raw = (a - b) % self.prime
        elif self.operation == "mul":
            answer_raw = (a * b) % self.prime
        elif self.operation == "div":
            # Modular division: a / b = a * b^(-1) mod p
            # b^(-1) mod p = b^(p-2) mod p (Fermat's little theorem)
            b_inv = pow(b, self.prime - 2, self.prime)
            answer_raw = (a * b_inv) % self.prime

        # Apply token offset to avoid T5 vocab collision
        # Numbers: raw 0-96 → tokens 31000-31096
        a_token = self.token_offset + a
        b_token = self.token_offset + b
        answer_token = self.token_offset + answer_raw

        # Create input sequence: [a, op, b, =, answer] (all with offset)
        input_ids = torch.tensor(
            [a_token, self.op_token, b_token, self.eq_token, answer_token],
            dtype=torch.long
        )

        return {
            'input_ids': input_ids,
            'answer': torch.tensor([answer_token], dtype=torch.long),
            'task_type': 'math',
        }

    def get_vocab_size(self) -> int:
        """Return vocabulary size needed for this dataset."""
        # Numbers 0 to prime-1, plus 5 special tokens (+, -, *, /, =)
        return self.prime + 5

    @staticmethod
    def decode_example(
        input_ids: torch.Tensor,
        prime: int = 97,
        token_offset: int = DEFAULT_TOKEN_OFFSET,
    ) -> str:
        """Decode an example back to human-readable form."""
        # With offset: numbers are offset to offset+prime-1
        # Operators are offset+prime to offset+prime+3
        # Equals is offset+prime+4
        ops = {
            token_offset + prime: '+',
            token_offset + prime + 1: '-',
            token_offset + prime + 2: '*',
            token_offset + prime + 3: '/',
        }
        eq = token_offset + prime + 4

        tokens = input_ids.tolist()
        result = []
        for t in tokens:
            if token_offset <= t < token_offset + prime:
                # Number token: subtract offset to get raw value
                result.append(str(t - token_offset))
            elif t in ops:
                result.append(f" {ops[t]} ")
            elif t == eq:
                result.append(" = ")
            else:
                result.append(f"[{t}]")

        return "".join(result)


def test_modular_arithmetic_dataset():
    """Quick test of the dataset."""
    print("Testing ModularArithmeticDataset...")

    ds = ModularArithmeticDataset(prime=97, operation="add", split="train")

    # Check a few examples
    for i in range(3):
        item = ds[i]
        decoded = ModularArithmeticDataset.decode_example(
            item['input_ids'], 97, ds.token_offset
        )
        print(f"  Example {i}: {decoded}")
        print(f"    Answer token: {item['answer'].item()} (raw: {item['answer'].item() - ds.token_offset})")
        print(f"    Task type: {item['task_type']}")

    print(f"  Vocab size: {ds.get_vocab_size()}")
    print("  ✓ Test passed!")


if __name__ == "__main__":
    test_modular_arithmetic_dataset()

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

    Token format:
        - Numbers 0 to prime-1 map directly to token IDs 0 to prime-1
        - Operation tokens: + = prime, - = prime+1, * = prime+2, / = prime+3
        - Equals token: = = prime+4
    """

    # Operation token offsets (added to prime)
    OP_ADD = 0
    OP_SUB = 1
    OP_MUL = 2
    OP_DIV = 3
    OP_EQ = 4

    def __init__(
        self,
        prime: int = 97,
        operation: Literal["add", "sub", "mul", "div"] = "add",
        split: Literal["train", "val", "test"] = "train",
        train_fraction: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize modular arithmetic dataset.

        Args:
            prime: The modulus for arithmetic (default 97, a prime)
            operation: Which operation to use
            split: train/val/test split
            train_fraction: Fraction of examples for training (rest split 50-50 val/test)
            seed: Random seed for reproducibility
        """
        self.prime = prime
        self.operation = operation
        self.split = split

        # Token IDs
        self.op_token = prime + {
            "add": self.OP_ADD,
            "sub": self.OP_SUB,
            "mul": self.OP_MUL,
            "div": self.OP_DIV,
        }[operation]
        self.eq_token = prime + self.OP_EQ

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

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single example.

        Returns:
            dict with:
                - input_ids: [a, op, b, eq, answer] tensor
                - answer: [answer] tensor (for easy loss computation)
                - task_type: "math" (for metric routing)
        """
        a, b = self.pairs[idx]

        # Compute answer based on operation
        if self.operation == "add":
            answer = (a + b) % self.prime
        elif self.operation == "sub":
            answer = (a - b) % self.prime
        elif self.operation == "mul":
            answer = (a * b) % self.prime
        elif self.operation == "div":
            # Modular division: a / b = a * b^(-1) mod p
            # b^(-1) mod p = b^(p-2) mod p (Fermat's little theorem)
            b_inv = pow(b, self.prime - 2, self.prime)
            answer = (a * b_inv) % self.prime

        # Create input sequence: [a, op, b, =, answer]
        input_ids = torch.tensor([a, self.op_token, b, self.eq_token, answer], dtype=torch.long)

        return {
            'input_ids': input_ids,
            'answer': torch.tensor([answer], dtype=torch.long),
            'task_type': 'math',
        }

    def get_vocab_size(self) -> int:
        """Return vocabulary size needed for this dataset."""
        # Numbers 0 to prime-1, plus 5 special tokens (+, -, *, /, =)
        return self.prime + 5

    @staticmethod
    def decode_example(input_ids: torch.Tensor, prime: int = 97) -> str:
        """Decode an example back to human-readable form."""
        ops = {prime: '+', prime + 1: '-', prime + 2: '*', prime + 3: '/'}
        eq = prime + 4

        tokens = input_ids.tolist()
        result = []
        for t in tokens:
            if t < prime:
                result.append(str(t))
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
        decoded = ModularArithmeticDataset.decode_example(item['input_ids'], 97)
        print(f"  Example {i}: {decoded}")
        print(f"    Answer: {item['answer'].item()}")
        print(f"    Task type: {item['task_type']}")

    print(f"  Vocab size: {ds.get_vocab_size()}")
    print("  ✓ Test passed!")


if __name__ == "__main__":
    test_modular_arithmetic_dataset()

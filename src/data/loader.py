"""
Data loading utilities for Atlas training.

Supports:
- Dolmino dataset (jsonl.zst format - Zstandard compressed JSONL)
- Streaming for large datasets
- Tokenization with T5/GPT-2 tokenizer
"""

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Optional, Dict, Any, Iterator
import json
import random

# Optional imports - will check at runtime
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


class DolminoDatasetZstd(IterableDataset):
    """
    Streaming dataset for Dolmino jsonl.zst files.

    Streams data directly from zstd-compressed JSONL files.
    Shuffles at the file level for efficiency.

    Args:
        data_dir: Path to dolmino data directory
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        data_dir: Path,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        if not HAS_ZSTD:
            raise ImportError("zstandard required: pip install zstandard")

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Find all jsonl.zst files
        self.files = list(self.data_dir.glob("**/*.jsonl.zst"))

        if not self.files:
            raise ValueError(f"No jsonl.zst files found in {data_dir}")

        print(f"Found {len(self.files)} jsonl.zst files")

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        # Shuffle files
        files = self.files.copy()
        random.shuffle(files)

        # Buffer for accumulating tokens
        token_buffer = []

        for file_path in files:
            try:
                # Open zstd compressed file
                dctx = zstd.ZstdDecompressor()
                with open(file_path, 'rb') as fh:
                    with dctx.stream_reader(fh) as reader:
                        # Read line by line
                        text_stream = reader.read().decode('utf-8')
                        lines = text_stream.strip().split('\n')

                        # Shuffle lines within file
                        random.shuffle(lines)

                        for line in lines:
                            if not line.strip():
                                continue

                            try:
                                data = json.loads(line)
                            except json.JSONDecodeError:
                                continue

                            # Get text - try common column names
                            text = None
                            for col in ["text", "content", "document"]:
                                if col in data:
                                    text = data[col]
                                    break

                            if text is None or not isinstance(text, str):
                                continue

                            # Tokenize
                            tokens = self.tokenizer.encode(text, add_special_tokens=False)
                            token_buffer.extend(tokens)

                            # Yield complete sequences
                            while len(token_buffer) >= self.max_seq_len:
                                sequence = token_buffer[:self.max_seq_len]
                                token_buffer = token_buffer[self.max_seq_len:]

                                yield {
                                    "input_ids": torch.tensor(sequence, dtype=torch.long),
                                    "labels": torch.tensor(sequence, dtype=torch.long),
                                }

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue


class DolminoDatasetFromFiles(IterableDataset):
    """
    Streaming dataset from a specific list of jsonl.zst files.

    Used for train/val split - each gets a separate set of files.
    """

    def __init__(
        self,
        files: list,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        if not HAS_ZSTD:
            raise ImportError("zstandard required: pip install zstandard")

        self.files = files
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _read_zstd_jsonl(self, file_path: Path) -> Iterator[dict]:
        """Read a zstd-compressed JSONL file line by line."""
        dctx = zstd.ZstdDecompressor()
        with open(file_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                # Read in chunks for memory efficiency
                text_buffer = ""
                while True:
                    chunk = reader.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        break
                    text_buffer += chunk.decode('utf-8')

                    # Process complete lines
                    while '\n' in text_buffer:
                        line, text_buffer = text_buffer.split('\n', 1)
                        if line.strip():
                            try:
                                yield json.loads(line)
                            except json.JSONDecodeError:
                                continue

                # Handle last line if no trailing newline
                if text_buffer.strip():
                    try:
                        yield json.loads(text_buffer)
                    except json.JSONDecodeError:
                        pass

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate over tokenized sequences."""
        # Shuffle files each epoch
        files = self.files.copy()
        random.shuffle(files)

        # Buffer for accumulating tokens
        token_buffer = []

        for file_path in files:
            try:
                for data in self._read_zstd_jsonl(file_path):
                    # Get text - try common column names
                    text = None
                    for col in ["text", "content", "document"]:
                        if col in data:
                            text = data[col]
                            break

                    if text is None or not isinstance(text, str):
                        continue

                    # Tokenize
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.extend(tokens)

                    # Yield complete sequences
                    while len(token_buffer) >= self.max_seq_len:
                        sequence = token_buffer[:self.max_seq_len]
                        token_buffer = token_buffer[self.max_seq_len:]

                        yield {
                            "input_ids": torch.tensor(sequence, dtype=torch.long),
                            "labels": torch.tensor(sequence, dtype=torch.long),
                        }

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue


class SimpleTextDataset(Dataset):
    """
    Simple in-memory dataset for smaller files.

    Good for testing and debugging.

    Args:
        texts: List of text strings
        tokenizer: HuggingFace tokenizer
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        texts: list,
        tokenizer,
        max_seq_len: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Tokenize all texts and concatenate
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

        # Split into sequences
        self.sequences = []
        for i in range(0, len(all_tokens) - max_seq_len, max_seq_len):
            self.sequences.append(all_tokens[i:i + max_seq_len])

        print(f"Created {len(self.sequences)} sequences")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "labels": torch.tensor(sequence, dtype=torch.long),
        }


def create_dataloaders(
    data_dir: Path,
    tokenizer_name: str = "google/t5-v1_1-base",
    max_seq_len: int = 4096,
    batch_size: int = 8,
    num_workers: int = 4,
    val_split: float = 0.20,  # 80/20 split
) -> tuple:
    """
    Create train and validation dataloaders with proper separation.

    Supports two directory structures:
    1. The Pile structure: train/*.jsonl.zst + val.jsonl.zst
    2. Dolmino structure: flat or nested directories with random split

    IMPORTANT: Train and val use DIFFERENT files to detect memorization.
    Split is done at the file level for streaming efficiency.

    Args:
        data_dir: Path to data directory
        tokenizer_name: HuggingFace tokenizer name
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of dataloader workers
        val_split: Fraction for validation (default 0.20 = 80/20 split)

    Returns:
        train_loader, val_loader, tokenizer
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers required: pip install transformers")
    if not HAS_ZSTD:
        raise ImportError("zstandard required: pip install zstandard")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = Path(data_dir)

    # Check for The Pile structure (train/ directory with shards)
    train_dir = data_dir / "train"
    val_file = data_dir / "val.jsonl.zst"

    if train_dir.exists() and list(train_dir.glob("*.jsonl.zst")):
        # The Pile structure detected
        train_files = sorted(train_dir.glob("*.jsonl.zst"))
        print(f"Detected Pile structure: {len(train_files)} train shards")

        if val_file.exists():
            val_files = [val_file]
            print(f"Using dedicated val.jsonl.zst for validation")
        else:
            # Fall back to splitting train files
            n_val = max(1, int(len(train_files) * val_split))
            random.seed(42)
            shuffled = train_files.copy()
            random.shuffle(shuffled)
            val_files = shuffled[:n_val]
            train_files = shuffled[n_val:]
            print(f"No val.jsonl.zst found, splitting train files")
    else:
        # Dolmino/flat structure - find all files and split
        all_files = sorted(data_dir.glob("**/*.jsonl.zst"))

        if not all_files:
            raise ValueError(f"No jsonl.zst files found in {data_dir}")

        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        shuffled_files = all_files.copy()
        random.shuffle(shuffled_files)

        # Split files by val_split ratio
        n_val = max(1, int(len(shuffled_files) * val_split))
        val_files = shuffled_files[:n_val]
        train_files = shuffled_files[n_val:]

    print(f"Data split: {len(train_files)} train files, {len(val_files)} val files")

    # Create separate datasets
    train_dataset = DolminoDatasetFromFiles(
        files=train_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    val_dataset = DolminoDatasetFromFiles(
        files=val_files,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,  # Single worker for validation
    )

    return train_loader, val_loader, tokenizer


def create_test_dataloader(
    vocab_size: int = 32100,
    seq_len: int = 4096,
    n_samples: int = 10000,
    batch_size: int = 8,
) -> DataLoader:
    """
    Create a synthetic dataloader for testing.

    Generates random token sequences - useful for testing model mechanics
    without loading real data.
    """
    class SyntheticDataset(Dataset):
        def __init__(self, vocab_size, seq_len, n_samples):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.n_samples = n_samples

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            # Random tokens
            tokens = torch.randint(0, self.vocab_size, (self.seq_len,))
            return {"input_ids": tokens, "labels": tokens}

    dataset = SyntheticDataset(vocab_size, seq_len, n_samples)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

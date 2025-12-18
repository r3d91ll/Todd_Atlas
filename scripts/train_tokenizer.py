#!/usr/bin/env python3
"""
Train a custom BPE tokenizer on Dolmino or The Pile data.

This creates a LLaMA-style tokenizer (SentencePiece BPE) trained on our actual data.
The tokenizer can be published alongside the model.

Usage:
    # Train on Dolmino (default)
    python scripts/train_tokenizer.py --vocab-size 32000 --output tokenizer/atlas_tokenizer

    # Train on The Pile
    python scripts/train_tokenizer.py --vocab-size 32000 --output tokenizer/atlas_tokenizer_pile \
        --data-dir datasets/raw/the_pile_hf --dataset-type pile

Reference: LLaMA tokenizer design
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterator
import random
import zstandard as zstd
import io

# Use sentencepiece for training (same as LLaMA)
import sentencepiece as spm
from transformers import LlamaTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors, decoders


def iterate_pile_files(data_dir: Path, max_files: int = 5) -> Iterator[str]:
    """Iterate over The Pile JSONL.zst files and yield text content.

    The Pile structure:
    - train/*.jsonl.zst (numbered shards: 00.jsonl.zst, 01.jsonl.zst, ...)
    - val.jsonl.zst
    - test.jsonl.zst

    Each record has: {"text": "...", "meta": {"pile_set_name": "..."}}
    """
    # Find train shards (sorted for reproducibility)
    train_dir = data_dir / "train"
    if train_dir.exists():
        train_files = sorted(train_dir.glob("*.jsonl.zst"))
    else:
        train_files = sorted(data_dir.glob("*.jsonl.zst"))

    # Filter to only real files (not LFS pointers - they're tiny)
    real_files = [f for f in train_files if f.stat().st_size > 1000]

    # Limit number of files
    real_files = real_files[:max_files]

    print(f"Found {len(train_files)} Pile shards, using {len(real_files)} (size > 1KB)")
    print(f"Processing from {data_dir}")

    for i, file_path in enumerate(real_files):
        file_size_gb = file_path.stat().st_size / (1024**3)
        print(f"  Processing shard {i+1}/{len(real_files)}: {file_path.name} ({file_size_gb:.1f} GB)")

        try:
            dctx = zstd.ZstdDecompressor()
            with open(file_path, 'rb') as f:
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    doc_count = 0
                    for line in text_stream:
                        try:
                            record = json.loads(line)
                            text = record.get('text', '')
                            if text and len(text) > 100:  # Skip very short texts
                                yield text
                                doc_count += 1
                                if doc_count % 100000 == 0:
                                    print(f"    Processed {doc_count:,} documents from {file_path.name}")
                        except json.JSONDecodeError:
                            continue
                    print(f"    Finished {file_path.name}: {doc_count:,} documents")
        except Exception as e:
            print(f"  Warning: Error reading {file_path}: {e}")
            continue


def iterate_dolmino_files(data_dir: Path, max_files: int = 1000) -> Iterator[str]:
    """Iterate over Dolmino JSONL files and yield text content.

    Handles:
    - Files in subdirectories (category folders)
    - Zstandard compressed files (.jsonl.zst)
    - Plain JSONL files (.jsonl)
    """
    # Find all jsonl files (compressed or not) recursively
    zst_files = list(data_dir.glob("**/*.jsonl.zst"))
    jsonl_files = list(data_dir.glob("**/*.jsonl"))
    all_files = zst_files + jsonl_files

    # Shuffle and limit files
    random.shuffle(all_files)
    all_files = all_files[:max_files]

    print(f"Found {len(zst_files)} .zst files, {len(jsonl_files)} .jsonl files")
    print(f"Processing {len(all_files)} files from {data_dir}")

    for i, file_path in enumerate(all_files):
        if i % 50 == 0:
            print(f"  Processing file {i}/{len(all_files)}: {file_path.name}")

        try:
            # Handle zstd compressed files
            if file_path.suffix == '.zst':
                dctx = zstd.ZstdDecompressor()
                with open(file_path, 'rb') as f:
                    with dctx.stream_reader(f) as reader:
                        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                        for line in text_stream:
                            try:
                                record = json.loads(line)
                                text = record.get('text', '')
                                if text and len(text) > 100:  # Skip very short texts
                                    yield text
                            except json.JSONDecodeError:
                                continue
            else:
                # Plain JSONL
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            text = record.get('text', '')
                            if text and len(text) > 100:
                                yield text
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"  Warning: Error reading {file_path}: {e}")
            continue


def create_training_corpus(data_dir: Path, output_file: Path, max_files: int = 1000, max_chars: int = 500_000_000, dataset_type: str = "dolmino"):
    """Create a text file for SentencePiece training."""
    print(f"Creating training corpus from {data_dir}")
    print(f"Dataset type: {dataset_type}")
    print(f"Max files: {max_files}, Max chars: {max_chars:,}")

    total_chars = 0
    total_docs = 0

    # Select iterator based on dataset type
    if dataset_type == "pile":
        iterator = iterate_pile_files(data_dir, max_files)
    else:
        iterator = iterate_dolmino_files(data_dir, max_files)

    with open(output_file, 'w', encoding='utf-8') as out:
        for text in iterator:
            # Write each document on its own line
            # Clean up whitespace
            text = ' '.join(text.split())
            out.write(text + '\n')

            total_chars += len(text)
            total_docs += 1

            if total_chars >= max_chars:
                print(f"Reached {max_chars:,} character limit")
                break

            if total_docs % 10000 == 0:
                print(f"  Processed {total_docs:,} documents, {total_chars:,} characters")

    print(f"Created corpus: {total_docs:,} documents, {total_chars:,} characters")
    print(f"Saved to: {output_file}")
    return output_file


def train_sentencepiece_tokenizer(
    corpus_file: Path,
    output_prefix: str,
    vocab_size: int = 32000,
):
    """Train a SentencePiece BPE tokenizer (LLaMA-style)."""
    print(f"\nTraining SentencePiece tokenizer...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Output prefix: {output_prefix}")

    # SentencePiece training arguments (similar to LLaMA)
    spm.SentencePieceTrainer.train(
        input=str(corpus_file),
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type='bpe',  # Byte-Pair Encoding (like LLaMA)

        # Character coverage (for handling rare characters)
        character_coverage=0.9995,

        # Special tokens (LLaMA-style)
        pad_id=0,           # <pad>
        unk_id=1,           # <unk>
        bos_id=2,           # <s> (beginning of sequence)
        eos_id=3,           # </s> (end of sequence)

        # Additional settings
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True,

        # Byte fallback for unknown characters
        byte_fallback=True,

        # Split digits (like LLaMA)
        split_digits=True,

        # Allow whitespace-only pieces
        allow_whitespace_only_pieces=True,

        # Normalization
        normalization_rule_name='identity',  # No normalization (preserve original text)
        remove_extra_whitespaces=False,

        # User-defined symbols (can add special tokens here)
        user_defined_symbols=[],
    )

    print(f"SentencePiece model saved to: {output_prefix}.model")
    print(f"SentencePiece vocab saved to: {output_prefix}.vocab")


def convert_to_huggingface(
    spm_model_path: str,
    output_dir: Path,
    tokenizer_name: str = "atlas-tokenizer",
):
    """Convert SentencePiece model to HuggingFace format."""
    print(f"\nConverting to HuggingFace format...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the SentencePiece model and create a LlamaTokenizerFast
    # This makes it compatible with HuggingFace transformers
    tokenizer = LlamaTokenizerFast(
        vocab_file=spm_model_path,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        add_bos_token=True,
        add_eos_token=False,
        legacy=False,
    )

    # Save in HuggingFace format
    tokenizer.save_pretrained(output_dir)

    # Add tokenizer config for publishing
    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizerFast",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "add_bos_token": True,
        "add_eos_token": False,
        "model_max_length": 4096,
        "clean_up_tokenization_spaces": False,
    }

    with open(output_dir / "tokenizer_config.json", 'w') as f:
        json.dump(tokenizer_config, f, indent=2)

    # Create a README for the tokenizer
    readme = f"""# {tokenizer_name}

A custom BPE tokenizer trained on Dolmino dataset for the Atlas language model.

## Training Details

- **Algorithm**: SentencePiece BPE (same as LLaMA)
- **Vocabulary Size**: {tokenizer.vocab_size}
- **Training Data**: Dolmino mix (subset of Dolma)
- **Special Tokens**: <pad>, <unk>, <s>, </s>

## Usage

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/{tokenizer_name}")

text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)
```

## Model Compatibility

This tokenizer is designed for:
- Atlas 50M/100M language models
- Any causal language model with vocab_size={tokenizer.vocab_size}

## License

Apache 2.0 (same as training data)
"""

    with open(output_dir / "README.md", 'w') as f:
        f.write(readme)

    print(f"HuggingFace tokenizer saved to: {output_dir}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    return tokenizer


def test_tokenizer(tokenizer, test_texts: list = None):
    """Test the tokenizer on sample texts."""
    if test_texts is None:
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "In 2024, artificial intelligence continued to advance rapidly.",
            "The equation E=mc^2 describes mass-energy equivalence.",
        ]

    print(f"\n{'='*60}")
    print("Tokenizer Test Results")
    print('='*60)

    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)

        print(f"\nOriginal: {text[:80]}{'...' if len(text) > 80 else ''}")
        print(f"Tokens ({len(tokens)}): {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Decoded: {decoded[:80]}{'...' if len(decoded) > 80 else ''}")
        print(f"Compression: {len(text)/len(tokens):.2f} chars/token")


def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer on Dolmino or Pile data")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/raw/dolma3/dolmino_mix_100B/data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tokenizer/atlas_tokenizer"),
        help="Output directory for tokenizer",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Vocabulary size (default: 32000, same as LLaMA)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2000,
        help="Maximum number of data files to use for training",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1_000_000_000,  # 1B characters
        help="Maximum characters for training corpus",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="dolmino",
        choices=["dolmino", "pile"],
        help="Dataset type: dolmino (flat dirs) or pile (train/*.jsonl.zst)",
    )
    parser.add_argument(
        "--skip-corpus",
        action="store_true",
        help="Skip corpus creation (use existing corpus file)",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    output_dir = project_root / args.output
    corpus_file = output_dir / "training_corpus.txt"
    spm_prefix = output_dir / "spm_model"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Atlas Tokenizer Training")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Dataset type: {args.dataset_type}")
    print(f"Output directory: {output_dir}")
    print(f"Vocab size: {args.vocab_size}")
    print(f"Max files: {args.max_files}")
    print(f"Max chars: {args.max_chars:,}")
    print()

    # Step 1: Create training corpus
    if not args.skip_corpus:
        create_training_corpus(
            data_dir=data_dir,
            output_file=corpus_file,
            max_files=args.max_files,
            max_chars=args.max_chars,
            dataset_type=args.dataset_type,
        )
    else:
        print(f"Using existing corpus: {corpus_file}")

    # Step 2: Train SentencePiece tokenizer
    train_sentencepiece_tokenizer(
        corpus_file=corpus_file,
        output_prefix=str(spm_prefix),
        vocab_size=args.vocab_size,
    )

    # Step 3: Convert to HuggingFace format
    tokenizer = convert_to_huggingface(
        spm_model_path=str(spm_prefix) + ".model",
        output_dir=output_dir,
        tokenizer_name="atlas-tokenizer",
    )

    # Step 4: Test tokenizer
    test_tokenizer(tokenizer)

    print(f"\n{'='*60}")
    print("Tokenizer training complete!")
    print(f"{'='*60}")
    print(f"\nTo use this tokenizer:")
    print(f"  from transformers import AutoTokenizer")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")
    print(f"\nTo publish to HuggingFace Hub:")
    print(f"  tokenizer.push_to_hub('your-username/atlas-tokenizer')")


if __name__ == "__main__":
    main()

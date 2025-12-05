#!/usr/bin/env python3
"""
Tokenize text data for Atlas training using GPT-2 tokenizer.

This script converts raw text files into tokenized binary format (.bin)
that can be efficiently loaded during training via memory mapping.

Usage:
    python tokenize_data.py --input /path/to/data.txt --output ./data/

The output directory will contain:
    - train.bin: Memory-mapped numpy array of uint16 tokens
    - metadata.json: Tokenization metadata (vocab size, token count, etc.)
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Tokenize text data for Atlas training")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input text file (or directory of .txt files)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for tokenized data",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="Tokenizer to use (default: gpt2)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Number of characters to process at a time",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.99,
        help="Train/val split ratio (default: 0.99 train)",
    )
    args = parser.parse_args()

    # Import tokenizers (do this after arg parsing for faster help)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers library required. Install with:")
        print("  pip install transformers")
        return 1

    # Load tokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {vocab_size}")

    # Ensure vocab fits in uint16
    if vocab_size > 65535:
        print(f"WARNING: vocab_size {vocab_size} > 65535, using uint32")
        dtype = np.uint32
    else:
        dtype = np.uint16

    # Collect input files
    input_path = Path(args.input)
    if input_path.is_file():
        text_files = [input_path]
    elif input_path.is_dir():
        text_files = sorted(input_path.glob("**/*.txt"))
        if not text_files:
            print(f"ERROR: No .txt files found in {input_path}")
            return 1
    else:
        print(f"ERROR: Input path not found: {input_path}")
        return 1

    print(f"Found {len(text_files)} text file(s)")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Tokenize all files
    all_tokens = []
    total_chars = 0

    for text_file in text_files:
        print(f"\nProcessing: {text_file}")
        file_size = text_file.stat().st_size
        print(f"  File size: {file_size / 1e6:.1f} MB")

        with open(text_file, "r", encoding="utf-8", errors="ignore") as f:
            # Process in chunks for memory efficiency
            pbar = tqdm(total=file_size, unit="B", unit_scale=True, desc="  Tokenizing")

            while True:
                chunk = f.read(args.chunk_size)
                if not chunk:
                    break

                # Tokenize chunk
                tokens = tokenizer.encode(chunk, add_special_tokens=False)
                all_tokens.extend(tokens)

                total_chars += len(chunk)
                pbar.update(len(chunk.encode("utf-8")))

            pbar.close()

    total_tokens = len(all_tokens)
    if total_tokens == 0:
        print("ERROR: No tokens produced from input files")
        return 1

    print(f"\nTotal characters: {total_chars:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Compression ratio: {total_chars / total_tokens:.2f} chars/token")

    # Convert to numpy array
    print("\nConverting to numpy array...")
    tokens_array = np.array(all_tokens, dtype=dtype)

    # Split into train/val
    split_idx = int(len(tokens_array) * args.split_ratio)
    train_tokens = tokens_array[:split_idx]
    val_tokens = tokens_array[split_idx:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")

    # Save as memory-mapped files
    train_path = output_path / "train.bin"
    val_path = output_path / "val.bin"

    print(f"\nSaving train.bin ({len(train_tokens) * dtype().itemsize / 1e6:.1f} MB)...")
    train_mmap = np.memmap(train_path, dtype=dtype, mode="w+", shape=(len(train_tokens),))
    train_mmap[:] = train_tokens
    train_mmap.flush()
    del train_mmap

    print(f"Saving val.bin ({len(val_tokens) * dtype().itemsize / 1e6:.1f} MB)...")
    val_mmap = np.memmap(val_path, dtype=dtype, mode="w+", shape=(len(val_tokens),))
    val_mmap[:] = val_tokens
    val_mmap.flush()
    del val_mmap

    # Save metadata
    metadata = {
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "dtype": str(dtype),
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "source_files": [str(f) for f in text_files],
        "total_chars": total_chars,
        "compression_ratio": total_chars / total_tokens,
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved metadata to {metadata_path}")
    print("\nTokenization complete!")
    print(f"  Output directory: {output_path}")
    print(f"  train.bin: {len(train_tokens):,} tokens")
    print(f"  val.bin: {len(val_tokens):,} tokens")

    return 0


if __name__ == "__main__":
    exit(main())

#!/usr/bin/env python3
"""
Download and prepare Shakespeare and Dumas corpuses from Project Gutenberg.

Outputs jsonl.zst format compatible with Atlas data loader.

Usage:
    python scripts/download_corpus.py --corpus shakespeare
    python scripts/download_corpus.py --corpus dumas
    python scripts/download_corpus.py --corpus all
"""

import argparse
import json
import requests
import re
from pathlib import Path
from typing import List, Tuple

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("Warning: zstandard not installed. Will output plain jsonl.")


# Project Gutenberg URLs
# Shakespeare: Complete Works
SHAKESPEARE_URLS = [
    # Complete Works of Shakespeare (various editions)
    ("https://www.gutenberg.org/cache/epub/100/pg100.txt", "complete_works"),
]

# Dumas: Major works in French
DUMAS_URLS = [
    # Les Trois Mousquetaires (The Three Musketeers)
    ("https://www.gutenberg.org/cache/epub/13951/pg13951.txt", "trois_mousquetaires"),
    # Vingt Ans Après (Twenty Years After)
    ("https://www.gutenberg.org/cache/epub/54111/pg54111.txt", "vingt_ans_apres"),
    # Le Vicomte de Bragelonne (sequel)
    ("https://www.gutenberg.org/cache/epub/2609/pg2609.txt", "vicomte_bragelonne_1"),
    ("https://www.gutenberg.org/cache/epub/2681/pg2681.txt", "vicomte_bragelonne_2"),
    # Le Comte de Monte-Cristo
    ("https://www.gutenberg.org/cache/epub/17989/pg17989.txt", "monte_cristo_1"),
    ("https://www.gutenberg.org/cache/epub/17990/pg17990.txt", "monte_cristo_2"),
    # La Reine Margot
    ("https://www.gutenberg.org/cache/epub/20319/pg20319.txt", "reine_margot"),
    # Les Quarante-Cinq
    ("https://www.gutenberg.org/cache/epub/54082/pg54082.txt", "quarante_cinq"),
]


def download_text(url: str) -> str:
    """Download text from URL with proper encoding handling."""
    print(f"  Downloading {url}...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Project Gutenberg uses UTF-8
    response.encoding = 'utf-8'
    return response.text


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate."""
    # Find start marker
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG",
        "*** START OF THE PROJECT GUTENBERG",
        "*END*THE SMALL PRINT!",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Find the end of the line after the marker
            end_line = text.find('\n', idx)
            if end_line != -1:
                start_idx = end_line + 1
                break

    # Find end marker
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG",
        "*** END OF THE PROJECT GUTENBERG",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
    ]

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx].strip()


def split_into_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """Split text into paragraphs, filtering short ones."""
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)

    # Clean and filter
    result = []
    for para in paragraphs:
        para = para.strip()
        # Normalize whitespace
        para = re.sub(r'\s+', ' ', para)
        if len(para) >= min_length:
            result.append(para)

    return result


def write_jsonl_zstd(paragraphs: List[str], output_path: Path, shard_size: int = 1000):
    """Write paragraphs to jsonl.zst format, optionally splitting into shards.

    Args:
        paragraphs: List of text paragraphs
        output_path: Output file path (will be modified for shards)
        shard_size: Number of paragraphs per shard (0 = no sharding)
    """
    if shard_size > 0 and len(paragraphs) > shard_size:
        # Split into multiple shards for proper train/val splitting
        n_shards = (len(paragraphs) + shard_size - 1) // shard_size
        for shard_idx in range(n_shards):
            start = shard_idx * shard_size
            end = min(start + shard_size, len(paragraphs))
            shard_paras = paragraphs[start:end]

            # Create shard filename
            stem = output_path.stem.replace('.jsonl', '')
            shard_path = output_path.parent / f"{stem}_shard{shard_idx:02d}.jsonl.zst"

            _write_single_file(shard_paras, shard_path)

        print(f"  Wrote {n_shards} shards with ~{shard_size} paragraphs each")
    else:
        _write_single_file(paragraphs, output_path)


def _write_single_file(paragraphs: List[str], output_path: Path):
    """Write paragraphs to a single jsonl.zst file."""
    jsonl_content = ""
    for para in paragraphs:
        jsonl_content += json.dumps({"text": para}, ensure_ascii=False) + "\n"

    if HAS_ZSTD:
        # Compress with zstandard
        cctx = zstd.ZstdCompressor(level=3)
        compressed = cctx.compress(jsonl_content.encode('utf-8'))
        with open(output_path, 'wb') as f:
            f.write(compressed)
        print(f"  Wrote {output_path.name} ({len(compressed):,} bytes)")
    else:
        # Fall back to plain jsonl
        plain_path = output_path.with_suffix('.jsonl')
        with open(plain_path, 'w', encoding='utf-8') as f:
            f.write(jsonl_content)
        print(f"  Wrote {plain_path.name} ({len(jsonl_content):,} bytes)")


def download_shakespeare(output_dir: Path):
    """Download and process Shakespeare corpus."""
    print("\nDownloading Shakespeare corpus...")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_paragraphs = 0
    total_chars = 0

    for url, name in SHAKESPEARE_URLS:
        try:
            text = download_text(url)
            text = clean_gutenberg_text(text)
            paragraphs = split_into_paragraphs(text)

            output_path = output_dir / f"{name}.jsonl.zst"
            write_jsonl_zstd(paragraphs, output_path)

            total_paragraphs += len(paragraphs)
            total_chars += sum(len(p) for p in paragraphs)

        except Exception as e:
            print(f"  Error downloading {name}: {e}")

    print(f"\nShakespeare corpus complete:")
    print(f"  Paragraphs: {total_paragraphs:,}")
    print(f"  Characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")


def download_dumas(output_dir: Path):
    """Download and process Dumas corpus (French)."""
    print("\nDownloading Dumas corpus (French)...")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_paragraphs = 0
    total_chars = 0

    for url, name in DUMAS_URLS:
        try:
            text = download_text(url)
            text = clean_gutenberg_text(text)
            paragraphs = split_into_paragraphs(text)

            output_path = output_dir / f"{name}.jsonl.zst"
            write_jsonl_zstd(paragraphs, output_path)

            total_paragraphs += len(paragraphs)
            total_chars += sum(len(p) for p in paragraphs)

        except Exception as e:
            print(f"  Error downloading {name}: {e}")

    print(f"\nDumas corpus complete:")
    print(f"  Paragraphs: {total_paragraphs:,}")
    print(f"  Characters: {total_chars:,}")
    print(f"  Estimated tokens: ~{total_chars // 4:,}")


def main():
    parser = argparse.ArgumentParser(description="Download corpus from Project Gutenberg")
    parser.add_argument(
        "--corpus",
        choices=["shakespeare", "dumas", "all"],
        default="all",
        help="Which corpus to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Output directory (default: data/)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Project Gutenberg Corpus Downloader")
    print("=" * 60)

    if not HAS_ZSTD:
        print("\nWARNING: zstandard not installed.")
        print("Install with: pip install zstandard")
        print("Falling back to uncompressed jsonl output.\n")

    if args.corpus in ["shakespeare", "all"]:
        download_shakespeare(args.output_dir / "shakespeare")

    if args.corpus in ["dumas", "all"]:
        download_dumas(args.output_dir / "dumas")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

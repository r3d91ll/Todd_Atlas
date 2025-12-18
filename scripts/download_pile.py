#!/usr/bin/env python3
"""
Download The Pile dataset from HuggingFace backup.

Uses the monology/pile-uncopyrighted backup which excludes potentially
infringing content (Books3, BookCorpus2, OpenSubtitles, YTSubtitles, OWT2).

The full Pile is ~825GB. We can download a subset for initial training.

Usage:
    # Download first 10GB worth of data (streaming)
    python scripts/download_pile.py --max-gb 10

    # Download specific subsets only
    python scripts/download_pile.py --subsets "ArXiv,Github,Wikipedia (en)"

Reference: https://huggingface.co/datasets/monology/pile-uncopyrighted
"""

import argparse
import json
from pathlib import Path
from datasets import load_dataset
import os

# Pile subsets available in the uncopyrighted version
PILE_SUBSETS = [
    "ArXiv",
    "DM Mathematics",
    "Enron Emails",
    "EuroParl",
    "FreeLaw",
    "Github",
    "Gutenberg (PG-19)",
    "HackerNews",
    "NIH ExPorter",
    "PhilPapers",
    "Pile-CC",
    "PubMed Abstracts",
    "PubMed Central",
    "StackExchange",
    "USPTO Backgrounds",
    "Ubuntu IRC",
    "Wikipedia (en)",
]


def main():
    parser = argparse.ArgumentParser(description="Download The Pile dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/raw/the_pile/data"),
        help="Output directory for downloaded data",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=50.0,
        help="Maximum GB to download (approximate)",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to download",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=None,
        help="Comma-separated list of subsets to include (default: all)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Which split to download",
    )
    parser.add_argument(
        "--use-full-pile",
        action="store_true",
        help="Use full Pile instead of uncopyrighted version",
    )
    args = parser.parse_args()

    # Choose dataset
    if args.use_full_pile:
        dataset_name = "monology/pile"
        print("Using full Pile dataset (includes potentially copyrighted content)")
    else:
        dataset_name = "monology/pile-uncopyrighted"
        print("Using Pile-uncopyrighted (legally safe)")

    # Setup output
    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.split}.jsonl"

    # Parse subsets filter
    subset_filter = None
    if args.subsets:
        subset_filter = set(s.strip() for s in args.subsets.split(","))
        print(f"Filtering to subsets: {subset_filter}")

    print(f"Downloading {dataset_name} ({args.split} split)")
    print(f"Output: {output_file}")
    print(f"Max size: {args.max_gb} GB")

    # Stream the dataset (don't download everything at once)
    print("\nLoading dataset (streaming mode)...")
    ds = load_dataset(
        dataset_name,
        split=args.split,
        streaming=True,
    )

    # Track progress
    total_bytes = 0
    total_docs = 0
    max_bytes = int(args.max_gb * 1024 * 1024 * 1024)

    subset_counts = {}

    print(f"\nStreaming to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for example in ds:
            # Check subset filter
            meta = example.get("meta", {})
            pile_set_name = meta.get("pile_set_name", "unknown")

            if subset_filter and pile_set_name not in subset_filter:
                continue

            # Track subset distribution
            subset_counts[pile_set_name] = subset_counts.get(pile_set_name, 0) + 1

            # Write document
            text = example.get("text", "")
            doc = {"text": text, "meta": meta}
            line = json.dumps(doc, ensure_ascii=False) + "\n"
            f.write(line)

            total_bytes += len(line.encode("utf-8"))
            total_docs += 1

            # Progress
            if total_docs % 10000 == 0:
                gb = total_bytes / (1024 ** 3)
                print(f"  Downloaded {total_docs:,} docs, {gb:.2f} GB")

            # Check limits
            if total_bytes >= max_bytes:
                print(f"\nReached size limit ({args.max_gb} GB)")
                break

            if args.max_docs and total_docs >= args.max_docs:
                print(f"\nReached document limit ({args.max_docs} docs)")
                break

    # Summary
    print(f"\n{'='*60}")
    print("Download Complete!")
    print(f"{'='*60}")
    print(f"Total documents: {total_docs:,}")
    print(f"Total size: {total_bytes / (1024**3):.2f} GB")
    print(f"Output file: {output_file}")
    print(f"\nSubset distribution:")
    for subset, count in sorted(subset_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total_docs
        print(f"  {subset}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 2: Manual Coherence Validation Tool

This tool presents prompts from the training corpus and asks the model
to generate continuations. A human evaluator rates the coherence.

Usage:
    python tools/manual_coherence_test.py --checkpoint path/to/checkpoint.pt --corpus shakespeare
    python tools/manual_coherence_test.py --checkpoint path/to/checkpoint.pt --corpus devega

The tool will:
1. Load the model from checkpoint
2. Sample random sentences from the corpus
3. Generate continuations
4. Prompt human for coherence rating (1-5)
5. Report pass/fail based on 80% threshold

If passed, saves a validation marker for Stage 3 to begin.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import torch
import zstandard as zstd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus_sentences(corpus_path: Path, max_sentences: int = 500) -> List[str]:
    """Load sentences from corpus shards."""
    sentences = []
    dctx = zstd.ZstdDecompressor()

    for shard in sorted(corpus_path.glob("*.jsonl.zst")):
        with open(shard, 'rb') as f:
            decompressed = dctx.decompress(f.read())
            for line in decompressed.decode('utf-8').strip().split('\n'):
                if line:
                    entry = json.loads(line)
                    text = entry.get('text', '')
                    # Split into sentences (simple heuristic)
                    for sent in text.replace('\n', ' ').split('. '):
                        sent = sent.strip()
                        if len(sent) > 20 and len(sent) < 200:
                            sentences.append(sent + '.')
                            if len(sentences) >= max_sentences:
                                return sentences

    return sentences


def load_model_and_tokenizer(checkpoint_path: Path, config_path: Path):
    """Load model from checkpoint."""
    import yaml
    from transformers import AutoTokenizer

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load tokenizer
    tokenizer_path = config.get('data', {}).get('tokenizer_path', 'tokenizer/atlas_multilingual')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Load model
    from src.model.atlas import Atlas, AtlasConfig

    model_config = AtlasConfig(
        d_model=config['model']['d_model'],
        n_layers=config['model']['n_layers'],
        n_heads=config['model']['n_heads'],
        d_ff=config['model']['d_ff'],
        vocab_size=config['model']['vocab_size'],
        max_seq_len=config['model']['max_seq_len'],
        d_key=config['model']['d_key'],
        d_value=config['model']['d_value'],
    )

    model = Atlas(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    device = config.get('training', {}).get('device', 'cuda:0')
    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_continuation(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 50,
    temperature: float = 0.8,
) -> str:
    """Generate a continuation of the prompt."""
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate
    with torch.no_grad():
        # Simple autoregressive generation
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Forward pass
            outputs = model(generated)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs

            # Get next token probabilities
            next_logits = logits[:, -1, :] / temperature

            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append
            generated = torch.cat([generated, next_token], dim=-1)

            # Stop at EOS or max length
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode (exclude prompt)
    continuation = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)

    return continuation.strip()


def run_validation_session(
    model,
    tokenizer,
    device: str,
    sentences: List[str],
    n_samples: int = 100,
    output_path: Optional[Path] = None,
) -> Tuple[float, List[dict]]:
    """Run interactive validation session."""

    print("\n" + "=" * 60)
    print("STAGE 2: Manual Coherence Validation")
    print("=" * 60)
    print(f"\nYou will evaluate {n_samples} generated continuations.")
    print("Rate each on a scale of 1-5:")
    print("  1 = Incoherent / nonsensical")
    print("  2 = Barely coherent / major issues")
    print("  3 = Somewhat coherent / noticeable issues")
    print("  4 = Mostly coherent / minor issues")
    print("  5 = Fully coherent / matches style")
    print("\nPress Enter to start, or 'q' to quit.\n")

    input()

    # Sample prompts
    random.shuffle(sentences)
    prompts = sentences[:n_samples]

    results = []
    ratings = []

    for i, prompt in enumerate(prompts):
        print(f"\n--- Sample {i+1}/{n_samples} ---")
        print(f"PROMPT: {prompt}")

        # Generate continuation
        try:
            continuation = generate_continuation(model, tokenizer, prompt, device)
        except Exception as e:
            print(f"  [Generation error: {e}]")
            continuation = "[ERROR]"

        print(f"CONTINUATION: {continuation}")

        # Get rating
        while True:
            rating_input = input("\nRating (1-5, 'q' to quit, 's' to skip): ").strip().lower()

            if rating_input == 'q':
                print("\nQuitting early...")
                break
            elif rating_input == 's':
                print("  Skipped")
                break
            elif rating_input in ['1', '2', '3', '4', '5']:
                rating = int(rating_input)
                ratings.append(rating)
                results.append({
                    'prompt': prompt,
                    'continuation': continuation,
                    'rating': rating,
                })
                break
            else:
                print("  Invalid input. Enter 1-5, 'q', or 's'.")

        if rating_input == 'q':
            break

    # Calculate results
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        pass_rate = sum(1 for r in ratings if r >= 4) / len(ratings)
    else:
        avg_rating = 0
        pass_rate = 0

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Samples evaluated: {len(ratings)}")
    print(f"Average rating: {avg_rating:.2f}/5")
    print(f"Pass rate (>=4): {pass_rate:.1%}")
    print(f"Threshold: 80%")
    print(f"Status: {'PASSED' if pass_rate >= 0.80 else 'FAILED'}")

    # Save results
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_samples': len(ratings),
                'avg_rating': avg_rating,
                'pass_rate': pass_rate,
                'passed': pass_rate >= 0.80,
                'results': results,
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return pass_rate, results


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Manual Coherence Validation")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to config YAML (auto-detected if not specified)")
    parser.add_argument('--corpus', type=str, choices=['shakespeare', 'devega'],
                        required=True, help="Which corpus to use")
    parser.add_argument('--n-samples', type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument('--output', type=str, default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    # Determine paths
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    # Auto-detect config if not specified
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(f"configs/atlas_{args.corpus}.yaml")

    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        return 1

    # Corpus path
    corpus_path = Path(f"data/{args.corpus}")
    if not corpus_path.exists():
        print(f"Error: Corpus not found: {corpus_path}")
        return 1

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        run_dir = checkpoint_path.parent.parent
        output_path = run_dir / "stage2_validation.json"

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {output_path}")

    # Load corpus
    print("\nLoading corpus sentences...")
    sentences = load_corpus_sentences(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    # Load model
    print("\nLoading model...")
    try:
        model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, config_path)
        print(f"  Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Run validation
    pass_rate, results = run_validation_session(
        model, tokenizer, device, sentences,
        n_samples=args.n_samples,
        output_path=output_path,
    )

    # Save stage transition marker if passed
    if pass_rate >= 0.80:
        marker_path = checkpoint_path.parent / "stage2_passed"
        marker_path.touch()
        print(f"\nStage 2 PASSED - marker saved: {marker_path}")
        print("Ready to proceed to Stage 3!")
        return 0
    else:
        print(f"\nStage 2 FAILED - continue Stage 1 training")
        return 1


if __name__ == "__main__":
    sys.exit(main())

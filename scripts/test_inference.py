#!/usr/bin/env python3
"""
Simple inference test for Atlas Omega models.
Tests if the model generates coherent responses.
"""

import argparse
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.atlas_omega import AtlasOmega, AtlasOmegaConfig
from transformers import AutoTokenizer


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config from checkpoint
    config = checkpoint.get("config")
    if isinstance(config, dict):
        config = AtlasOmegaConfig(**config)
    
    # Create model
    model = AtlasOmega(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config


def generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.7,
    device: str = "cuda",
):
    """Generate text from prompt."""
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    generated = input_ids.clone()
    memory_states = None
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Forward pass
            outputs = model(generated, memory_states=memory_states)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            
            # Get next token logits (last position)
            next_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text


def main():
    parser = argparse.ArgumentParser(description="Test Atlas model inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/atlas_tokenizer", help="Path to tokenizer")
    parser.add_argument("--prompt", type=str, default="A quick brown fox", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load model
    model, config = load_model(args.checkpoint, args.device)
    
    # Generate
    print(f"\nPrompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print("-" * 50)
    
    output = generate(
        model,
        tokenizer,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        device=args.device,
    )
    
    print(f"\nGenerated:\n{output}")
    print("-" * 50)


if __name__ == "__main__":
    main()

"""
Stage 3: Generative Memory Trainer

This trainer teaches the model to use memory for creative generation:
1. Sample a prompt from the corpus
2. Model queries memory for relevant past context
3. Model generates a continuation
4. Store (prompt, generation) pair in memory
5. Compute loss on generation quality + memory utilization

The key insight: Train the model to consult memory BEFORE generating,
making memory an active part of the creative process, not just retrieval.
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import zstandard as zstd


@dataclass
class GenerativeMemoryConfig:
    """Configuration for generative memory training."""
    # Training
    max_steps: int = 50000
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_steps: int = 500
    grad_clip: float = 1.0

    # Generation
    max_generation_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 50

    # Memory
    memory_query_before_generate: bool = True
    store_generations_in_memory: bool = True
    memory_query_top_k: int = 5

    # Loss weights
    lm_loss_weight: float = 1.0
    memory_utilization_weight: float = 0.1

    # Logging
    log_interval: int = 50
    checkpoint_interval: int = 1000
    metrics_path: str = "metrics_stream.jsonl"

    # Device
    device: str = "cuda:0"
    dtype: str = "bfloat16"


class GenerativeMemoryTrainer:
    """
    Stage 3 trainer: Memory-augmented generation.

    Unlike Stage 1 (masked word completion), this stage trains the model
    to actively use memory for creative generation. The model learns:
    - Query memory before generating
    - Generate coherent continuations informed by memory
    - Store generations for future reference
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        corpus_sentences: List[str],
        config: GenerativeMemoryConfig,
        checkpoint_dir: Path,
        stage1_checkpoint: Optional[Path] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.corpus_sentences = corpus_sentences
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Set device and dtype
        self.device = torch.device(config.device)
        self.dtype = getattr(torch, config.dtype)
        self.model = self.model.to(self.device)

        # Load Stage 1 checkpoint if provided
        if stage1_checkpoint:
            self.load_checkpoint(stage1_checkpoint)

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_steps,
            eta_min=config.learning_rate * 0.1,
        )

        # Metrics tracking
        self.metrics_path = self.checkpoint_dir / config.metrics_path
        self.step = 0
        self.generation_memory: List[Dict] = []  # Store (prompt, generation) pairs

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from Stage 1 checkpoint."""
        print(f"Loading Stage 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("  Stage 1 weights loaded successfully")

    def save_checkpoint(self, prefix: str = "stage3"):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{prefix}_step_{self.step}.pt"
        torch.save({
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'generation_memory_size': len(self.generation_memory),
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def query_memory(self, prompt: str, top_k: int = 5) -> List[Dict]:
        """
        Query memory for relevant past generations.

        This is the key innovation: the model learns to consult its
        generation history before creating new content.
        """
        if not self.generation_memory:
            return []

        # Simple similarity: find generations with similar prompts
        # In production, this would use embedding similarity
        prompt_words = set(prompt.lower().split())
        scored = []

        for entry in self.generation_memory:
            entry_words = set(entry['prompt'].lower().split())
            overlap = len(prompt_words & entry_words)
            scored.append((overlap, entry))

        # Return top-k most relevant
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def generate_with_memory(
        self,
        prompt: str,
        max_tokens: int = 64,
    ) -> Tuple[str, float]:
        """
        Generate continuation with memory consultation.

        Returns:
            Tuple of (generated_text, memory_utilization_score)
        """
        # 1. Query memory for context
        memory_context = []
        memory_util_score = 0.0

        if self.config.memory_query_before_generate:
            relevant = self.query_memory(prompt, self.config.memory_query_top_k)
            if relevant:
                # Include relevant past generations in context
                for entry in relevant:
                    memory_context.append(f"[MEMORY] {entry['generation']}")
                memory_util_score = len(relevant) / self.config.memory_query_top_k

        # 2. Build full prompt with memory context
        if memory_context:
            full_prompt = " ".join(memory_context) + " [PROMPT] " + prompt
        else:
            full_prompt = prompt

        # 3. Encode and generate
        input_ids = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_ids = input_ids.to(self.device)

        # Limit input length
        if input_ids.shape[1] > 256:
            input_ids = input_ids[:, -256:]

        # Generate autoregressively
        self.model.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(generated)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs

                # Get next token logits
                next_logits = logits[:, -1, :] / self.config.temperature

                # Top-k filtering
                if self.config.top_k > 0:
                    v, _ = torch.topk(next_logits, self.config.top_k)
                    next_logits[next_logits < v[:, [-1]]] = float('-inf')

                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat([generated, next_token], dim=-1)

                # Stop at EOS
                if next_token.item() == self.tokenizer.eos_token_id:
                    break

        # Decode continuation only
        continuation = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        return continuation, memory_util_score

    def compute_generation_loss(
        self,
        prompt: str,
        generated: str,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute loss for the generated text.

        Uses teacher forcing with the concatenation of prompt + generation
        as the target sequence.
        """
        # Combine prompt and generation
        full_text = prompt + " " + generated

        # Tokenize
        tokens = self.tokenizer.encode(full_text, return_tensors='pt')
        tokens = tokens.to(self.device)

        # Limit length
        if tokens.shape[1] > 512:
            tokens = tokens[:, :512]

        if tokens.shape[1] < 2:
            return torch.tensor(0.0, device=self.device), {}

        # Forward pass (training mode for dropout etc.)
        self.model.train()
        outputs = self.model(tokens[:, :-1])
        logits = outputs['logits'] if isinstance(outputs, dict) else outputs

        # Language modeling loss
        targets = tokens[:, 1:]
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id or 0,
        )

        metrics = {
            'lm_loss': lm_loss.item(),
            'seq_len': tokens.shape[1],
        }

        return lm_loss, metrics

    def train_step(self) -> Dict:
        """Execute one training step."""
        # 1. Sample random prompt from corpus
        prompt = random.choice(self.corpus_sentences)

        # 2. Generate with memory consultation
        generation, memory_util = self.generate_with_memory(
            prompt,
            max_tokens=self.config.max_generation_tokens,
        )

        # 3. Compute loss
        lm_loss, loss_metrics = self.compute_generation_loss(prompt, generation)

        # 4. Combined loss with memory utilization bonus
        # Lower loss when memory is utilized (encourages memory use)
        total_loss = (
            self.config.lm_loss_weight * lm_loss
            - self.config.memory_utilization_weight * memory_util
        )

        # 5. Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )

        self.optimizer.step()
        self.scheduler.step()

        # 6. Store generation in memory
        if self.config.store_generations_in_memory and len(generation) > 10:
            self.generation_memory.append({
                'step': self.step,
                'prompt': prompt,
                'generation': generation,
            })
            # Limit memory size
            if len(self.generation_memory) > 10000:
                self.generation_memory = self.generation_memory[-10000:]

        # 7. Collect metrics
        metrics = {
            'step': self.step,
            'loss': total_loss.item(),
            'lm_loss': loss_metrics.get('lm_loss', 0),
            'memory_util': memory_util,
            'memory_size': len(self.generation_memory),
            'lr': self.scheduler.get_last_lr()[0],
            'prompt_len': len(prompt),
            'gen_len': len(generation),
        }

        return metrics

    def log_metrics(self, metrics: Dict):
        """Log metrics to file."""
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def train(self):
        """Main training loop."""
        print("\n" + "=" * 60)
        print("STAGE 3: Generative Memory Training")
        print("=" * 60)
        print(f"Max steps: {self.config.max_steps}")
        print(f"Corpus sentences: {len(self.corpus_sentences)}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")

        start_time = time.time()

        while self.step < self.config.max_steps:
            self.step += 1

            # Training step
            metrics = self.train_step()

            # Logging
            if self.step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = self.step / elapsed

                print(
                    f"Step {self.step}/{self.config.max_steps} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"LM: {metrics['lm_loss']:.4f} | "
                    f"MemUtil: {metrics['memory_util']:.2f} | "
                    f"MemSize: {metrics['memory_size']} | "
                    f"Speed: {steps_per_sec:.1f} steps/s"
                )

                self.log_metrics(metrics)

            # Checkpointing
            if self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

        # Final checkpoint
        self.save_checkpoint(prefix="stage3_final")
        print("\nStage 3 training complete!")


def load_corpus_sentences(corpus_path: Path) -> List[str]:
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
                    # Split into sentences
                    for sent in text.replace('\n', ' ').split('. '):
                        sent = sent.strip()
                        if 20 < len(sent) < 200:
                            sentences.append(sent + '.')

    return sentences


def main():
    """CLI entry point."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Stage 3 Generative Memory Training")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument('--stage1-checkpoint', type=str, required=True,
                        help="Path to Stage 1 converged checkpoint")
    parser.add_argument('--output-dir', type=str, default=None,
                        help="Output directory (default: from config)")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Determine corpus path
    corpus_path = Path(config['data']['dataset_path'])
    tokenizer_path = config['data']['tokenizer_path']

    # Load corpus
    print("Loading corpus...")
    sentences = load_corpus_sentences(corpus_path)
    print(f"  Loaded {len(sentences)} sentences")

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Build model
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

    # Create trainer config
    curriculum_config = config.get('curriculum', {}).get('stage3', {})
    trainer_config = GenerativeMemoryConfig(
        max_steps=curriculum_config.get('max_steps', 50000),
        device=config['training']['device'],
        dtype=config['training']['dtype'],
        memory_query_before_generate=curriculum_config.get('memory_query_before_generate', True),
        store_generations_in_memory=curriculum_config.get('store_generations_in_memory', True),
    )

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config['training']['checkpoint_dir']).parent / "stage3"

    # Create trainer
    trainer = GenerativeMemoryTrainer(
        model=model,
        tokenizer=tokenizer,
        corpus_sentences=sentences,
        config=trainer_config,
        checkpoint_dir=output_dir,
        stage1_checkpoint=Path(args.stage1_checkpoint),
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()

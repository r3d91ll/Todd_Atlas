# Atlas 1B Build Plan

**Status:** Planning
**Created:** 2024-12-17
**Last Updated:** 2024-12-17
**Branch:** `1b-omega` (to be created after 300M evaluation)

---

## Overview

This document outlines all changes, additions, and improvements for the Atlas 1B model build, informed by lessons learned from the 300M training run.

### Goals

1. **Scale to 1B parameters** - Sufficient capacity for coherent text generation
2. **Full observability** - TCF metrics, hidden state capture, memory persistence
3. **Cloud-ready** - Optimized for H100/H200 deployment on RunPod
4. **Better notifications** - Telegram bot for real-time alerts
5. **Data quality tooling** - Batch provenance tracking for debugging

---

## 1. Model Architecture

### 1.1 Configuration Changes

| Parameter | 300M | 1B | Rationale |
|-----------|------|-----|-----------|
| `d_model` | 1024 | 1536 | Increased representation capacity |
| `n_layers` | 16 | 24 | Deeper network for complex patterns |
| `n_heads` | 8 | 12 | More attention heads |
| `d_ff` | 4096 | 6144 | Larger feedforward (4x d_model) |
| `d_key` | 1024 | 1536 | Match d_model |
| `d_value` | 1024 | 1536 | Match d_model |
| `vocab_size` | 32000 | 32000 | Unchanged |
| `max_seq_len` | 4096 | 4096 | Unchanged |

### 1.2 Memory Module

| Parameter | 300M | 1B | Notes |
|-----------|------|-----|-------|
| Memory matrix M | [B, 1024, 1024] | [B, 1536, 1536] | Larger associative memory |
| Surprise matrix S | [B, 1024, 1024] | [B, 1536, 1536] | Match M dimensions |
| `poly_degree` | 2 | 2 | Keep polynomial features |
| `context_window` | 16 | 16 | Keep context window |

### 1.3 Parameter Count Estimate

```
Embeddings:       32000 * 1536           =    49.2M
Attention (x24):  24 * 4 * 1536^2        =   226.5M
FFN (x24):        24 * 2 * 1536 * 6144   =   452.9M
Memory (x24):     24 * (projections)     =   ~150M
LayerNorms:       ~2M
LM Head:          1536 * 32000           =    49.2M
─────────────────────────────────────────────────────
Total:                                   ~   930M - 1.1B
```

### 1.4 Config File

Create `configs/atlas_1b_omega.yaml`:

```yaml
model:
  d_model: 1536
  n_layers: 24
  n_heads: 12
  d_ff: 6144
  vocab_size: 32000
  max_seq_len: 4096
  d_key: 1536
  d_value: 1536
  poly_degree: 2
  context_window: 16
  dropout: 0.1

training:
  batch_size: 4              # Larger for H100/H200
  grad_accum_steps: 32       # Effective batch = 4 * 32 * 4096 = 524K tokens
  learning_rate: 0.0001      # Slightly lower for larger model
  min_lr: 0.00001
  warmup_steps: 2000
  stage1_steps: 80000        # ~20B tokens
  stage2_steps: 8000         # ~2B tokens fine-tuning
  stage1_chunk_size: 2048
  stage2_chunk_size: 256
  val_interval: 500
  checkpoint_interval: 5000
  val_patience: 25

hardware:
  devices: [0]               # Single GPU
  distributed: false
  mixed_precision: true
  compile: true              # torch.compile for speed

data:
  tokenizer_path: "tokenizer/atlas_tokenizer"
  train_dirs:
    - "../datasets/raw/dolma3/dolmino_mix_100B/data"
  categories:
    - "ingredient1-common_crawl-high-quality_20_literature"
    - "ingredient1-common_crawl-high-quality_19_science_math_and_technology"
    - "ingredient1-code-meta-reasoning"
  val_split: 0.1

notifications:
  telegram:
    enabled: true
    bot_token: "${TELEGRAM_BOT_TOKEN}"
    chat_id: "${TELEGRAM_CHAT_ID}"
  events:
    - checkpoint_saved
    - validation_complete
    - training_complete
    - error_occurred
    - ppl_spike  # PPL > 2x rolling average
```

---

## 2. Observability & Interpretability

### 2.1 TCF Metrics Module

Create `src/training/tcf_metrics.py`:

```python
"""
Temporal Conveyance Framework metrics for geometric analysis.
"""

import torch
import numpy as np
from typing import Dict, Tuple

class TCFMetrics:
    """Compute TCF metrics on model tensors."""

    @staticmethod
    def effective_dimensionality(M: torch.Tensor) -> float:
        """
        Compute effective dimensionality via SVD.
        D_eff = (sum(sigma_i))^2 / sum(sigma_i^2)

        Higher D_eff = richer representation
        Target: >= 34 for 512D, scales with sqrt(D)
        """
        U, S, V = torch.linalg.svd(M.float())
        S_sum = S.sum()
        S_sq_sum = (S ** 2).sum()
        if S_sq_sum > 0:
            return (S_sum ** 2 / S_sq_sum).item()
        return 0.0

    @staticmethod
    def collapse_indicator(M: torch.Tensor, top_k: int = 10) -> float:
        """
        Compute collapse indicator (beta).
        beta = sigma_1 / mean(sigma_2:k)

        Lower beta = healthier representation
        Target: < 2.0
        """
        U, S, V = torch.linalg.svd(M.float())
        if len(S) < top_k:
            top_k = len(S)
        if top_k < 2:
            return 0.0

        sigma_1 = S[0]
        sigma_rest = S[1:top_k].mean()

        if sigma_rest > 0:
            return (sigma_1 / sigma_rest).item()
        return float('inf')

    @staticmethod
    def memory_curvature(M_history: list) -> float:
        """
        Estimate curvature of memory trajectory.
        Uses finite differences on memory state sequence.

        Args:
            M_history: List of memory states [M_t-2, M_t-1, M_t]
        """
        if len(M_history) < 3:
            return 0.0

        # First derivatives (velocities)
        v1 = M_history[1] - M_history[0]
        v2 = M_history[2] - M_history[1]

        # Second derivative (acceleration)
        a = v2 - v1

        # Curvature approximation
        v_norm = v2.norm()
        if v_norm > 0:
            return (a.norm() / (v_norm ** 2)).item()
        return 0.0

    @staticmethod
    def compute_all(
        M: torch.Tensor,
        S: torch.Tensor,
        hidden: torch.Tensor,
        M_history: list = None,
    ) -> Dict[str, float]:
        """Compute all TCF metrics."""

        # Take first batch element for analysis
        M_sample = M[0].detach()
        S_sample = S[0].detach()
        H_sample = hidden[0].detach()

        metrics = {
            # Memory metrics
            "memory/D_eff": TCFMetrics.effective_dimensionality(M_sample),
            "memory/beta": TCFMetrics.collapse_indicator(M_sample),
            "memory/norm": M_sample.norm().item(),

            # Surprise metrics
            "surprise/norm": S_sample.norm().item(),
            "surprise/D_eff": TCFMetrics.effective_dimensionality(S_sample),

            # Hidden state metrics
            "hidden/D_eff": TCFMetrics.effective_dimensionality(
                H_sample.view(-1, H_sample.shape[-1]).T  # [D, seq]
            ),
            "hidden/norm": H_sample.norm().item(),
        }

        # Trajectory curvature if history available
        if M_history and len(M_history) >= 3:
            metrics["memory/curvature"] = TCFMetrics.memory_curvature(M_history)

        return metrics
```

### 2.2 Hidden State Capture

Modify `src/model/atlas_omega.py` to return hidden states:

```python
def forward(
    self,
    x: torch.Tensor,
    memory_states: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    return_hidden: bool = False,
    return_memory: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Forward pass with optional hidden/memory state return.

    Args:
        x: Input token IDs [batch, seq_len]
        memory_states: Optional per-layer (M, S) tuples
        return_hidden: Return last hidden state
        return_memory: Return final memory states

    Returns:
        logits: [batch, seq_len, vocab_size]
        hidden: (optional) [batch, seq_len, d_model]
        memory_states: (optional) List of (M, S) tuples per layer
    """
    hidden = self.embedding(x)

    new_memory_states = []
    for i, block in enumerate(self.blocks):
        layer_memory = memory_states[i] if memory_states else None
        hidden, new_mem = block(hidden, memory_state=layer_memory)
        new_memory_states.append(new_mem)

    hidden = self.final_norm(hidden)
    logits = self.lm_head(hidden)

    # Build return tuple
    outputs = (logits,)
    if return_hidden:
        outputs += (hidden,)
    if return_memory:
        outputs += (new_memory_states,)

    return outputs if len(outputs) > 1 else outputs[0]
```

### 2.3 Memory State Persistence

Modify checkpoint saving in `src/training/trainer.py`:

```python
def save_checkpoint(self, step: int, is_best: bool = False):
    """Save checkpoint with full memory state."""

    checkpoint = {
        "step": step,
        "model_state_dict": self.model.state_dict(),
        "optimizer_state_dict": self.optimizer.state_dict(),
        "scheduler_state_dict": self.scheduler.state_dict(),
        "config": self.config,
        "metrics": {
            "loss": self.current_loss,
            "ppl": self.current_ppl,
            "best_val_loss": self.best_val_loss,
        },
        # NEW: Memory state persistence
        "memory_states": self.current_memory_states,  # List of (M, S) per layer
        "tcf_metrics": self.current_tcf_metrics,      # Latest TCF measurements
    }

    path = self.checkpoint_dir / f"checkpoint_{step}.pt"
    torch.save(checkpoint, path)
```

### 2.4 Geometric Snapshot Tool

Create `scripts/geometric_snapshot.py`:

```python
"""
Capture full geometric state of model at a checkpoint.
Useful for detailed TCF analysis.
"""

import torch
import json
from pathlib import Path
from src.model.atlas_omega import AtlasOmega
from src.training.tcf_metrics import TCFMetrics

def capture_snapshot(
    checkpoint_path: str,
    output_dir: str,
    sample_input: torch.Tensor,
):
    """
    Capture and save full geometric state.

    Saves:
    - Per-layer memory matrices (M, S)
    - Per-layer hidden states
    - SVD decompositions
    - TCF metrics
    """
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = AtlasOmega(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Forward pass with full state capture
    with torch.no_grad():
        logits, hidden, memory_states = model(
            sample_input,
            return_hidden=True,
            return_memory=True,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-layer analysis
    for i, (M, S) in enumerate(memory_states):
        layer_dir = output_dir / f"layer_{i}"
        layer_dir.mkdir(exist_ok=True)

        # Raw tensors
        torch.save(M, layer_dir / "M.pt")
        torch.save(S, layer_dir / "S.pt")

        # SVD decomposition
        U, sigma, V = torch.linalg.svd(M[0].float())
        torch.save({"U": U, "sigma": sigma, "V": V}, layer_dir / "M_svd.pt")

        # TCF metrics
        metrics = TCFMetrics.compute_all(M, S, hidden)
        with open(layer_dir / "tcf_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    # Save hidden state
    torch.save(hidden, output_dir / "hidden_state.pt")

    print(f"Geometric snapshot saved to {output_dir}")
```

---

## 3. Notifications (Telegram)

### 3.1 Telegram Module

Create `src/infrastructure/telegram.py`:

```python
"""
Telegram bot notifications for training events.
"""

import os
import requests
from typing import Optional
from datetime import datetime

class TelegramNotifier:
    """Send training notifications via Telegram."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            print("Telegram notifications disabled (missing token or chat_id)")

    def send(self, message: str, silent: bool = False) -> bool:
        """Send a message."""
        if not self.enabled:
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            response = requests.post(url, data={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "Markdown",
                "disable_notification": silent,
            }, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"Telegram send failed: {e}")
            return False

    def checkpoint_saved(self, step: int, loss: float, ppl: float):
        """Notify checkpoint saved."""
        msg = f"""
*Checkpoint Saved*
Step: `{step:,}`
Loss: `{loss:.4f}`
PPL: `{ppl:.1f}`
Time: {datetime.now().strftime("%H:%M:%S")}
        """.strip()
        self.send(msg)

    def training_complete(self, total_steps: int, final_ppl: float, duration_hours: float):
        """Notify training complete."""
        msg = f"""
*Training Complete*
Total Steps: `{total_steps:,}`
Final PPL: `{final_ppl:.1f}`
Duration: `{duration_hours:.1f}` hours
        """.strip()
        self.send(msg)

    def error_occurred(self, error: str, step: int):
        """Notify error occurred."""
        msg = f"""
*ERROR*
Step: `{step:,}`
Error: `{error[:200]}`
        """.strip()
        self.send(msg)

    def ppl_spike(self, step: int, ppl: float, avg_ppl: float):
        """Notify PPL spike detected."""
        msg = f"""
*PPL Spike Detected*
Step: `{step:,}`
Current PPL: `{ppl:.1f}`
Rolling Avg: `{avg_ppl:.1f}`
Ratio: `{ppl/avg_ppl:.1f}x`
        """.strip()
        self.send(msg)

    def stage_transition(self, stage: int, step: int):
        """Notify stage transition."""
        msg = f"""
*Stage Transition*
Now entering Stage `{stage}`
Step: `{step:,}`
        """.strip()
        self.send(msg)
```

### 3.2 Integration with Trainer

Add to training loop:

```python
# In trainer __init__
self.notifier = TelegramNotifier()

# On checkpoint save
self.notifier.checkpoint_saved(step, loss, ppl)

# On PPL spike (>2x rolling average)
if ppl > self.rolling_ppl_avg * 2:
    self.notifier.ppl_spike(step, ppl, self.rolling_ppl_avg)

# On error
except Exception as e:
    self.notifier.error_occurred(str(e), step)
    raise

# On completion
self.notifier.training_complete(total_steps, final_ppl, duration)
```

---

## 4. Data Quality & Debugging

### 4.1 Batch Provenance Tracking

Modify `src/data/loader.py`:

```python
def __iter__(self):
    """Yield batches with provenance metadata."""
    for file_idx, file_path in enumerate(self.files):
        with open(file_path, 'r') as f:
            for line_idx, line in enumerate(f):
                tokens = self.tokenizer.encode(line)

                yield {
                    "input_ids": tokens,
                    "attention_mask": [1] * len(tokens),
                    # Provenance metadata
                    "_source_file": str(file_path),
                    "_source_line": line_idx,
                    "_file_idx": file_idx,
                }
```

### 4.2 High-Loss Batch Logging

Add to training loop:

```python
# Track high-loss batches
HIGH_LOSS_THRESHOLD = 8.0  # Configurable
HIGH_LOSS_LOG = output_dir / "high_loss_batches.jsonl"

if loss.item() > HIGH_LOSS_THRESHOLD:
    record = {
        "step": step,
        "loss": loss.item(),
        "ppl": math.exp(loss.item()),
        "source_file": batch.get("_source_file"),
        "source_line": batch.get("_source_line"),
        "timestamp": datetime.now().isoformat(),
    }
    with open(HIGH_LOSS_LOG, "a") as f:
        json.dump(record, f)
        f.write("\n")

    logger.warning(f"High loss {loss.item():.2f} at {record['source_file']}:{record['source_line']}")
```

---

## 5. Cloud Deployment

### 5.1 RunPod Setup Script

Create `scripts/runpod_setup.sh`:

```bash
#!/bin/bash
# RunPod instance setup for Atlas 1B training

set -e

echo "=== Atlas 1B RunPod Setup ==="

# Update system
apt-get update && apt-get install -y git wget

# Clone repository
git clone https://github.com/YOUR_REPO/Atlas.git
cd Atlas

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Download tokenizer (if not in repo)
# python scripts/download_tokenizer.py

# Set environment variables
export TELEGRAM_BOT_TOKEN="your_token_here"
export TELEGRAM_CHAT_ID="your_chat_id_here"

echo "=== Setup Complete ==="
echo "Run: python scripts/train.py --config configs/atlas_1b_omega.yaml"
```

### 5.2 Training Launch Script

Create `scripts/train_1b_cloud.py`:

```python
"""
Launch script for 1B training on cloud GPU.
Handles setup, monitoring, and graceful shutdown.
"""

import os
import sys
import argparse
import signal
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/atlas_1b_omega.yaml")
    parser.add_argument("--resume", type=str, help="Checkpoint to resume from")
    parser.add_argument("--output-dir", default="runs/atlas_1b_omega")
    args = parser.parse_args()

    # Verify GPU
    import torch
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.0f}GB)")

    # Verify Telegram
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        print("Telegram notifications: ENABLED")
    else:
        print("Telegram notifications: DISABLED")

    # Start training
    from src.training.trainer import AtlasTrainer

    trainer = AtlasTrainer(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume,
    )

    # Graceful shutdown handler
    def shutdown_handler(signum, frame):
        print("\nGraceful shutdown requested...")
        trainer.save_checkpoint(trainer.current_step, emergency=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
```

### 5.3 Cost Monitoring

Add to notifications:

```python
def cost_update(self, hours_elapsed: float, hourly_rate: float = 2.39):
    """Send cost update (every 4 hours)."""
    cost = hours_elapsed * hourly_rate
    msg = f"""
*Cost Update*
Hours: `{hours_elapsed:.1f}`
Rate: `${hourly_rate}/hr`
Total: `${cost:.2f}`
    """.strip()
    self.send(msg, silent=True)
```

---

## 6. Training Strategy

### 6.1 Token Budget

| Approach | Tokens | Steps | Est. Time (H200) | Est. Cost |
|----------|--------|-------|------------------|-----------|
| Chinchilla (20x) | 20B | ~38K | ~22h | ~$79 |
| Extended (40x) | 40B | ~76K | ~44h | ~$158 |

**Recommendation:** Start with 20B tokens, evaluate, extend if needed.

### 6.2 Two-Stage Training (TNT)

| Stage | Steps | Chunk Size | Purpose |
|-------|-------|------------|---------|
| 1 | 80,000 | 2048 | Bulk pre-training |
| 2 | 8,000 | 256 | Fine-grained coherence |

### 6.3 Checkpointing Strategy

- Save every 5,000 steps
- Keep last 5 checkpoints + best validation
- Full geometric snapshot every 20,000 steps
- Emergency checkpoint on SIGTERM

---

## 7. Evaluation Plan

### 7.1 Inference Tests

Run at each checkpoint:
- "The quick brown fox" - coherence
- "Once upon a time" - narrative continuation
- "The capital of France is" - factual recall
- "def fibonacci(n):" - code generation
- Custom prompts relevant to use case

### 7.2 Success Criteria

| Metric | Target | Must Have |
|--------|--------|-----------|
| PPL | < 50 | < 100 |
| Coherent sentences | 3+ sentences | 2+ sentences |
| Topic maintenance | Yes | Mostly |
| No repetition loops | Yes | Minimal |
| Factual recall | Some | Not required |

### 7.3 Geometric Validation

- D_eff should increase then stabilize
- Beta should stay < 2.0
- Memory curvature should decrease (settling)
- Hidden state geometry should be rich

---

## 8. File Changes Summary

### New Files
- `configs/atlas_1b_omega.yaml`
- `src/training/tcf_metrics.py`
- `src/infrastructure/telegram.py`
- `scripts/geometric_snapshot.py`
- `scripts/runpod_setup.sh`
- `scripts/train_1b_cloud.py`

### Modified Files
- `src/model/atlas_omega.py` - Add return_hidden, return_memory
- `src/training/trainer.py` - Memory persistence, TCF logging, Telegram
- `src/data/loader.py` - Batch provenance tracking

### Branch Strategy
1. Create `1b-omega` branch from main after 300M evaluation
2. All 1B changes on this branch
3. Merge to main only after successful 1B training

---

## 9. Open Questions

_To be resolved after 300M evaluation:_

1. **Learning rate:** 0.0001 vs 0.00015 for 1B?
2. **Batch size:** 4 or 8 on H200 (141GB)?
3. **Data mix:** Same categories or expand?
4. **Token budget:** 20B sufficient or need 40B?
5. **Stage 2 length:** 8K steps enough?

---

## 10. Timeline

| Phase | Task | Duration |
|-------|------|----------|
| 1 | 300M evaluation & report | 1 day |
| 2 | Implement 1B changes | 1-2 days |
| 3 | Local testing (small run) | 0.5 day |
| 4 | Cloud deployment (H100/H200) | 1-2 days |
| 5 | Evaluation & iteration | 1 day |

**Total estimated:** 5-7 days to validated 1B model

---

## Appendix A: 300M Lessons Learned

_Updated: December 17, 2024 after 300M run completion_

### What Worked

1. **Training Pipeline** - Full 110K steps completed, TNT two-stage training functional
2. **Memory Architecture** - Titans/Miras implementation runs without errors
3. **Observability** - Per-layer metrics (M_norm, S_norm, gates) provide excellent visibility
4. **Checkpointing** - Regular saves enabled recovery from terminal disconnect
5. **Hardware Utilization** - 54% VRAM usage, stable 97-100% GPU utilization

### What Didn't Work

1. **Gate Collapse** - Gates collapsed from 50% → 0.7% over training
   - Model learned to BYPASS memory entirely
   - Final: 99.3% attention, 0.7% memory
   - **Critical fix needed for 1B**

2. **Text Coherence** - 389M insufficient for coherent generation
   - Broken grammar ("I't", "I're", "you'm")
   - Repetition loops ("and the same, and the other")
   - No topic maintenance

3. **Memory Utilization** - M matrix remained static (M_std constant at 0.0282)

4. **SMS Notifications** - 3-6 hour delays, unusable for monitoring

### Architecture Observations

1. **Gate Dynamics:**
   - Started balanced (50/50 attention/memory)
   - Progressively collapsed: 50% → 26% → 12% → 4% → 0.7%
   - Collapse accelerated after step 40K
   - Early layers (0, 5, 8, 14) retained slightly higher gates

2. **Surprise Accumulator:**
   - S_norm varied 600-3400 across layers
   - Gradients ARE flowing to memory
   - But collapsed gates block utilization

3. **Memory Matrix:**
   - M_norm clamped at 50.0 (hitting ceiling)
   - No structure developing (constant M_std)
   - May need to remove norm ceiling

### Data Quality Issues

1. **PPL Spikes:**
   - Step 45K: spike to 1,209
   - Step 90K: spike to 812
   - Likely malformed/noisy samples

2. **No Provenance:**
   - Cannot identify problematic batches
   - Need source file + line tracking

### Key Metrics at Completion

| Metric | Value |
|--------|-------|
| Final PPL | 229 |
| Final Loss | 5.43 |
| Avg Gate | 0.66% |
| Training Time | 38.9 hours |
| Total Tokens | 57.7B |

---

## Appendix B: Hardware Comparison

| GPU | VRAM | Cost/hr | 1B Time (20B tok) | Total Cost |
|-----|------|---------|-------------------|------------|
| A6000 (local) | 48GB | Free | ~120h | $0 |
| RTX 6000 Pro BW | 96GB | $1.84 | ~32h | ~$59 |
| H100 | 80GB | $2.39 | ~22h | ~$53 |
| H200 | 141GB | $3.59 | ~22h | ~$79 |
| B200 | 180GB | $6.00 | ~8h | ~$48 |

**Recommended:** H100 or H200 for best cost/availability balance.

# Atlas Memory Training - RunPod Template

Train and validate the Atlas memory architecture with real-time monitoring.

## What is Atlas?

Atlas is an implementation of the Titans/Miras memory framework from Google Research. It adds learnable memory to transformers, enabling models to store and retrieve information during inference.

**GitHub Repository:** [github.com/r3d91ll/Atlas](https://github.com/r3d91ll/Atlas)

## Quick Start

1. **Deploy this template** on an RTX 5090 (32GB VRAM)
2. **Wait ~2 minutes** for startup
3. **Open port 8501** to access the Streamlit dashboard
4. Training starts automatically with a 40M parameter test model

## What's Included

- **40M Parameter Model** - Small model for validation (~2-4 hours training)
- **Episodic Memory Training** - Prevents gate collapse through storage/retrieval cycles
- **Real-time Dashboard** - Monitor training metrics, memory health, gate values
- **Telegram Alerts** - Optional notifications for critical events

## Dashboard

Access the Streamlit dashboard at port 8501:

- **Live Training** - Loss curves, perplexity, training progress
- **Memory Deep Dive** - Memory magnitude, rank, surprise accumulators per layer
- **Alerts & Health** - Gate collapse risk, automatic anomaly detection

## Volume Mount

Mount a persistent volume to `/app/runs` to preserve:
- Model checkpoints
- Training metrics (JSONL)
- Dashboard data

**Important:** Do NOT mount to `/app` - this overwrites the container contents.

## Environment Variables (Optional)

### Hugging Face Upload (Recommended)
Automatically upload trained model to your private HF repo after training:
```
HF_TOKEN=hf_xxxxxxxxxxxx      # Required - your HF access token (write permission)
HF_USERNAME=your_username      # Required - your HF username
HF_REPO=atlas-40m-episodic     # Optional - repo name (default: atlas-40m-episodic)
```

After training, find your model at: `https://huggingface.co/YOUR_USERNAME/atlas-40m-episodic`

### Telegram Alerts
Get notified of training events:
```
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Hardware Requirements

| GPU | VRAM | Status |
|-----|------|--------|
| RTX 5090 | 32GB | Recommended |
| RTX 4090 | 24GB | May need batch_size reduction |
| H100 | 80GB | Use `hopper` tag instead |

## Configuration

Default settings optimized for RTX 5090:
- Batch size: 24
- Sequence length: 1024
- Training steps: 5000
- Gradient accumulation: 2

## Learn More

- **Full Documentation:** [github.com/r3d91ll/Atlas](https://github.com/r3d91ll/Atlas)
- **Architecture Details:** Based on [Titans](https://arxiv.org/abs/2501.00663) and [Miras](https://arxiv.org/abs/2501.01951) papers
- **Training Reports:** See the 300M training analysis in the GitHub repo

## Tags

| Tag | Purpose |
|-----|---------|
| `5090_40M` | RTX 5090 optimized (this template) |
| `hopper` | H100/H200 optimized |
| `latest` | Most recent build |

---

**Docker Image:** `ghcr.io/r3d91ll/atlas-training:5090_40M`

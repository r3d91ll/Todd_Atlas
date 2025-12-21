# Atlas Training - RunPod Deployment Guide

## Container Image

```
ghcr.io/r3d91ll/atlas-training:latest
ghcr.io/r3d91ll/atlas-training:v1.2.0
```

**Note**: Training data (1.2GB Dolmino literature corpus) is baked into the image - no download required.

## Quick Deploy on RunPod

### 1. Create a Pod

**GPU Selection:**
- **389M Full Training**: H100 80GB (recommended) or A100 80GB
- **40M Test**: Any GPU with 24GB+ VRAM (A6000, RTX 4090, etc.)

**Container Settings:**
- **Image**: `ghcr.io/r3d91ll/atlas-training:v1.2.0`
- **Docker Command**: See below based on training type

### 2. Environment Variables

Set these in RunPod's environment variables section:

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Optional | Your Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Optional | Your Telegram chat ID |
| `DATA_PATH` | Optional | Override data path (default: /data) |

### 3. Volume Mounts

**Training Data:**
Mount your training data to `/data` in the container.

For Dolmino data, the container expects:
```text
/data/
├── ingredient1-common_crawl-high-quality_19_science_math_and_technology/
│   ├── shard_00000000.jsonl.zst
│   ├── shard_00000001.jsonl.zst
│   └── ...
```

**Output Directory:**
Mount a persistent volume to `/app/runs` to save checkpoints and metrics.

### 4. Docker Commands

**40M Test Run (recommended first):**
```bash
./scripts/launch_40m_test.sh
```
- Duration: ~2-4 hours
- Steps: 5,000
- Purpose: Validate pipeline before spending on H100

**389M Full Training:**
```bash
./scripts/launch_389m_cloud.sh
```
- Duration: ~2 days on H100
- Steps: 57,000
- Purpose: Production episodic memory training

### 5. Expose Ports

| Port | Service |
|------|---------|
| 8501 | Streamlit Dashboard |

## RunPod Template Configuration

```json
{
  "name": "Atlas 389M Episodic Training",
  "imageName": "ghcr.io/r3d91ll/atlas-training:v1.2.0",
  "dockerArgs": "./scripts/launch_389m_cloud.sh",
  "gpuCount": 1,
  "gpuTypeId": "NVIDIA H100 80GB HBM3",
  "volumeInGb": 200,
  "containerDiskInGb": 50,
  "volumeMountPath": "/app",
  "ports": "8501/http",
  "env": [
    {"key": "TELEGRAM_BOT_TOKEN", "value": "YOUR_TOKEN"},
    {"key": "TELEGRAM_CHAT_ID", "value": "YOUR_CHAT_ID"}
  ]
}
```

## Monitoring

### Dashboard
Once the pod is running, access the Streamlit dashboard at:
```
https://<pod-id>-8501.proxy.runpod.net
```

### Telegram Alerts
If configured, you'll receive alerts for:
- Training start/completion
- Gate collapse risk > 80%
- Checkpoints saved
- Errors

### Logs
```bash
# Via RunPod console
cat /app/runs/atlas_389m_episodic/training.log

# Dashboard log
cat /app/runs/atlas_389m_episodic/dashboard.log
```

## Data

Training data (1.2GB Dolmino literature corpus) is baked into the Docker image at `/app/data/dolmino/`.
- 682 shards of compressed JSONL (zstd)
- Same dataset used in original 300M training run
- No download or data preparation required

## Troubleshooting

### Container fails to start
- Check if image pulled correctly: `docker pull ghcr.io/r3d91ll/atlas-training:v1.2.0`
- Verify GPU is available: `nvidia-smi`

### No data found
- Ensure data is mounted at `/data`
- Check DATA_PATH environment variable

### Dashboard not accessible
- Verify port 8501 is exposed
- Check dashboard log: `cat /app/runs/*/dashboard.log`

### OOM Errors
- 389M requires ~40GB VRAM - use H100 or A100
- 40M works on 24GB GPUs

## Expected Results

### 40M Test
- Gate mean: Should stay >50% (vs 0.7% collapse in baseline)
- Loss: ~5-6 at completion
- Collapse risk: Should stay <50%

### 389M Full Training
- Gate mean target: >20% at end
- Retrieval accuracy target: >90% during retrieval phases
- Perplexity target: <200

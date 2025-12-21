#!/bin/bash
# Train Atlas-Shakespeare (10M model) locally on A6000
#
# Expected training time: 1-2 hours
# GPU: cuda:1

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=============================================="
echo "ATLAS-SHAKESPEARE TRAINING"
echo "=============================================="
echo "Model: 10M parameters (d=256, L=4)"
echo "Corpus: Complete Works of Shakespeare (~1M tokens)"
echo "Device: cuda:1"
echo ""

# Create output directories
mkdir -p runs/atlas_shakespeare/checkpoints

# Launch training
python scripts/train_episodic.py \
    --config configs/atlas_shakespeare.yaml \
    --device cuda:1

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Checkpoints: runs/atlas_shakespeare/checkpoints/"
echo "Metrics: runs/atlas_shakespeare/metrics_stream.jsonl"

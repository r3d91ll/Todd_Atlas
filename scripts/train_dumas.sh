#!/bin/bash
# Train Atlas-Dumas (10M model) locally on A6000
#
# Expected training time: 1-2 hours
# GPU: cuda:1

set -e

cd "$(dirname "$0")/.."
source venv/bin/activate

echo "=============================================="
echo "ATLAS-DUMAS TRAINING (FRENCH)"
echo "=============================================="
echo "Model: 10M parameters (d=256, L=4)"
echo "Corpus: Alexandre Dumas collected works (~1.3M tokens)"
echo "Device: cuda:1"
echo ""

# Create output directories
mkdir -p runs/atlas_dumas/checkpoints

# Launch training
python scripts/train_episodic.py \
    --config configs/atlas_dumas.yaml \
    --device cuda:1

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo "Checkpoints: runs/atlas_dumas/checkpoints/"
echo "Metrics: runs/atlas_dumas/metrics_stream.jsonl"

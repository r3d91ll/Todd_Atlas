#!/bin/bash
# Atlas setup script

set -e

cd "$(dirname "$0")/.."

echo "=== Atlas Setup ==="

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch numpy pyyaml tqdm

# Optional but recommended
pip install transformers pyarrow

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate: source venv/bin/activate"
echo "To test:     python scripts/test_model.py"
echo "To train:    python scripts/train.py --config configs/atlas_50m.yaml"

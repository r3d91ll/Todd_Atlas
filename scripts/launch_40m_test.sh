#!/bin/bash
# Launch Atlas Dashboard (training started manually)
#
# Usage:
#   ./scripts/launch_40m_test.sh
#
# To start training, run in another terminal:
#   python scripts/train_episodic.py --config configs/atlas_40m_local_test.yaml

set -e

# Use absolute paths - handle both local and Docker environments
if [ -d "/app/scripts" ]; then
    PROJECT_DIR="/app"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment if it exists (local dev)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Ensure dependencies are installed
pip install streamlit plotly --quiet 2>/dev/null || true

# Create output directories
mkdir -p runs/atlas_40m_local/checkpoints
mkdir -p data

# Training data check
DATA_DIR="$PROJECT_DIR/data/dolmino"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "WARNING: Training data not found at $DATA_DIR"
    echo "Training will fail until data is available."
fi

# Set data path for training
export DATA_PATH="$DATA_DIR"

# Determine GPU device
DEVICE=${CUDA_DEVICE:-cuda:0}
if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
    DEVICE="cuda:0"
fi
export CUDA_DEVICE="$DEVICE"

echo "=============================================="
echo "Atlas Episodic Memory Training"
echo "=============================================="
echo "Project:   $PROJECT_DIR"
echo "Device:    $DEVICE"
echo "Dashboard: http://localhost:8501"
echo ""
echo "=============================================="
echo "MANUAL TRAINING MODE"
echo "=============================================="
echo ""
echo "Dashboard will start automatically."
echo "To start training, open a terminal and run:"
echo ""
echo "  cd /app && python scripts/train_episodic.py --config configs/atlas_40m_local_test.yaml"
echo ""
echo "Or use the helper script:"
echo ""
echo "  /app/scripts/start_training.sh"
echo ""
echo "=============================================="
echo ""

# Start Streamlit dashboard (foreground - keeps container alive)
echo "Starting dashboard..."
python -m streamlit run training_framework/monitoring/streamlit_monitor.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.enableWebsocketCompression false \
    --browser.gatherUsageStats false \
    -- \
    --metrics-path runs/atlas_40m_local/metrics_stream.jsonl \
    --max-steps 5000

# If dashboard exits, keep container alive
echo "Dashboard exited. Container will stay alive."
echo "Restart dashboard with: python -m streamlit run training_framework/monitoring/streamlit_monitor.py --server.port 8501 --server.address 0.0.0.0"
tail -f /dev/null

#!/bin/bash
# Launch Atlas 389M Episodic Training with Dashboard
# Designed for H100 cloud deployment (Docker container)
#
# Usage:
#   ./scripts/launch_389m_cloud.sh
#
# Environment variables (set in Docker or export before running):
#   TELEGRAM_BOT_TOKEN - Telegram bot token for alerts
#   TELEGRAM_CHAT_ID   - Telegram chat ID for alerts
#   CUDA_VISIBLE_DEVICES - GPU to use (default: 0)
#
# For Docker: Use this as ENTRYPOINT

set -e

# Use absolute paths - handle both local and Docker environments
if [ -d "/app/scripts" ]; then
    # Docker environment
    PROJECT_DIR="/app"
else
    # Local environment
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
fi
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Activate virtual environment (if exists)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Ensure dependencies
pip install streamlit plotly --quiet 2>/dev/null || true

# Create output directories
mkdir -p runs/atlas_389m_episodic
mkdir -p data

# Training data is baked into the image at data/dolmino/
# For 389M training, we cycle through the 1.2GB dataset multiple times
DATA_DIR="$PROJECT_DIR/data/dolmino"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    echo "This should not happen - data is baked into the Docker image."
    exit 1
fi

# Set data path for training
export DATA_PATH="$DATA_DIR"

# Write secrets from env vars if provided
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    cat > configs/secrets.yaml << EOF
telegram:
  bot_token: "$TELEGRAM_BOT_TOKEN"
  chat_id: "$TELEGRAM_CHAT_ID"
EOF
    echo "Telegram alerts configured from environment"
fi

echo "=============================================="
echo "Atlas 389M Episodic Memory Training"
echo "=============================================="
echo "Config: configs/atlas_389m_episodic.yaml"
echo "Device: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Data:   $DATA_PATH"
echo "Dashboard: http://0.0.0.0:8501"
echo ""

# Start Streamlit dashboard
# Proxy-friendly settings for RunPod and other cloud providers
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
    --metrics-path runs/atlas_389m_episodic/metrics_stream.jsonl \
    --max-steps 57000 \
    > runs/atlas_389m_episodic/dashboard.log 2>&1 &

DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID"
sleep 3

if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "Dashboard running"
else
    echo "WARNING: Dashboard failed"
    cat runs/atlas_389m_episodic/dashboard.log | tail -10
fi

echo ""
echo "Starting 389M training (~2 days on H100)..."
echo "=============================================="
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping dashboard..."
    kill $DASHBOARD_PID 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Start training
python scripts/train_episodic.py \
    --config configs/atlas_389m_episodic.yaml

echo ""
echo "Training complete!"

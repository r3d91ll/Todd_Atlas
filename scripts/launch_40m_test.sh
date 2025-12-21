#!/bin/bash
# Launch Atlas Local Test with Dashboard
# Designed for local GPU testing (Docker or native)
#
# Usage:
#   ./scripts/launch_40m_test.sh
#
# Environment variables:
#   TELEGRAM_BOT_TOKEN - Telegram bot token for alerts (optional)
#   TELEGRAM_CHAT_ID   - Telegram chat ID for alerts (optional)
#   CUDA_VISIBLE_DEVICES - GPU to use (default: 0)
#
# Note: Uses test_episodic_small.yaml (6M model) for fast validation

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
mkdir -p runs/test_episodic

# Check for data - use synthetic if not available
DATA_DIR="${DATA_PATH:-$PROJECT_DIR/data/dolmino}"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "No training data found at $DATA_DIR"
    echo "Using synthetic data for testing"
    export USE_SYNTHETIC=true
else
    export DATA_PATH="$DATA_DIR"
    echo "Using data from: $DATA_PATH"
fi

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
echo "Atlas Local Test (6M model)"
echo "=============================================="
echo "Config: configs/test_episodic_small.yaml"
echo "Device: ${CUDA_VISIBLE_DEVICES:-0}"
echo "Dashboard: http://0.0.0.0:8501"
echo ""

# Start Streamlit dashboard
echo "Starting dashboard..."
python -m streamlit run training_framework/monitoring/streamlit_monitor.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false \
    -- \
    --metrics-path runs/test_episodic/metrics_stream.jsonl \
    --max-steps 1000 \
    > runs/test_episodic/dashboard.log 2>&1 &

DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID"
sleep 3

if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "Dashboard running"
else
    echo "WARNING: Dashboard failed to start"
    cat runs/test_episodic/dashboard.log | tail -10
fi

echo ""
echo "Starting test training (~1-2 hours)..."
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
    --config configs/test_episodic_small.yaml

echo ""
echo "Training complete!"

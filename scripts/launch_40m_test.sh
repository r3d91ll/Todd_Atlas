#!/bin/bash
# Launch Atlas training with dashboard
# Designed for both local and Docker deployment
#
# Usage:
#   ./scripts/launch_40m_test.sh
#
# For Docker: This script is the ENTRYPOINT - starts everything automatically

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

# Activate virtual environment if it exists (local dev)
# In Docker, we don't need venv - packages are installed globally
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Ensure dependencies are installed
pip install streamlit plotly --quiet 2>/dev/null || true

# Create output directories
mkdir -p runs/atlas_40m_local
mkdir -p data

# Training data is baked into the image at data/dolmino/
DATA_DIR="$PROJECT_DIR/data/dolmino"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A $DATA_DIR 2>/dev/null)" ]; then
    echo "ERROR: Training data not found at $DATA_DIR"
    echo "This should not happen - data is baked into the Docker image."
    exit 1
fi

# Set data path for training
export DATA_PATH="$DATA_DIR"

# Determine GPU device (Docker uses cuda:0 via NVIDIA_VISIBLE_DEVICES mapping)
DEVICE=${CUDA_DEVICE:-cuda:1}
if [ -n "$NVIDIA_VISIBLE_DEVICES" ]; then
    DEVICE="cuda:0"  # Docker remaps GPU
fi

echo "=============================================="
echo "Atlas Episodic Memory Training"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Device:  $DEVICE"
echo "Dashboard: http://localhost:8501"
if [ -n "$DATA_PATH" ]; then
    echo "Data:    $DATA_PATH (from env)"
fi
echo ""

# Start Streamlit dashboard in background (headless for Docker)
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
    --metrics-path runs/atlas_40m_local/metrics_stream.jsonl \
    --max-steps 5000 \
    > runs/atlas_40m_local/dashboard.log 2>&1 &

DASHBOARD_PID=$!
echo "Dashboard PID: $DASHBOARD_PID"

# Wait for dashboard to start
sleep 3

# Verify dashboard is running
if kill -0 $DASHBOARD_PID 2>/dev/null; then
    echo "Dashboard started successfully"
else
    echo "WARNING: Dashboard may have failed to start"
    cat runs/atlas_40m_local/dashboard.log 2>/dev/null | tail -10
fi

echo ""
echo "Starting training..."
echo "=============================================="
echo ""

# Trap to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping dashboard (PID: $DASHBOARD_PID)..."
    kill $DASHBOARD_PID 2>/dev/null || true
    wait $DASHBOARD_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT INT TERM

# Start training (foreground - blocks until complete)
# Pass device override for Docker compatibility
export CUDA_DEVICE="$DEVICE"
python scripts/train_episodic.py \
    --config configs/atlas_40m_local_test.yaml

TRAINING_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Exit code: $TRAINING_EXIT_CODE"
echo ""

# If running on RunPod, stop the pod to save money
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "RunPod detected. Stopping pod in 60 seconds..."
    echo "Check your results at: /app/runs/atlas_40m_local/"
    echo "Checkpoint: /app/runs/atlas_40m_local/checkpoints/"
    echo "Metrics: /app/runs/atlas_40m_local/metrics_stream.jsonl"
    echo ""
    echo "To prevent shutdown, run: touch /tmp/keep_alive"
    echo ""

    # Give user 60 seconds to intervene
    for i in {60..1}; do
        if [ -f "/tmp/keep_alive" ]; then
            echo "Keep-alive file detected. Pod will NOT be stopped."
            echo "Remove /tmp/keep_alive and the pod will stop on next training completion."
            exit 0
        fi
        echo -ne "Stopping in $i seconds...\r"
        sleep 1
    done

    echo ""
    echo "Stopping RunPod pod: $RUNPOD_POD_ID"

    # Try runpodctl first (available in RunPod containers)
    if command -v runpodctl &> /dev/null; then
        runpodctl stop pod $RUNPOD_POD_ID || true
    fi

    # Also try the API directly as fallback
    if [ -n "$RUNPOD_API_KEY" ]; then
        curl -s -X POST "https://api.runpod.io/graphql" \
            -H "Content-Type: application/json" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -d "{\"query\": \"mutation { podStop(input: {podId: \\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
            || true
    fi

    echo "Stop command sent. Pod should power down shortly."
else
    echo "Not running on RunPod. Container will exit normally."
fi

exit $TRAINING_EXIT_CODE

#!/bin/bash
# Start Atlas training with optional HuggingFace upload
#
# Usage:
#   ./scripts/start_training.sh
#
# Environment variables:
#   HF_TOKEN      - HuggingFace token for upload (optional)
#   HF_USERNAME   - HuggingFace username (optional)
#   HF_REPO       - Repository name (default: atlas-40m-episodic)

set -e

cd /app

echo "=============================================="
echo "STARTING ATLAS TRAINING"
echo "=============================================="
echo ""

# Run training
python scripts/train_episodic.py --config configs/atlas_40m_local_test.yaml
TRAINING_EXIT_CODE=$?

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "=============================================="
echo "Exit code: $TRAINING_EXIT_CODE"
echo ""

# Upload to Hugging Face if configured
if [ -n "$HF_TOKEN" ]; then
    echo "=============================================="
    echo "UPLOADING TO HUGGING FACE"
    echo "=============================================="

    HF_REPO=${HF_REPO:-"atlas-40m-episodic"}
    HF_USERNAME=${HF_USERNAME:-""}

    if [ -z "$HF_USERNAME" ]; then
        echo "Detecting HF username from token..."
        HF_USERNAME=$(python -c "from huggingface_hub import whoami; print(whoami()['name'])" 2>/dev/null || echo "")
    fi

    if [ -n "$HF_USERNAME" ]; then
        FULL_REPO="$HF_USERNAME/$HF_REPO"
        echo "Uploading to: $FULL_REPO"
        echo ""

        python -c "
from huggingface_hub import HfApi, create_repo
import os

api = HfApi()
repo_id = '$FULL_REPO'

try:
    create_repo(repo_id, private=True, exist_ok=True)
    print(f'Repository ready: {repo_id}')
except Exception as e:
    print(f'Repo note: {e}')

checkpoint_dir = 'runs/atlas_40m_local/checkpoints'
if os.path.exists(checkpoint_dir):
    print('Uploading checkpoints...')
    api.upload_folder(folder_path=checkpoint_dir, repo_id=repo_id, path_in_repo='checkpoints')
    print('Checkpoints uploaded!')

metrics_file = 'runs/atlas_40m_local/metrics_stream.jsonl'
if os.path.exists(metrics_file):
    print('Uploading metrics...')
    api.upload_file(path_or_fileobj=metrics_file, path_in_repo='metrics_stream.jsonl', repo_id=repo_id)
    print('Metrics uploaded!')

config_file = 'configs/atlas_40m_local_test.yaml'
if os.path.exists(config_file):
    api.upload_file(path_or_fileobj=config_file, path_in_repo='config.yaml', repo_id=repo_id)
    print('Config uploaded!')

print(f'')
print(f'Upload complete! View at: https://huggingface.co/{repo_id}')
" 2>&1 || echo "WARNING: HuggingFace upload failed"
    else
        echo "ERROR: Could not determine HF_USERNAME. Set it as an environment variable."
    fi
else
    echo "HF_TOKEN not set. Skipping HuggingFace upload."
fi

echo ""
echo "=============================================="
echo "DONE - Dashboard remains available at port 8501"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  Checkpoints: /app/runs/atlas_40m_local/checkpoints/"
echo "  Metrics:     /app/runs/atlas_40m_local/metrics_stream.jsonl"
echo ""

exit $TRAINING_EXIT_CODE

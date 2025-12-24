#!/bin/bash
# =============================================================================
# Atlas Parallel Training Launcher
# =============================================================================
# Launches independent Shakespeare (GPU 0) and de Vega (GPU 1) training
# Each model follows the 3-stage curriculum:
#   Stage 1: Masked word completion → memory retrieval
#   Stage 2: Manual coherence validation (human-in-the-loop)
#   Stage 3: Memory-augmented generation → creative memory usage
#
# Usage:
#   ./scripts/launch_parallel_training.sh [stage]
#
# Arguments:
#   stage: 1, 2, or 3 (default: 1)
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
source venv/bin/activate

STAGE="${1:-1}"

echo "============================================================"
echo "Atlas Parallel Training Launcher"
echo "============================================================"
echo "Stage: $STAGE"
echo "Project: $PROJECT_DIR"
echo "============================================================"

case $STAGE in
    1)
        echo ""
        echo "STAGE 1: Masked Word Completion Training"
        echo "  Shakespeare → GPU 0 (cuda:0)"
        echo "  de Vega     → GPU 1 (cuda:1)"
        echo ""
        echo "Starting in 5 seconds... (Ctrl+C to cancel)"
        sleep 5

        # Create run directories
        mkdir -p runs/atlas_shakespeare/checkpoints
        mkdir -p runs/atlas_devega/checkpoints

        # Launch Shakespeare training (GPU 0)
        echo "Launching Shakespeare training..."
        nohup python -u scripts/train_episodic.py \
            --config configs/atlas_shakespeare.yaml \
            > runs/atlas_shakespeare/training.log 2>&1 &
        SHAKESPEARE_PID=$!
        echo "  PID: $SHAKESPEARE_PID"
        echo "  Log: runs/atlas_shakespeare/training.log"

        # Small delay to avoid GPU init conflicts
        sleep 3

        # Launch de Vega training (GPU 1)
        echo "Launching de Vega training..."
        nohup python -u scripts/train_episodic.py \
            --config configs/atlas_devega.yaml \
            > runs/atlas_devega/training.log 2>&1 &
        DEVEGA_PID=$!
        echo "  PID: $DEVEGA_PID"
        echo "  Log: runs/atlas_devega/training.log"

        # Save PIDs for monitoring
        echo "$SHAKESPEARE_PID" > runs/atlas_shakespeare/train.pid
        echo "$DEVEGA_PID" > runs/atlas_devega/train.pid

        echo ""
        echo "============================================================"
        echo "Training Started!"
        echo "============================================================"
        echo ""
        echo "Monitor with:"
        echo "  tail -f runs/atlas_shakespeare/training.log"
        echo "  tail -f runs/atlas_devega/training.log"
        echo ""
        echo "Dashboards:"
        echo "  Shakespeare: http://localhost:8501"
        echo "  de Vega:     http://localhost:8502"
        echo ""
        echo "Stop training:"
        echo "  kill $SHAKESPEARE_PID $DEVEGA_PID"
        echo ""
        echo "When Stage 1 converges, run Stage 2 validation:"
        echo "  ./scripts/launch_parallel_training.sh 2"
        ;;

    2)
        echo ""
        echo "STAGE 2: Manual Coherence Validation"
        echo ""
        echo "This stage requires human evaluation."
        echo "You will be prompted to rate generated continuations."
        echo ""

        # Find latest checkpoints
        SHAK_CKPT=$(ls -t runs/atlas_shakespeare/checkpoints/*.pt 2>/dev/null | head -1)
        VEGA_CKPT=$(ls -t runs/atlas_devega/checkpoints/*.pt 2>/dev/null | head -1)

        if [ -z "$SHAK_CKPT" ]; then
            echo "ERROR: No Shakespeare checkpoint found."
            echo "Complete Stage 1 first."
            exit 1
        fi

        if [ -z "$VEGA_CKPT" ]; then
            echo "ERROR: No de Vega checkpoint found."
            echo "Complete Stage 1 first."
            exit 1
        fi

        echo "Latest checkpoints:"
        echo "  Shakespeare: $SHAK_CKPT"
        echo "  de Vega:     $VEGA_CKPT"
        echo ""

        echo "Which model to validate?"
        echo "  1) Shakespeare"
        echo "  2) de Vega"
        echo "  3) Both (sequentially)"
        read -p "Choice [1/2/3]: " CHOICE

        case $CHOICE in
            1)
                echo ""
                echo "Validating Shakespeare model..."
                python tools/manual_coherence_test.py \
                    --checkpoint "$SHAK_CKPT" \
                    --corpus shakespeare \
                    --n-samples 100
                ;;
            2)
                echo ""
                echo "Validating de Vega model..."
                python tools/manual_coherence_test.py \
                    --checkpoint "$VEGA_CKPT" \
                    --corpus devega \
                    --n-samples 100
                ;;
            3)
                echo ""
                echo "Validating Shakespeare model..."
                python tools/manual_coherence_test.py \
                    --checkpoint "$SHAK_CKPT" \
                    --corpus shakespeare \
                    --n-samples 100

                echo ""
                echo "Validating de Vega model..."
                python tools/manual_coherence_test.py \
                    --checkpoint "$VEGA_CKPT" \
                    --corpus devega \
                    --n-samples 100
                ;;
            *)
                echo "Invalid choice"
                exit 1
                ;;
        esac

        echo ""
        echo "If both models passed, proceed to Stage 3:"
        echo "  ./scripts/launch_parallel_training.sh 3"
        ;;

    3)
        echo ""
        echo "STAGE 3: Memory-Augmented Generation Training"
        echo ""

        # Check for Stage 2 validation markers
        SHAK_PASSED=false
        VEGA_PASSED=false

        if [ -f "runs/atlas_shakespeare/checkpoints/stage2_passed" ]; then
            SHAK_PASSED=true
            echo "  Shakespeare: Stage 2 PASSED ✓"
        else
            echo "  Shakespeare: Stage 2 NOT PASSED ✗"
        fi

        if [ -f "runs/atlas_devega/checkpoints/stage2_passed" ]; then
            VEGA_PASSED=true
            echo "  de Vega: Stage 2 PASSED ✓"
        else
            echo "  de Vega: Stage 2 NOT PASSED ✗"
        fi

        if [ "$SHAK_PASSED" = false ] || [ "$VEGA_PASSED" = false ]; then
            echo ""
            echo "WARNING: Not all models passed Stage 2 validation!"
            read -p "Continue anyway? [y/N]: " CONTINUE
            if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
                echo "Aborting. Complete Stage 2 validation first."
                exit 1
            fi
        fi

        # Find Stage 1 converged checkpoints
        SHAK_CKPT=$(ls -t runs/atlas_shakespeare/checkpoints/stage1_converged*.pt 2>/dev/null | head -1)
        VEGA_CKPT=$(ls -t runs/atlas_devega/checkpoints/stage1_converged*.pt 2>/dev/null | head -1)

        # Fallback to latest checkpoint
        if [ -z "$SHAK_CKPT" ]; then
            SHAK_CKPT=$(ls -t runs/atlas_shakespeare/checkpoints/*.pt 2>/dev/null | head -1)
        fi
        if [ -z "$VEGA_CKPT" ]; then
            VEGA_CKPT=$(ls -t runs/atlas_devega/checkpoints/*.pt 2>/dev/null | head -1)
        fi

        echo ""
        echo "Using checkpoints:"
        echo "  Shakespeare: $SHAK_CKPT"
        echo "  de Vega:     $VEGA_CKPT"
        echo ""
        echo "Starting Stage 3 in 5 seconds... (Ctrl+C to cancel)"
        sleep 5

        # Create Stage 3 output directories
        mkdir -p runs/atlas_shakespeare/stage3
        mkdir -p runs/atlas_devega/stage3

        # Launch Shakespeare Stage 3 (GPU 0)
        echo "Launching Shakespeare Stage 3..."
        nohup python -u src/training/generative_memory_trainer.py \
            --config configs/atlas_shakespeare.yaml \
            --stage1-checkpoint "$SHAK_CKPT" \
            --output-dir runs/atlas_shakespeare/stage3 \
            > runs/atlas_shakespeare/stage3.log 2>&1 &
        SHAKESPEARE_PID=$!
        echo "  PID: $SHAKESPEARE_PID"

        sleep 3

        # Launch de Vega Stage 3 (GPU 1)
        echo "Launching de Vega Stage 3..."
        nohup python -u src/training/generative_memory_trainer.py \
            --config configs/atlas_devega.yaml \
            --stage1-checkpoint "$VEGA_CKPT" \
            --output-dir runs/atlas_devega/stage3 \
            > runs/atlas_devega/stage3.log 2>&1 &
        DEVEGA_PID=$!
        echo "  PID: $DEVEGA_PID"

        echo ""
        echo "============================================================"
        echo "Stage 3 Training Started!"
        echo "============================================================"
        echo ""
        echo "Monitor with:"
        echo "  tail -f runs/atlas_shakespeare/stage3.log"
        echo "  tail -f runs/atlas_devega/stage3.log"
        echo ""
        echo "Stop training:"
        echo "  kill $SHAKESPEARE_PID $DEVEGA_PID"
        ;;

    *)
        echo "Usage: $0 [stage]"
        echo "  stage: 1, 2, or 3"
        exit 1
        ;;
esac

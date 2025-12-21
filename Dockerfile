# Atlas Episodic Memory Training
# Docker image for GPU training with real-time monitoring
#
# Build:   docker build -t atlas-training .
# Run:     docker run --gpus all -p 8501:8501 -v /path/to/data:/data atlas-training
#
# For Telegram alerts, pass environment variables:
#   -e TELEGRAM_BOT_TOKEN=xxx -e TELEGRAM_CHAT_ID=xxx

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY training_framework/ training_framework/
COPY configs/ configs/
COPY scripts/ scripts/

# Copy training data (baked in for self-contained deployment)
COPY data/dolmino/ data/dolmino/

# Create directories for outputs
RUN mkdir -p runs/atlas_40m_local runs/atlas_389m_episodic

# Set data path
ENV DATA_PATH=/app/data/dolmino

# Make scripts executable
RUN chmod +x scripts/*.sh

# Expose Streamlit dashboard port
EXPOSE 8501

# Default environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/app

# Health check for dashboard
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default: run 40M local test
# Override with: docker run ... atlas-training /app/scripts/launch_389m_cloud.sh
WORKDIR /app
ENTRYPOINT ["/bin/bash"]
CMD ["/app/scripts/launch_40m_test.sh"]

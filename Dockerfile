FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Set environment variables to prevent CUDA installation
ENV TORCH_CUDA_ARCH_LIST=""
ENV PYTORCH_CUDA_ARCH_LIST=""
ENV USE_CUDA=0
ENV USE_CUDNN=0
ENV USE_MKLDNN=1

# Copy dependency files first
COPY pyproject.toml uv.lock* ./

# Install minimal system dependencies for PyTorch CPU
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first
RUN uv add "torch>=2.8.0" "torchvision>=0.23.0" --extra-index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN uv sync --frozen --no-dev

# Clean up UV cache to reduce image size
RUN uv cache clean

# Copy application code
COPY . .

EXPOSE 8000

CMD ["uv", "run", "python", "-m", "api"]
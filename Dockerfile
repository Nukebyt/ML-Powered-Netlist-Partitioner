# Dockerfile for ML-guided FM partitioning demo
# - CPU-only image (no CUDA). If you need GPU support, change base image and install CUDA toolkits.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

# System deps (minimal) for pip, SSL and wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel

# Install pure-Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch & torchvision & torchaudio from official PyTorch index
# Then install PyTorch Geometric and its companion libs using the wheel index that
# matches the installed torch version. This avoids building heavy packages from source.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch" "torchvision" "torchaudio" \
 && TORCH_VERSION=$(python - <<'PY'
import torch, re
v = torch.__version__
v = re.sub(r'\+.*$', '', v)
print(v)
PY
) \
 && pip install --no-cache-dir --find-links https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html \
    torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric || \
    (echo "PyG install failed - check compatibility of torch/${TORCH_VERSION} and PyG; see https://pytorch-geometric.readthedocs.io" && exit 1)

# Copy app sources
COPY . /app

EXPOSE 8501

# Run the Streamlit app (bind to 0.0.0.0 so it's accessible from outside container)
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]

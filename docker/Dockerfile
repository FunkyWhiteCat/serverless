# syntax=docker/dockerfile:1.6
#
# Worker image for Serverless ComfyUI + Qwen-Image 2512.
# Model weights live on a RunPod Network Volume at /runpod-volume —
# they are NOT baked into this image.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    VOLUME_ROOT=/runpod-volume/ComfyUI \
    APP_ROOT=/app \
    PYTHONPATH=/app/src

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        python3 \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI at a pinned commit. Any commit from January 2026 or newer
# supports Qwen-Image 2512 natively; replace with a specific SHA once
# we've picked one from the upstream history.
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && cd /opt/ComfyUI \
    && git checkout ${COMFYUI_REF}

# Install PyTorch for CUDA 12.4 first so ComfyUI's requirements don't
# pull a default CPU/CUDA-mismatched wheel, then the rest of its deps,
# then the handler-only packages.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchvision torchaudio \
    && pip install -r /opt/ComfyUI/requirements.txt \
    && pip install runpod websocket-client

# ComfyUI reads extra_model_paths.yaml from its own directory on startup.
COPY docker/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

# Application code (handler + workflow JSON). Keep repo layout at /app
# so src/workflows.py's default path resolution still works.
COPY src        /app/src
COPY workflows  /app/workflows
RUN chmod +x /app/src/start.sh

WORKDIR /app
CMD ["/app/src/start.sh"]

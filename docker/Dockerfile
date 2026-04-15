# syntax=docker/dockerfile:1.6
#
# Worker image for Serverless ComfyUI + Qwen-Image 2512.
# Model weights live on a RunPod Network Volume at /runpod-volume —
# they are NOT baked into this image.
#
# CUDA 13 is required for Blackwell-generation GPUs (B200 / GB200,
# compute capability sm_100). On cu124 PyTorch, ComfyUI prints
# "You need pytorch with cu130 or higher to use optimized CUDA
# operations." and silently falls back to PTX/JIT-compiled kernels,
# which leaves ~30–40 % of steady-state throughput on the table.
# Staying on ubuntu22.04 keeps Python at 3.10 so ComfyUI's
# requirements.txt doesn't need re-validation against 3.12.
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu22.04

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

# Install PyTorch for CUDA 13.0 first so ComfyUI's requirements don't
# pull a default CPU/CUDA-mismatched wheel, then the rest of its deps,
# then the handler-only packages.
#
# The PyTorch cu130 index is set as the primary so `torch` resolves
# to a cu130 build, and PyPI is added as a fallback index so the
# cuda-bindings transitive dep (which is missing from whl/cu130; see
# pytorch/pytorch#172926) can still be found.
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
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

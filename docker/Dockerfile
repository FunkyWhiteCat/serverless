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
#
# Using the -devel- variant (not -runtime-) because we need nvcc +
# CUDA headers at build time to compile sageattention from source —
# no pre-built wheel for the torch 2.11 / cu130 combo exists on PyPI
# yet. Image size is ~3 GB larger than runtime; acceptable for a
# serverless worker where the image is pulled once per physical host
# and then reused across warm invocations.
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

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
        python3-dev \
        build-essential \
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
#
# sageattention is installed *after* torch so its setup.py can detect
# the already-installed torch version. ComfyUI's --use-sage-attention
# flag (wired in start.sh) will route the Qwen-Image UNet attention
# through sage's fused fp8/bf16 kernels — on H200 this typically
# shaves another 10–20 % off each KSampler step vs. plain pytorch
# SDPA. --no-build-isolation lets the source build (if pip falls
# through to one) reuse the system torch instead of pulling a
# fresh CPU-only wheel into an isolated env.
RUN pip install --upgrade pip \
    && pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch torchvision torchaudio \
    && pip install -r /opt/ComfyUI/requirements.txt \
    && pip install runpod websocket-client \
    && pip install --no-build-isolation sageattention

# ComfyUI reads extra_model_paths.yaml from its own directory on startup.
COPY docker/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

# Application code (handler + workflow JSON). Keep repo layout at /app
# so src/workflows.py's default path resolution still works.
COPY src        /app/src
COPY workflows  /app/workflows
RUN chmod +x /app/src/start.sh

WORKDIR /app
CMD ["/app/src/start.sh"]

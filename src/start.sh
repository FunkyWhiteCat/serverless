#!/usr/bin/env bash
# Launches ComfyUI on localhost and hands control to the RunPod handler.
set -euo pipefail

VOLUME_ROOT="${VOLUME_ROOT:-/runpod-volume/ComfyUI}"
COMFY_HOST="${COMFY_HOST:-127.0.0.1}"
COMFY_PORT="${COMFY_PORT:-8188}"

# Ensure every directory referenced by extra_model_paths.yaml exists on
# the volume. ComfyUI's prestartup phase eagerly calls os.listdir() on
# the custom_nodes path and will FileNotFoundError if it's missing, and
# the other model dirs are scanned at load time — create them all up
# front so a freshly hydrated volume (which only has the three weight
# files) doesn't trip any startup probe. mkdir -p is idempotent.
mkdir -p "${VOLUME_ROOT}/input" \
         "${VOLUME_ROOT}/output" \
         "${VOLUME_ROOT}/temp" \
         "${VOLUME_ROOT}/custom_nodes" \
         "${VOLUME_ROOT}/models/checkpoints" \
         "${VOLUME_ROOT}/models/diffusion_models" \
         "${VOLUME_ROOT}/models/text_encoders" \
         "${VOLUME_ROOT}/models/vae" \
         "${VOLUME_ROOT}/models/loras" \
         "${VOLUME_ROOT}/models/clip_vision" \
         "${VOLUME_ROOT}/models/controlnet" \
         "${VOLUME_ROOT}/models/upscale_models"

# Tell ComfyUI to use the volume for input/output/temp directly via its
# own CLI flags, instead of symlinking /opt/ComfyUI/{input,output,temp}
# onto the volume. ComfyUI's git clone already ships a real /opt/ComfyUI/
# output directory, so `ln -sfn` would silently create the link INSIDE
# it instead of replacing it, and SaveImage would write to container-
# local disk where the handler can't reach it.
python3 /opt/ComfyUI/main.py \
    --listen "${COMFY_HOST}" \
    --port "${COMFY_PORT}" \
    --disable-auto-launch \
    --disable-metadata \
    --output-directory "${VOLUME_ROOT}/output" \
    --input-directory  "${VOLUME_ROOT}/input" \
    --temp-directory   "${VOLUME_ROOT}/temp" &

# The handler blocks on runpod.serverless.start(...) so this never returns.
exec python3 -u /app/src/handler.py

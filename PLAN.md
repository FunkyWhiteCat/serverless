# Plan — Serverless ComfyUI on RunPod with Network Volume (Qwen-Image 2512)

Goal: stand up a RunPod Serverless endpoint that runs ComfyUI and generates
images with **Qwen-Image 2512**, where all model weights live on a
**Network Volume** (not baked into the Docker image).

The image stays small and boots fast; swapping models/LoRAs is a volume
operation, not an image rebuild.

---

## 1. Architecture

```
         ┌──────────────────────────────┐
  client │  HTTPS POST /run (workflow)  │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  RunPod Serverless Endpoint  │
         │  ┌────────────────────────┐  │
         │  │ Worker container       │  │
         │  │  • handler.py (runpod) │  │
         │  │  • ComfyUI (127.0.0.1) │  │
         │  └──────────┬─────────────┘  │
         └─────────────┼────────────────┘
                       │  bind-mount
                       ▼
         ┌──────────────────────────────┐
         │  Network Volume              │
         │  /runpod-volume/ComfyUI/...  │
         │    models/ (Qwen-Image 2512) │
         │    custom_nodes/             │
         │    input/  output/           │
         └──────────────────────────────┘
```

Key choices:
- **Network Volume holds weights, custom_nodes, input/, output/.** The image
  holds only ComfyUI code + Python deps + handler. Cold start reads from the
  volume; no 20 GB image pulls.
- **Datacenter pinning.** Serverless workers can only attach a Network Volume
  in the same datacenter. Pick one region (e.g. `EU-RO-1` or `US-KS-2`) that
  has the GPU SKUs we want, and create both the volume and the endpoint
  there.
- **Handler is the official pattern** from
  [`runpod-workers/worker-comfyui`](https://github.com/runpod-workers/worker-comfyui)
  (formerly `blib-la/runpod-worker-comfy`): accept `{workflow, images?}`,
  start ComfyUI on localhost, POST to `/prompt`, poll `/history/<id>`, return
  base64 images (or push to S3/R2 if configured).

---

## 2. Network Volume layout

Create a Network Volume sized for the Qwen-Image 2512 bundle + headroom for
LoRAs and output. Qwen-Image fp8 ≈ 20 GB, fp16 ≈ 40 GB, text encoder ≈ 8 GB,
VAE < 1 GB. **Start at 100 GB**; resize later if needed.

Mount path inside the worker: `/runpod-volume`.

```
/runpod-volume/
└── ComfyUI/
    ├── models/
    │   ├── diffusion_models/
    │   │   └── qwen_image_2512_fp8_e4m3fn.safetensors
    │   ├── text_encoders/
    │   │   └── qwen_2.5_vl_7b_fp8_scaled.safetensors
    │   ├── vae/
    │   │   └── qwen_image_vae.safetensors
    │   └── loras/
    │       └── Qwen-Image-Lightning-8steps.safetensors   # optional
    ├── custom_nodes/
    │   └── ComfyUI-Manager/                              # optional, nice to have
    ├── input/
    └── output/
```

> Filenames are placeholders. Before going live, pin the exact file list from
> `Comfy-Org/Qwen-Image_ComfyUI` on Hugging Face for the **2512** release and
> record the SHA256 of each file in `models.lock` (see §6).

### How to populate the volume

Two options, in order of preference:

1. **One-time pod.** Launch a cheap on-demand pod in the same datacenter,
   attach the volume, `huggingface-cli download` the 2512 files directly into
   `/runpod-volume/ComfyUI/models/...`, then terminate the pod. Weights
   persist.
2. **First-boot downloader.** Have the handler check for a sentinel file
   (`/runpod-volume/ComfyUI/.ready`) on start and, if missing, download the
   models before serving requests. Simpler but makes the first cold start on
   a fresh volume multi-minute. Keep this as a fallback, not the primary
   path.

---

## 3. Container image

Small Dockerfile. No weights inside.

```Dockerfile
# docker/Dockerfile
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    COMFY_PORT=8188

RUN apt-get update && apt-get install -y --no-install-recommends \
        git python3 python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ComfyUI at a pinned commit that supports Qwen-Image 2512 natively
ARG COMFYUI_REF=master
RUN git clone https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI \
    && cd /opt/ComfyUI && git checkout ${COMFYUI_REF}

RUN pip install --upgrade pip \
    && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip install -r /opt/ComfyUI/requirements.txt \
    && pip install runpod websocket-client

# Point ComfyUI at the Network Volume for all model dirs + I/O
COPY docker/extra_model_paths.yaml /opt/ComfyUI/extra_model_paths.yaml

# Handler
COPY src/handler.py       /opt/handler.py
COPY src/start.sh         /opt/start.sh
RUN chmod +x /opt/start.sh

WORKDIR /opt
CMD ["/opt/start.sh"]
```

`docker/extra_model_paths.yaml` — the single source of truth for model paths:

```yaml
runpod_volume:
  base_path: /runpod-volume/ComfyUI
  checkpoints:       models/checkpoints
  diffusion_models:  models/diffusion_models
  text_encoders:     models/text_encoders
  vae:               models/vae
  loras:             models/loras
  clip_vision:       models/clip_vision
  custom_nodes:      custom_nodes
```

`src/start.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Bind ComfyUI output/input to the volume so outputs persist if needed
mkdir -p /runpod-volume/ComfyUI/output /runpod-volume/ComfyUI/input
ln -sfn /runpod-volume/ComfyUI/output /opt/ComfyUI/output
ln -sfn /runpod-volume/ComfyUI/input  /opt/ComfyUI/input

# Start ComfyUI in the background, bind to localhost only
python3 /opt/ComfyUI/main.py \
    --listen 127.0.0.1 --port "${COMFY_PORT}" \
    --disable-auto-launch --disable-metadata &

# Hand control to the RunPod handler (it blocks on the SDK runtime)
exec python3 -u /opt/handler.py
```

---

## 4. Handler (`src/handler.py`)

Thin wrapper around ComfyUI's HTTP + websocket API. Responsibilities:

1. Wait for `http://127.0.0.1:8188/system_stats` to return 200 (ComfyUI ready).
2. Accept `event["input"] = { "workflow": {...}, "images": [{name, image_b64}]? }`.
3. Write any provided input images into `/runpod-volume/ComfyUI/input/`.
4. `POST /prompt` with the workflow + a generated `client_id`.
5. Subscribe to the ws `/ws?clientId=...` and wait for `executing` → `None`
   (done) or surface errors.
6. For each `SaveImage` output in `/history/<prompt_id>`, read the file from
   `/runpod-volume/ComfyUI/output/<subfolder>/<filename>` and return as
   base64 (or upload to S3 if `BUCKET_ENDPOINT_URL` is set — RunPod handler
   SDK supports this natively).
7. Return `{ "images": [...], "prompt_id": ..., "timings": {...} }`.

Failure modes the handler must cover:
- ComfyUI never comes up → fail the job with a clear error, not a timeout.
- Workflow validation error from `/prompt` → return the node-level error dict.
- Missing model file on the volume → detect via the error text and return a
  dedicated `MODEL_NOT_FOUND` error so the caller knows to re-provision the
  volume rather than retry.

Reuse `runpod-workers/worker-comfyui`'s handler as a starting point — don't
reinvent the websocket plumbing.

---

## 5. Qwen-Image 2512 workflow

Ship a canonical text-to-image workflow at `workflows/qwen_image_2512_t2i.json`.
It mirrors the reference workflow that Comfy-Org publishes for Qwen-Image:

Nodes:
1. `UNETLoader` → `qwen_image_2512_fp8_e4m3fn.safetensors`, weight dtype
   `fp8_e4m3fn`
2. `CLIPLoader` → `qwen_2.5_vl_7b_fp8_scaled.safetensors`, type `qwen_image`
3. `VAELoader` → `qwen_image_vae.safetensors`
4. `ModelSamplingAuraFlow` (shift ≈ 3.1 — tune per release notes)
5. `CLIPTextEncode` × 2 (positive / negative)
6. `EmptySD3LatentImage` (1328×1328 default for Qwen-Image)
7. `KSampler` — sampler `euler`, scheduler `simple`, steps 20, cfg 2.5
   (or steps 8, cfg 1.0 when the Lightning LoRA is stacked via `LoraLoader`)
8. `VAEDecode` → `SaveImage`

Expose a "fast" variant that inserts `LoraLoader` with
`Qwen-Image-Lightning-8steps` and drops `KSampler` to 8 steps — that's the
difference between a ~30 s job and a ~10 s job on a 4090-class GPU.

The workflow JSON is what clients POST in `input.workflow`; they override
prompt text, seed, and size via the node inputs. Keep a Python helper in
`src/workflows.py` that loads the JSON and patches those fields so callers
don't have to hand-edit node IDs.

---

## 6. Model provenance (`models.lock`)

Commit a `models.lock` file listing every weight file we expect on the
volume, with:

- HF repo + revision
- filename
- sha256
- bytes

The one-time populator script (`scripts/populate_volume.py`) reads this file,
downloads each entry with `huggingface_hub.hf_hub_download`, verifies the
hash, and places it under the correct `models/<kind>/` subdir. Same script
runs as the fallback first-boot downloader in §2.

This is what makes "Qwen-Image 2512" reproducible — the version is the lock
file, not a tag in our heads.

---

## 7. RunPod endpoint configuration

Create the Serverless endpoint via the RunPod web console (or Terraform
provider if we want it in code later):

| Setting             | Value                                                           |
|---------------------|-----------------------------------------------------------------|
| Container image     | `ghcr.io/<org>/comfyui-qwen-image:<sha>` (built in CI)          |
| Container disk      | 20 GB (code + pip cache, no weights)                            |
| GPU types           | `RTX 4090`, `L40S`, `A100 40GB` — prefer 24 GB+ for fp8        |
| Min workers         | 0                                                               |
| Max workers         | 3 to start                                                      |
| Idle timeout        | 5 s                                                             |
| FlashBoot           | **on** — this is the big cold-start win                         |
| Execution timeout   | 300 s                                                           |
| Network Volume      | attach the volume created in §2 (same datacenter)               |
| Env vars            | `COMFY_PORT=8188`, optional `BUCKET_*` for S3 output            |

GPU sizing notes:
- fp8 Qwen-Image fits on a 24 GB card with room for a reasonable batch.
- fp16 wants 48 GB+; only worth it if quality regressions from fp8 show up.
- Qwen 2.5-VL text encoder is ~8 GB on its own; don't try to run this on a
  16 GB card.

---

## 8. Repo layout

```
.
├── PLAN.md                         (this file)
├── docker/
│   ├── Dockerfile
│   └── extra_model_paths.yaml
├── src/
│   ├── handler.py
│   ├── start.sh
│   └── workflows.py
├── workflows/
│   ├── qwen_image_2512_t2i.json
│   └── qwen_image_2512_t2i_lightning.json
├── scripts/
│   ├── populate_volume.py          # one-shot volume hydrator
│   └── smoke_test.py               # hits /runsync with a tiny prompt
├── models.lock
└── .github/workflows/
    └── build-and-push.yml          # GHCR build on tag
```

---

## 9. Build → deploy → verify loop

1. **Build.** `docker buildx build --platform linux/amd64 -t ghcr.io/<org>/comfyui-qwen-image:$(git rev-parse --short HEAD) .` and push.
   CI does this on tag so the image digest is pinned.
2. **Hydrate the volume** (first time only, or when `models.lock` changes):
   rent a cheap pod in the target DC with the volume attached, run
   `python scripts/populate_volume.py --lock models.lock`, terminate the pod.
3. **Update endpoint** to point at the new image tag. FlashBoot warms it.
4. **Smoke test.** `python scripts/smoke_test.py --endpoint <id> --api-key $RP_KEY`
   POSTs a tiny 512×512 prompt and asserts we get ≥1 image back.
5. **Benchmark.** Record cold-start + warm `/runsync` latency and tokens-per-second-ish
   throughput for the two workflow variants (20-step vs 8-step Lightning).
6. **Monitor.** RunPod dashboard for worker count, failures, GPU util; send
   handler errors (with `prompt_id`) to whatever logging stack we standardise
   on — stdout is fine for v1.

---

## 10. Explicit non-goals for v1

- No multi-tenant auth beyond the RunPod endpoint's own API key.
- No image caching / dedup across jobs.
- No fine-tuning, no training, no img2img UI — just a t2i endpoint that
  accepts a workflow JSON.
- No autoscaling tuning beyond RunPod defaults; revisit after we have real
  traffic numbers.
- No Qwen-Image-Edit (2509) workflow — that's a separate workflow file we
  can add later without changing the image or the volume layout.

---

## 11. Open questions to resolve before implementation

1. **Exact 2512 filenames + hashes.** Pull from `Comfy-Org/Qwen-Image_ComfyUI`
   (or the official Qwen release) and pin in `models.lock`.
2. **Datacenter.** Which RunPod DC has the best mix of 24 GB+ availability
   and latency for our callers?
3. **Output transport.** Inline base64 (simple, bigger payloads) vs S3/R2
   presigned URLs (needs a bucket). Default to base64 for v1 unless a caller
   needs otherwise.
4. **ComfyUI pin.** Pick a specific ComfyUI commit that has stable 2512
   support and set `COMFYUI_REF` to it — don't float on `master`.
5. **Lightning LoRA licence.** Confirm the Lightning LoRA's licence is
   compatible with our usage before shipping the 8-step workflow as default.

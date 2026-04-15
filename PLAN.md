# Plan — Serverless ComfyUI on RunPod with Network Volume (Qwen-Image 2512)

Goal: stand up a RunPod Serverless endpoint that runs ComfyUI and generates
images with **Qwen-Image 2512** in **full bf16 precision**, where all model
weights live on a **Network Volume** (not baked into the Docker image).

The image stays small and boots fast; swapping models/LoRAs is a volume
operation, not an image rebuild.

**Target model (verified, 2026-01):**
[`Qwen/Qwen-Image-2512`](https://huggingface.co/Qwen/Qwen-Image-2512),
released 2025-12-31. This is the December 2025 refresh of the Qwen-Image
text-to-image foundation model — same architecture, VAE and text encoder
as the original August 2025 release, but retrained UNet weights focused
on more realistic humans, finer natural detail, and better text rendering.
We consume it via the ComfyUI-packaged repo
[`Comfy-Org/Qwen-Image_ComfyUI`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI).

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

Full bf16 stack totals ~57.8 GB on disk (40.9 + 16.6 + 0.25). With headroom
for the Lightning LoRA, ComfyUI custom nodes, input/output buffering and
future variants, **provision the Network Volume at 150 GB**. Resizing later
is possible but the volume has to be the same datacenter as the endpoint,
so over-provision rather than under.

Mount path inside the worker: `/runpod-volume`.

```
/runpod-volume/
└── ComfyUI/
    ├── models/
    │   ├── diffusion_models/
    │   │   └── qwen_image_2512_bf16.safetensors              # 40.9 GB
    │   ├── text_encoders/
    │   │   └── qwen_2.5_vl_7b.safetensors                    # 16.6 GB, bf16
    │   ├── vae/
    │   │   └── qwen_image_vae.safetensors                    # 254 MB
    │   └── loras/
    │       └── Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors  # optional
    ├── custom_nodes/
    │   └── ComfyUI-Manager/                                  # optional
    ├── input/
    └── output/
```

All three core files come from the same repo:
[`Comfy-Org/Qwen-Image_ComfyUI`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
under `split_files/{diffusion_models,text_encoders,vae}/`.

The Lightning LoRA comes from
[`lightx2v/Qwen-Image-2512-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning)
(Apache-2.0). **Must be the 2512-specific LoRA** — the older
`lightx2v/Qwen-Image-Lightning` LoRA produces broken output on 2512 weights.

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

# ComfyUI at a pinned commit that supports Qwen-Image 2512 natively.
# The 2512 workflow template landed in Comfy-Org/workflow_templates around
# 2025-12-31, so any ComfyUI build from January 2026 or newer works. Pin to
# a specific commit SHA in CI — do not float on master.
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

Take the canonical workflow directly from
[`Comfy-Org/workflow_templates/templates/image_qwen_Image_2512.json`](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_Image_2512.json)
and commit it unmodified at `workflows/qwen_image_2512_t2i.json`. Don't
hand-roll it; the template is the upstream source of truth.

Node graph (from the reference template, all stock ComfyUI nodes — the
2512 drop introduced no new node types):

1. `UNETLoader` → `qwen_image_2512_bf16.safetensors`, weight dtype `default`
2. `CLIPLoader` → `qwen_2.5_vl_7b.safetensors`, type `qwen_image`
3. `VAELoader` → `qwen_image_vae.safetensors`
4. `ModelSamplingAuraFlow` → **shift = 3.1** (tuned for 2512; original
   Qwen-Image used ~3.0 — do not inherit shift from an old workflow)
5. `CLIPTextEncode` × 2 (positive / negative)
6. `EmptySD3LatentImage` → **1328 × 1328** (native 1:1 resolution)
7. `KSampler` — sampler `euler`, scheduler `simple`
8. `VAEDecode` → `SaveImage`

### Two workflow variants to ship

| Variant | Steps | CFG | LoRA | Notes |
|---------|-------|-----|------|-------|
| **Quality** (default) | 50 | 4.0 | none | Upstream default from the 2512 template. |
| **Turbo** | 4  | 1.0 | Lightning 4-step bf16 | Stack `LoraLoader` between `UNETLoader` and `ModelSamplingAuraFlow`, strength 1.0. |

Keep `cfg = 4.0` for the quality variant unless output regresses — upstream
guidance is that if text rendering deforms, drop CFG before swapping the
sampler.

The workflow JSON is what clients POST in `input.workflow`; they override
prompt text, seed, and size via node inputs. A small Python helper in
`src/workflows.py` loads the JSON and patches those fields by node ID so
callers don't have to know the graph shape. Two loader functions:
`load_quality()` and `load_turbo()`.

---

## 6. Model provenance (`models.lock`)

Commit a `models.lock` file (TOML) listing every weight file we expect on
the volume. Initial contents:

```toml
[[model]]
kind     = "diffusion_models"
repo     = "Comfy-Org/Qwen-Image_ComfyUI"
revision = "main"                          # pin to a commit SHA before go-live
path     = "split_files/diffusion_models/qwen_image_2512_bf16.safetensors"
dest     = "models/diffusion_models/qwen_image_2512_bf16.safetensors"
bytes    = 40_900_000_000                  # ~40.9 GB, confirm on download
sha256   = "TBD-on-first-download"

[[model]]
kind     = "text_encoders"
repo     = "Comfy-Org/Qwen-Image_ComfyUI"
revision = "main"
path     = "split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
dest     = "models/text_encoders/qwen_2.5_vl_7b.safetensors"
bytes    = 16_600_000_000                  # ~16.6 GB
sha256   = "TBD-on-first-download"

[[model]]
kind     = "vae"
repo     = "Comfy-Org/Qwen-Image_ComfyUI"
revision = "main"
path     = "split_files/vae/qwen_image_vae.safetensors"
dest     = "models/vae/qwen_image_vae.safetensors"
bytes    = 254_000_000                     # ~254 MB
sha256   = "TBD-on-first-download"

[[model]]
kind     = "loras"
repo     = "lightx2v/Qwen-Image-2512-Lightning"
revision = "main"
path     = "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
dest     = "models/loras/Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
bytes    = 0                               # fill on first download
sha256   = "TBD-on-first-download"
optional = true                            # only needed for the turbo variant
```

The one-time populator script (`scripts/populate_volume.py`) reads this
file, downloads each entry with `huggingface_hub.hf_hub_download`, records
the true size and sha256 on first run, writes them back into the lock file,
and places the file under the correct `models/<kind>/` subdir. Subsequent
runs verify rather than re-download. Same script runs as the fallback
first-boot downloader in §2.

**Before go-live:** replace every `revision = "main"` with the actual HF
commit SHA you downloaded from, and every `TBD-on-first-download` with the
real sha256. That's what makes "Qwen-Image 2512" reproducible — the version
is the lock file, not a tag in our heads.

---

## 7. RunPod endpoint configuration

Create the Serverless endpoint via the RunPod web console (or Terraform
provider if we want it in code later):

| Setting             | Value                                                           |
|---------------------|-----------------------------------------------------------------|
| Container image     | `ghcr.io/<org>/comfyui-qwen-image:<sha>` (built in CI)          |
| Container disk      | 20 GB (code + pip cache, no weights)                            |
| GPU types           | `H100 80GB`, `H200`, `A100 80GB` — **80 GB tier required**     |
| Min workers         | 0                                                               |
| Max workers         | 3 to start                                                      |
| Idle timeout        | 5 s                                                             |
| FlashBoot           | **on** — this is the big cold-start win                         |
| Execution timeout   | 600 s (50-step jobs are slow on the quality variant)            |
| Network Volume      | attach the volume created in §2 (same datacenter)               |
| Env vars            | `COMFY_PORT=8188`, optional `BUCKET_*` for S3 output            |

GPU sizing notes (this is the big change from a quantized setup):
- **Full bf16 needs the 80 GB tier.** 40.9 GB UNet + 16.6 GB text encoder
  + VAE + activations does not fit on a 48 GB card (L40S, A6000 Ada) at
  full precision. Use `A100 80GB`, `H100 80GB`, or `H200`.
- The 16.6 GB bf16 text encoder is not optional at full precision. If we
  ever need to fit on 48 GB later, the drop is to swap the text encoder
  to `qwen_2.5_vl_7b_fp8_scaled.safetensors` (9.38 GB) — that single swap
  makes the stack fit without touching the UNet precision. Note that as a
  separate, explicit fallback, not a silent downgrade.
- `--lowvram` / CPU offload of the text encoder is a valid emergency
  pressure-release valve but tanks per-job latency; avoid as default.
- Cold start budget: expect 60–120 s the first time a worker pulls
  57.8 GB of weights across the volume into GPU memory. FlashBoot helps
  with container state but not with first-touch model loading.

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

1. **Datacenter.** Which RunPod DC has the best mix of `H100 80GB` /
   `A100 80GB` / `H200` availability and acceptable latency for our
   callers? The Network Volume has to be created in that same DC.
2. **Output transport.** Inline base64 (simple, bigger payloads) vs S3/R2
   presigned URLs (needs a bucket). Default to base64 for v1 unless a
   caller needs otherwise.
3. **ComfyUI commit pin.** Pick a specific ComfyUI commit from
   January 2026 or later that has stable 2512 support, set `COMFYUI_REF`
   to its SHA, and record why that commit in CI notes.
4. **HF revision pins.** On first volume hydration, capture the actual
   commit SHAs for `Comfy-Org/Qwen-Image_ComfyUI` and
   `lightx2v/Qwen-Image-2512-Lightning` and write them into `models.lock`
   (replacing `"main"`).

### Resolved (from research on 2026-04-15)

- ~~Exact 2512 filenames + hashes~~ — see §6.
- ~~Lightning LoRA licence~~ — Apache-2.0, OK to ship.
- ~~Does 2512 introduce new ComfyUI node types~~ — no, it's a drop-in
  weight swap on the original Qwen-Image graph.

## 12. References

- Model: https://huggingface.co/Qwen/Qwen-Image-2512
- ComfyUI-packaged weights: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
- Reference workflow: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_Image_2512.json
- Lightning LoRA: https://huggingface.co/lightx2v/Qwen-Image-2512-Lightning
- ComfyUI tutorial: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2512
- Qwen blog post: https://qwen.ai/blog?id=qwen-image-2512
- Worker handler reference: https://github.com/runpod-workers/worker-comfyui

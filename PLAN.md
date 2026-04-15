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
for ComfyUI custom nodes, input/output buffering and future variants,
**provision the Network Volume at 100 GB**. Resizing later is possible but
the volume has to be the same datacenter as the endpoint, so over-provision
rather than under.

Mount path inside the worker: `/runpod-volume`.

```
/runpod-volume/
└── ComfyUI/
    ├── models/
    │   ├── diffusion_models/
    │   │   └── qwen_image_2512_bf16.safetensors    # 40.9 GB
    │   ├── text_encoders/
    │   │   └── qwen_2.5_vl_7b.safetensors          # 16.6 GB, bf16
    │   └── vae/
    │       └── qwen_image_vae.safetensors          # 254 MB
    ├── custom_nodes/
    │   └── ComfyUI-Manager/                        # optional
    ├── input/
    └── output/
```

All three weight files come from the same repo:
[`Comfy-Org/Qwen-Image_ComfyUI`](https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI)
under `split_files/{diffusion_models,text_encoders,vae}/`.

**No LoRAs.** This endpoint is quality-first: no Lightning / turbo / step-
distilled LoRAs. Jobs run the full 50-step schedule on the untouched bf16
weights. If a "fast mode" is ever wanted, it's a separate endpoint with a
separate workflow — not a flag on this one.

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
6. `EmptySD3LatentImage` → **1104 × 1472** (portrait 3:4; see table below)
7. `KSampler` — sampler `euler`, scheduler `simple`
8. `VAEDecode` → `SaveImage`

### Sampling parameters (quality-first, no turbo path)

| Param     | Value   |
|-----------|---------|
| Steps     | **50**  |
| CFG       | **4.0** |
| Sampler   | `euler` |
| Scheduler | `simple` |
| Shift     | **3.1** (`ModelSamplingAuraFlow`) |
| Resolution | **1104 × 1472** (portrait 3:4) |

### Supported resolutions (Qwen's official `aspect_ratios` dict)

All seven entries sit at ~1.76 MP, the model's native training resolution.
Callers can override `width` / `height` via the `EmptySD3LatentImage` node,
but should only ever use values from this list:

| Aspect | Width × Height |
|--------|----------------|
| 1:1    | 1328 × 1328 |
| 16:9   | 1664 × 928  |
| 9:16   | 928 × 1664  |
| 4:3    | 1472 × 1104 |
| **3:4**  | **1104 × 1472** *(default)* |
| 3:2    | 1584 × 1056 |
| 2:3    | 1056 × 1584 |

Source: [`QwenLM/Qwen-Image` README](https://github.com/QwenLM/Qwen-Image/blob/main/README.md),
same list applies to Qwen-Image 2512.

These are the upstream defaults from the 2512 template. Don't touch them
without a concrete quality complaint to fix. Upstream guidance: if text
rendering deforms, drop CFG before changing sampler.

The workflow JSON is what clients POST in `input.workflow`; they override
prompt text, seed and size via node inputs. A small helper in
`src/workflows.py` loads the JSON and patches those fields by node ID so
callers don't have to know the graph shape.

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

### 7a. Datacenter

**Chosen DC: `US-GA-2`** (Georgia, US). A 100 GB Network Volume has been
provisioned here. The Serverless endpoint will auto-pin to `US-GA-2` when
it attaches the volume.

Pre-flight check before building the endpoint: open **Serverless → GPU
types** in the RunPod web console and confirm at least one of `H100 80GB`,
`A100 80GB` or `H200` is listed as available in `US-GA-2`. If none of the
80 GB tier SKUs is available there, the volume has to be recreated in a
different DC — there is no cross-DC mounting.

> RunPod ("DC" = datacenter) runs GPU capacity in several physical
> regions. Two constraints drive the choice: Network Volumes are
> region-locked (endpoint must be in the same DC as the volume), and
> 80 GB GPU availability varies per DC. The choice was made by looking at
> **Storage → Network Volumes → New Network Volume** and picking a DC
> that had both volume capacity and the 80 GB GPU tier.

### 7b. Endpoint settings

Create the Serverless endpoint via the RunPod web console. **No external
container registry.** RunPod's Serverless GitHub integration builds the
Dockerfile directly from this repo on every push to the target branch —
zero CI on our side, no GHCR/DockerHub, no image tag management.

| Setting             | Value                                                                         |
|---------------------|-------------------------------------------------------------------------------|
| Build source        | **GitHub: `FunkyWhiteCat/serverless`, branch `<release>`, Dockerfile `docker/Dockerfile`** |
| Container disk      | 20 GB (code + pip cache, no weights)                                          |
| GPU types (priority)| `H200` → `H100 80GB SXM` → `H100 80GB PCIe` → `A100 80GB SXM` → `A100 80GB PCIe` |
| Min workers         | 0                                                                             |
| Max workers         | 3 to start                                                                    |
| Idle timeout        | 5 s                                                                           |
| FlashBoot           | **on** — cold-start win                                                       |
| Execution timeout   | 600 s (50-step jobs run long)                                                 |
| Network Volume      | attach the `US-GA-2` volume from §7a                                          |
| Env vars            | `COMFY_PORT=8188`                                                             |

**GPU selection policy.** RunPod Serverless lets you give a multi-select
GPU list; the platform picks an available worker from that list per job.
Listing them in descending-speed order (H200 first, A100 PCIe last) means
the fastest available card is always tried first. If H200s are saturated
in US-GA-2, jobs fall through to H100s, then A100s — no caller-facing
config needed.

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
├── models.lock                     (HF pins for Qwen-Image 2512)
├── docker/
│   ├── Dockerfile
│   └── extra_model_paths.yaml
├── src/
│   ├── handler.py
│   ├── start.sh
│   └── workflows.py
├── workflows/
│   └── qwen_image_2512_t2i.json
└── scripts/
    ├── populate_volume.py          # one-shot volume hydrator
    └── smoke_test.py               # hits /runsync with a tiny prompt
```

No `.github/workflows/` — RunPod's Serverless GitHub integration builds
the image on their side, so we don't need a CI workflow or an external
registry.

---

## 9. Build → deploy → verify loop

1. **Hydrate the volume** (one-time, or when `models.lock` changes):
   rent a cheap pod in `US-GA-2` with the volume attached and run
   `python scripts/populate_volume.py`. Step-by-step in §9a. Terminate
   the pod when the script finishes.
2. **Push code.** Commit handler + Dockerfile + workflow JSON to the
   release branch. RunPod's GitHub integration picks it up and rebuilds
   the worker image automatically.
3. **Wait for build.** Watch the build log in the RunPod Serverless UI.
   First build is ~5 min; incremental builds after that are faster.
4. **Smoke test.** `python scripts/smoke_test.py --endpoint <id> --api-key $RP_KEY`
   POSTs a short prompt at 1104×1472 and asserts we get ≥1 image back.
5. **Benchmark.** Record cold-start + warm `/runsync` wall time for the
   50-step quality workflow at 1104×1472.
6. **Monitor.** RunPod dashboard for worker count, failures, GPU util;
   handler errors (with `prompt_id`) go to stdout for v1.

### 9a. One-time volume hydration (step by step)

This runs the `scripts/populate_volume.py` script once, from inside a
throwaway RunPod pod that has the `US-GA-2` Network Volume attached. The
volume keeps the downloaded weights after the pod is terminated.

1. **Rent a throwaway pod in US-GA-2.** RunPod web console →
   **Pods → Deploy**. Filter by datacenter `US-GA-2`. Pick **the cheapest
   GPU available** (e.g. `RTX 2000 Ada` or `RTX A4000` — we only need
   network I/O, not compute). In the deploy dialog:
   - **Network Volume:** attach the US-GA-2 volume.
   - **Container image:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
     (or any recent PyTorch image; we just need Python 3.11 + pip).
   - **Container disk:** 20 GB is fine.
   - **Expose HTTP ports:** none needed.
2. **Open a web terminal** on the pod from the RunPod UI (or SSH in).
3. Run the hydration script:
   ```bash
   cd /workspace
   git clone https://github.com/FunkyWhiteCat/serverless.git
   cd serverless
   pip install "huggingface_hub>=0.26" hf_transfer
   HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/populate_volume.py
   ```
   Expect ~58 GB of downloads. At typical RunPod egress (500 MB/s–1 GB/s
   from HF with `hf_transfer` on) this takes roughly 1–3 minutes.
4. **Copy the sha256/bytes values** the script prints at the end, paste
   them into `models.lock` to replace the `TBD-on-first-download` lines,
   and commit + push.
5. **Verify** the files are in place:
   ```bash
   ls -lh /runpod-volume/ComfyUI/models/diffusion_models/
   ls -lh /runpod-volume/ComfyUI/models/text_encoders/
   ls -lh /runpod-volume/ComfyUI/models/vae/
   ```
6. **Terminate the pod** from the RunPod UI. The volume (and its files)
   persist; only the compute goes away.

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

1. **ComfyUI commit pin.** Pick a specific ComfyUI commit from
   January 2026 or later that has stable 2512 support, set `COMFYUI_REF`
   to its SHA, and record why that commit in CI notes.
2. **HF revision pins.** On first volume hydration, capture the actual
   commit SHA for `Comfy-Org/Qwen-Image_ComfyUI` and write it into
   `models.lock` (replacing `"main"`).

### Resolved

- ~~Exact 2512 filenames + hashes~~ — see §6.
- ~~Does 2512 introduce new ComfyUI node types~~ — no, drop-in weight swap.
- ~~Should we ship a Lightning turbo variant~~ — no. Quality-first.
- ~~Datacenter~~ — `US-GA-2` (§7a). Volume provisioned.
- ~~Container registry~~ — none. RunPod's GitHub integration builds
  directly from this repo (§7b / §9).
- ~~Output transport~~ — inline base64 in the JSON response.
- ~~Default resolution~~ — 1104×1472 portrait (Qwen official 3:4).
- ~~GPU selection~~ — fastest-first list, H200 → H100 SXM → H100 PCIe →
  A100 SXM → A100 PCIe. RunPod picks first available.

## 12. References

- Model: https://huggingface.co/Qwen/Qwen-Image-2512
- ComfyUI-packaged weights: https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI
- Reference workflow: https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_Image_2512.json
- ComfyUI tutorial: https://docs.comfy.org/tutorials/image/qwen/qwen-image-2512
- Qwen blog post: https://qwen.ai/blog?id=qwen-image-2512
- Worker handler reference: https://github.com/runpod-workers/worker-comfyui

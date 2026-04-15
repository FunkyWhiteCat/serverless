"""RunPod Serverless handler for ComfyUI + Qwen-Image 2512.

Responsibilities:
  1. Wait for ComfyUI's local HTTP server to be ready.
  2. Accept a job payload (full workflow, or convenience shape).
  3. Stage any input images onto the Network Volume.
  4. POST the workflow to ComfyUI /prompt and wait for completion
     over the ComfyUI /ws websocket.
  5. Collect output images from disk and return them as base64.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import runpod  # type: ignore[import-not-found]
import websocket  # websocket-client

import workflows as wf_mod

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
COMFY_HTTP = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_WS = f"ws://{COMFY_HOST}:{COMFY_PORT}/ws"

VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume/ComfyUI"))
INPUT_DIR = VOLUME_ROOT / "input"
OUTPUT_DIR = VOLUME_ROOT / "output"
TEMP_DIR = VOLUME_ROOT / "temp"

STARTUP_TIMEOUT_S = 240  # generous — ComfyUI on a cold worker is slow
POLL_INTERVAL_S = 0.5
WS_RECV_TIMEOUT_S = 600  # per-recv ceiling; matches endpoint execution_timeout
WARMUP_TIMEOUT_S = 900   # upper bound on first-time model load via /history
WARMUP_POLL_INTERVAL_S = 2.0


# ---------------------------------------------------------------------------
# ComfyUI plumbing
# ---------------------------------------------------------------------------

def wait_for_comfy(timeout_s: int = STARTUP_TIMEOUT_S) -> None:
    """Block until ComfyUI's /system_stats endpoint responds."""
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                f"{COMFY_HTTP}/system_stats", timeout=2
            ) as r:
                if r.status == 200:
                    return
        except Exception as e:  # noqa: BLE001 — we retry any failure
            last_err = e
        time.sleep(POLL_INTERVAL_S)
    raise RuntimeError(
        f"ComfyUI did not become ready within {timeout_s}s "
        f"(last error: {last_err!r})"
    )


def queue_prompt(workflow: dict, client_id: str) -> str:
    body = json.dumps({"prompt": workflow, "client_id": client_id}).encode()
    req = urllib.request.Request(
        f"{COMFY_HTTP}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            data = json.loads(r.read())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"/prompt rejected workflow: {err_body}") from None

    node_errors = data.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(
            f"/prompt returned node_errors: {json.dumps(node_errors)}"
        )
    return data["prompt_id"]


def wait_for_completion(ws: "websocket.WebSocket", prompt_id: str) -> None:
    """Block on ComfyUI's websocket until the given prompt finishes."""
    while True:
        msg = ws.recv()
        if isinstance(msg, bytes):
            # Binary preview frames — skip.
            continue
        if not msg:
            continue
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        msg_type = data.get("type")
        payload = data.get("data") or {}

        if msg_type == "executing":
            if (
                payload.get("prompt_id") == prompt_id
                and payload.get("node") is None
            ):
                return  # done
        elif msg_type == "execution_error":
            if payload.get("prompt_id") == prompt_id:
                raise RuntimeError(
                    f"ComfyUI execution_error: {json.dumps(payload)}"
                )
        elif msg_type == "execution_interrupted":
            if payload.get("prompt_id") == prompt_id:
                raise RuntimeError(
                    f"ComfyUI execution_interrupted: {json.dumps(payload)}"
                )


def fetch_history(prompt_id: str) -> dict:
    with urllib.request.urlopen(f"{COMFY_HTTP}/history/{prompt_id}") as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------
# Worker warmup (runs once at startup, before the handler is registered)
# ---------------------------------------------------------------------------

def _fetch_history_entry_or_none(prompt_id: str) -> dict | None:
    """Non-raising variant of fetch_history for use inside warmup polling."""
    try:
        with urllib.request.urlopen(
            f"{COMFY_HTTP}/history/{prompt_id}", timeout=10
        ) as r:
            history = json.loads(r.read())
    except Exception:  # noqa: BLE001 — transient HTTP failures retry
        return None
    return history.get(prompt_id)


def warmup() -> None:
    """Force ComfyUI to fault in all ~58 GB of Qwen-Image 2512 weights
    before we register the real handler with RunPod.

    Submits a throwaway 1-step version of the default workflow and polls
    ``/history`` to completion via plain HTTP (no websocket — we don't
    want the handler's per-recv timeout to apply during the multi-minute
    cold-load window). The work happens during worker initialization, so
    it counts against RunPod's (generous) worker-init budget rather than
    the per-job ``execution_timeout``.

    After warmup, real jobs see a warm worker with models resident in
    GPU memory, so the longest gap between websocket frames drops from
    "whole UNet load time" to "sub-second inter-node transition" — well
    within WS_RECV_TIMEOUT_S.
    """
    print(
        "[handler] warming up: submitting a 1-step workflow to pre-load "
        "Qwen-Image 2512 weights (may take several minutes on a cold "
        "worker while ~58 GB is read off the network volume)...",
        flush=True,
    )
    t0 = time.monotonic()

    wf = wf_mod.load_default()
    wf_mod.patch(
        wf,
        prompt="warmup",
        negative_prompt="",
        seed=0,
        steps=1,
        cfg=1.0,
    )

    client_id = "warmup-" + uuid.uuid4().hex
    prompt_id = queue_prompt(wf, client_id)
    print(f"[handler] warmup queued, prompt_id={prompt_id}", flush=True)

    deadline = time.monotonic() + WARMUP_TIMEOUT_S
    while time.monotonic() < deadline:
        entry = _fetch_history_entry_or_none(prompt_id)
        if entry is not None:
            status_str = (entry.get("status") or {}).get("status_str")
            if status_str == "success":
                elapsed = time.monotonic() - t0
                print(
                    f"[handler] warmup completed in {elapsed:.1f}s; "
                    f"worker is warm and ready",
                    flush=True,
                )
                return
            if status_str == "error":
                raise RuntimeError(
                    f"warmup errored: {json.dumps(entry.get('status'))}"
                )
        time.sleep(WARMUP_POLL_INTERVAL_S)

    raise RuntimeError(
        f"warmup did not finish within {WARMUP_TIMEOUT_S}s; "
        f"worker is giving up"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def stage_input_images(items: list[dict]) -> None:
    """Write ``{name, image}`` base64 blobs into ComfyUI's input dir."""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = item.get("name")
        b64 = item.get("image")
        if not name or not b64:
            raise ValueError(
                "each input image must have both 'name' and 'image' fields"
            )
        if "," in b64 and b64.lstrip().startswith("data:"):
            b64 = b64.split(",", 1)[1]  # strip any data: URL prefix
        (INPUT_DIR / name).write_bytes(base64.b64decode(b64))


def collect_output_images(history_entry: dict) -> list[dict]:
    """Read every output image referenced in the history and base64-encode it."""
    images: list[dict] = []
    outputs = history_entry.get("outputs", {}) or {}
    for node_id, node_out in outputs.items():
        for img in node_out.get("images", []) or []:
            folder_type = img.get("type", "output")
            subfolder = img.get("subfolder", "") or ""
            filename = img["filename"]

            if folder_type == "output":
                base = OUTPUT_DIR
            elif folder_type == "temp":
                base = TEMP_DIR
            else:
                # ComfyUI also emits "input" type for image-reference nodes;
                # we don't want to return those.
                continue

            path = base / subfolder / filename if subfolder else base / filename
            if not path.exists():
                raise RuntimeError(f"output image missing on volume: {path}")

            with path.open("rb") as f:
                data_b64 = base64.b64encode(f.read()).decode("ascii")

            images.append(
                {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": folder_type,
                    "node_id": node_id,
                    "data": data_b64,
                }
            )
    return images


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _build_workflow(inp: dict[str, Any]) -> dict[str, Any]:
    """Either use the caller's full workflow or patch the default."""
    if "workflow" in inp and inp["workflow"]:
        return inp["workflow"]
    return wf_mod.build_from_convenience(inp)


def handler(event: dict) -> dict:
    inp = event.get("input") or {}

    workflow = _build_workflow(inp)

    if inp.get("images"):
        stage_input_images(inp["images"])

    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect(f"{COMFY_WS}?clientId={client_id}", timeout=10)
    # websocket-client applies `timeout` to every subsequent recv(). On a
    # cold worker ComfyUI goes quiet on the ws while it mmaps ~57 GB of
    # bf16 weights off the Network Volume — that gap can be 30–120 s
    # before the first progress frame arrives. Bump the per-recv timeout
    # well above that; RunPod's endpoint-level execution_timeout is the
    # real ceiling and will kill a genuinely stuck job.
    ws.settimeout(WS_RECV_TIMEOUT_S)

    t0 = time.monotonic()
    try:
        prompt_id = queue_prompt(workflow, client_id)
        wait_for_completion(ws, prompt_id)
    finally:
        try:
            ws.close()
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass

    elapsed = time.monotonic() - t0
    history = fetch_history(prompt_id).get(prompt_id, {})
    images = collect_output_images(history)

    if not images:
        raise RuntimeError(
            f"workflow {prompt_id} produced no images; "
            f"history outputs: {json.dumps(history.get('outputs', {}))[:500]}"
        )

    return {
        "images": images,
        "prompt_id": prompt_id,
        "timings": {"exec_s": round(elapsed, 3)},
    }


def safe_handler(event: dict) -> dict:
    try:
        return handler(event)
    except Exception as exc:  # noqa: BLE001 — map every failure to JSON
        return {
            "error": str(exc),
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    print("[handler] waiting for ComfyUI to be ready...", flush=True)
    wait_for_comfy()
    print(f"[handler] ComfyUI ready at {COMFY_HTTP}", flush=True)
    warmup()
    runpod.serverless.start({"handler": safe_handler})

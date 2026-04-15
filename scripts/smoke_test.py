#!/usr/bin/env python3
"""Smoke test for the Serverless ComfyUI + Qwen-Image 2512 endpoint.

Hits the RunPod Serverless endpoint via /runsync with the convenience
payload shape, waits for a response, decodes the first returned image
and writes it to disk.

Usage:
    RUNPOD_API_KEY=<key> python scripts/smoke_test.py --endpoint-id <id>

    # Optional overrides:
    --prompt "..."          # positive prompt
    --negative "..."        # negative prompt
    --seed 42               # int
    --width 1104 --height 1472
    --timeout 600           # seconds
    --out smoke.png         # output file
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "A portrait photo of a red panda wearing oversized headphones, "
    "studio lighting, 85mm lens, shallow depth of field, photorealistic"
)


def post_runsync(endpoint_id: str, api_key: str, payload: dict, timeout_s: int) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
    req = urllib.request.Request(
        url,
        data=json.dumps({"input": payload}).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise SystemExit(f"runsync HTTP {e.code}: {err_body}")
    except urllib.error.URLError as e:
        raise SystemExit(f"runsync network error: {e}")
    elapsed = time.monotonic() - t0
    result = json.loads(body)
    print(
        f"[smoke] runsync returned in {elapsed:.1f}s, "
        f"status={result.get('status')}"
    )
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-id", required=True, help="RunPod Serverless endpoint ID")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1104)
    p.add_argument("--height", type=int, default=1472)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default="smoke_test_output.png")
    args = p.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: set RUNPOD_API_KEY in your environment", file=sys.stderr)
        return 1

    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
    }

    result = post_runsync(args.endpoint_id, api_key, payload, args.timeout)

    # RunPod wraps the handler response under result["output"] on /runsync.
    output = result.get("output") or {}
    err = output.get("error") or result.get("error")
    if err:
        print(f"[smoke] handler error: {err}")
        if isinstance(output, dict) and "traceback" in output:
            print(output["traceback"])
        return 2

    images = output.get("images") or []
    if not images:
        print(f"[smoke] no images in response. Raw: {json.dumps(result)[:2000]}")
        return 3

    first = images[0]
    data = base64.b64decode(first["data"])
    with open(args.out, "wb") as f:
        f.write(data)

    timings = output.get("timings") or {}
    print(f"[smoke] wrote {args.out} ({len(data)} bytes)")
    print(f"[smoke] prompt_id={output.get('prompt_id')}")
    print(f"[smoke] exec_s={timings.get('exec_s')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

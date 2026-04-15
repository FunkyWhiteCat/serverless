#!/usr/bin/env python3
"""MCP server exposing the Serverless Qwen-Image 2512 endpoint as a tool.

Usable from any MCP-aware client — Claude Code, Claude Desktop,
LM Studio, Cursor, Continue, etc. Ships one tool, ``generate_image``,
which forwards its arguments to the RunPod ``/runsync`` endpoint and
returns the generated PNG both inline (as an MCP ImageContent block,
so vision-capable models can see it in-context) and as a file on disk
(so the artifact persists after the chat turn).

Dependencies:
    pip install "mcp>=1.0"

Required environment variables:
    RUNPOD_API_KEY         your RunPod API key (starts with ``rpa_``)
    RUNPOD_ENDPOINT_ID     the serverless endpoint id (the opaque
                           string in the endpoint URL, NOT the name)

Optional environment variables:
    QWEN_IMAGE_DEFAULT_OUT_DIR
        directory to save PNGs when the caller doesn't supply
        save_path. Defaults to ~/qwen_image_output.
    RUNPOD_RUNSYNC_TIMEOUT
        HTTP client timeout in seconds for the runsync POST.
        Defaults to 900 (15 min) to cover cold worker + warmup +
        50-step generation comfortably.

--------------------------------------------------------------------
Registering with Claude Code
--------------------------------------------------------------------

    claude mcp add qwen-image-2512 \\
        -e RUNPOD_API_KEY=rpa_xxx \\
        -e RUNPOD_ENDPOINT_ID=xxxxxxxxxxxx \\
        -- python /absolute/path/to/scripts/qwen_image_mcp.py

Or, for a project-local registration, add this to ``.mcp.json`` at
the repo root:

    {
      "mcpServers": {
        "qwen-image-2512": {
          "command": "python",
          "args": ["/absolute/path/to/scripts/qwen_image_mcp.py"],
          "env": {
            "RUNPOD_API_KEY": "rpa_xxx",
            "RUNPOD_ENDPOINT_ID": "xxxxxxxxxxxx"
          }
        }
      }
    }

--------------------------------------------------------------------
Registering with LM Studio
--------------------------------------------------------------------

LM Studio 0.3.17+ has a built-in MCP client. Open:

    Settings  ->  Program  ->  Integrations  ->  "Edit mcp.json"

and paste the same ``mcpServers`` block as above. After saving,
LM Studio reloads the server and the ``generate_image`` tool becomes
available to any tool-calling model you run locally (Qwen 2.5, Llama
3.3, GPT-OSS, etc.). LM Studio renders ImageContent blocks inline
in the chat — the on-disk copy at save_path is your persistent
artifact.

--------------------------------------------------------------------
Registering with Claude Desktop
--------------------------------------------------------------------

Add the same block to ``~/Library/Application Support/Claude/
claude_desktop_config.json`` (macOS) or
``%APPDATA%\\Claude\\claude_desktop_config.json`` (Windows) and
restart the app.
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

# Mirrors src/workflows.py SUPPORTED_RESOLUTIONS. Keep in sync manually
# if the worker's supported set ever changes — the worker is the source
# of truth and will reject anything not in its own list.
_SUPPORTED_RESOLUTIONS: list[tuple[int, int]] = [
    (1328, 1328),  # 1:1
    (1664, 928),   # 16:9
    (928, 1664),   # 9:16
    (1472, 1104),  # 4:3
    (1104, 1472),  # 3:4  (default)
    (1584, 1056),  # 3:2
    (1056, 1584),  # 2:3
]

_DEFAULT_OUT_DIR = Path(
    os.environ.get(
        "QWEN_IMAGE_DEFAULT_OUT_DIR",
        str(Path.home() / "qwen_image_output"),
    )
)
_RUNSYNC_TIMEOUT = int(os.environ.get("RUNPOD_RUNSYNC_TIMEOUT", "900"))

mcp = FastMCP("qwen-image-2512")


@mcp.tool()
def generate_image(
    prompt: str,
    negative_prompt: str = "",
    seed: int | None = None,
    width: int = 1104,
    height: int = 1472,
    steps: int = 50,
    cfg: float = 4.0,
    save_path: str | None = None,
) -> list[ImageContent | TextContent]:
    """Generate a single image with Qwen-Image 2512 via RunPod Serverless.

    Sends the request to the endpoint configured by RUNPOD_API_KEY and
    RUNPOD_ENDPOINT_ID and blocks until the PNG comes back. First call
    on a cold worker can take several minutes (network-volume read of
    the 58 GB model is the bottleneck); subsequent calls against a
    warm worker finish in ~25 s.

    Qwen-Image 2512 was trained at ~1.76 MP and only supports these
    seven resolutions: 1328x1328 (1:1), 1664x928 (16:9), 928x1664
    (9:16), 1472x1104 (4:3), 1104x1472 (3:4, default), 1584x1056
    (3:2), 1056x1584 (2:3). Anything else is rejected by the worker.

    Args:
        prompt: Positive prompt. Required.
        negative_prompt: Negative prompt. Optional; default empty.
        seed: Integer seed. Optional; the worker will randomize if
            omitted so repeated calls don't collide.
        width: One of the supported widths above. Default 1104.
        height: One of the supported heights above. Default 1472.
        steps: KSampler steps. Default 50 (native quality).
        cfg: Classifier-free-guidance scale. Default 4.0.
        save_path: Where to save the PNG on disk. Optional; defaults
            to ~/qwen_image_output/YYYYmmdd_HHMMSS_<seed>.png. Parent
            directories are created on demand.

    Returns:
        A list with one ImageContent (the PNG inline, base64-encoded)
        and one TextContent (metadata: on-disk path, prompt_id,
        exec_s, dimensions, steps, cfg). Errors from the worker are
        returned as a single TextContent describing what went wrong.
    """
    api_key = os.environ.get("RUNPOD_API_KEY")
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not api_key or not endpoint_id:
        return [
            TextContent(
                type="text",
                text=(
                    "RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID must be set in "
                    "the environment where this MCP server was launched. "
                    "Check the MCP client's server config."
                ),
            )
        ]

    if (width, height) not in _SUPPORTED_RESOLUTIONS:
        allowed = ", ".join(f"{w}x{h}" for w, h in _SUPPORTED_RESOLUTIONS)
        return [
            TextContent(
                type="text",
                text=(
                    f"Resolution {width}x{height} is not an official "
                    f"Qwen-Image aspect ratio. Use one of: {allowed}."
                ),
            )
        ]

    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "cfg": cfg,
    }
    if seed is not None:
        payload["seed"] = int(seed)

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

    try:
        with urllib.request.urlopen(req, timeout=_RUNSYNC_TIMEOUT) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return [
            TextContent(
                type="text",
                text=f"runsync HTTP {e.code}: {err_body}",
            )
        ]
    except urllib.error.URLError as e:
        return [
            TextContent(
                type="text",
                text=f"runsync network error: {e}",
            )
        ]

    result = json.loads(body)
    output = result.get("output") or {}
    err = output.get("error") or result.get("error")
    if err:
        tb = output.get("traceback", "") if isinstance(output, dict) else ""
        return [
            TextContent(
                type="text",
                text=f"handler error: {err}\n{tb}".rstrip(),
            )
        ]

    images = output.get("images") or []
    if not images:
        return [
            TextContent(
                type="text",
                text=(
                    f"no images returned. status={result.get('status')}, "
                    f"raw={json.dumps(result)[:800]}"
                ),
            )
        ]

    first = images[0]
    img_b64 = first["data"]
    png_bytes = base64.b64decode(img_b64)

    # Pick a disk path: honor caller's save_path, else derive from timestamp.
    if save_path:
        out_path = Path(save_path).expanduser()
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        used_seed = payload.get("seed", "rand")
        out_path = _DEFAULT_OUT_DIR / f"{ts}_{used_seed}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(png_bytes)

    timings = output.get("timings") or {}
    summary = (
        f"Saved {out_path} ({len(png_bytes)} bytes)\n"
        f"prompt_id={output.get('prompt_id')} "
        f"exec_s={timings.get('exec_s')} "
        f"dimensions={width}x{height} steps={steps} cfg={cfg}"
    )

    return [
        ImageContent(type="image", data=img_b64, mimeType="image/png"),
        TextContent(type="text", text=summary),
    ]


if __name__ == "__main__":
    mcp.run()

"""Workflow loader + patcher for Qwen-Image 2512.

Callers can either POST a full ComfyUI workflow in ``event.input.workflow``,
or use the convenience shape
    { "prompt": str, "negative_prompt": str?, "seed": int?,
      "width": int?, "height": int?, "steps": int?, "cfg": float? }
and let this module patch the default template.
"""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_APP_ROOT = Path(os.environ.get("APP_ROOT", str(_HERE.parent)))
DEFAULT_WORKFLOW_PATH = Path(
    os.environ.get(
        "DEFAULT_WORKFLOW_PATH",
        str(_APP_ROOT / "workflows" / "qwen_image_2512_t2i.json"),
    )
)

# Qwen-Image's official aspect_ratios dict (from QwenLM/Qwen-Image README).
# All seven entries sit at ~1.76 MP, the model's native training resolution.
SUPPORTED_RESOLUTIONS: set[tuple[int, int]] = {
    (1328, 1328),  # 1:1
    (1664, 928),   # 16:9
    (928, 1664),   # 9:16
    (1472, 1104),  # 4:3
    (1104, 1472),  # 3:4  (default)
    (1584, 1056),  # 3:2
    (1056, 1584),  # 2:3
}

MAX_SEED = 2**31 - 1


def load_default() -> dict[str, Any]:
    """Return a fresh deep copy of the default Qwen-Image 2512 workflow."""
    with DEFAULT_WORKFLOW_PATH.open() as f:
        return json.load(f)


def _find_node(workflow: dict[str, Any], class_type: str) -> tuple[str, dict]:
    for node_id, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == class_type:
            return node_id, node
    raise KeyError(f"no {class_type} node in workflow")


def patch(
    workflow: dict[str, Any],
    *,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg: float | None = None,
) -> dict[str, Any]:
    """Mutate ``workflow`` in place with the given overrides and return it."""
    _, ksampler = _find_node(workflow, "KSampler")
    pos_ref = ksampler["inputs"]["positive"][0]
    neg_ref = ksampler["inputs"]["negative"][0]

    if prompt is not None:
        workflow[pos_ref]["inputs"]["text"] = str(prompt)
    if negative_prompt is not None:
        workflow[neg_ref]["inputs"]["text"] = str(negative_prompt)

    # Always write a seed. Randomize when the caller didn't supply one so
    # repeated default-prompt calls don't return the same image.
    ksampler["inputs"]["seed"] = (
        int(seed) if seed is not None else random.randint(0, MAX_SEED)
    )

    if steps is not None:
        ksampler["inputs"]["steps"] = int(steps)
    if cfg is not None:
        ksampler["inputs"]["cfg"] = float(cfg)

    if width is not None or height is not None:
        _, latent = _find_node(workflow, "EmptySD3LatentImage")
        new_w = int(width) if width is not None else int(latent["inputs"]["width"])
        new_h = int(height) if height is not None else int(latent["inputs"]["height"])
        if (new_w, new_h) not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"resolution {new_w}x{new_h} is not an official Qwen-Image "
                f"aspect_ratio; use one of {sorted(SUPPORTED_RESOLUTIONS)}"
            )
        latent["inputs"]["width"] = new_w
        latent["inputs"]["height"] = new_h

    return workflow


def build_from_convenience(inp: dict[str, Any]) -> dict[str, Any]:
    """Load the default workflow and apply caller overrides in one shot."""
    wf = load_default()
    patch(
        wf,
        prompt=inp.get("prompt"),
        negative_prompt=inp.get("negative_prompt"),
        seed=inp.get("seed"),
        width=inp.get("width"),
        height=inp.get("height"),
        steps=inp.get("steps"),
        cfg=inp.get("cfg"),
    )
    return wf

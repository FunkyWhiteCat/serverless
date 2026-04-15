#!/usr/bin/env python3
"""Hydrate a RunPod Network Volume with Qwen-Image 2512 weights.

Run this once, from inside a RunPod pod that has the target Network Volume
attached at /runpod-volume. Reads models.lock from the repo root and
downloads every listed file into /runpod-volume/ComfyUI/models/<kind>/.

Usage:
    pip install "huggingface_hub>=0.26" hf_transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 python scripts/populate_volume.py

Re-running is safe: files that already exist on the volume and match the
expected sha256 (when pinned) are skipped.

On first run, the `sha256` and `bytes` values in models.lock will still be
placeholders. The script prints the real values at the end; paste them
back into models.lock and commit.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_PATH = REPO_ROOT / "models.lock"
VOLUME_ROOT = Path(os.environ.get("VOLUME_ROOT", "/runpod-volume/ComfyUI"))
STAGING_DIR = VOLUME_ROOT / ".staging"
TBD = "TBD-on-first-download"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def human_size(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PiB"


def fetch_one(entry: dict) -> tuple[int, str]:
    repo = entry["repo"]
    revision = entry.get("revision", "main")
    path_in_repo = entry["path"]
    dest_rel = entry["dest"]
    expected_sha = entry.get("sha256", TBD)

    dest_abs = VOLUME_ROOT / dest_rel
    dest_abs.parent.mkdir(parents=True, exist_ok=True)

    if dest_abs.exists():
        size = dest_abs.stat().st_size
        actual = sha256_file(dest_abs)
        if expected_sha == TBD:
            print(f"  OK   {dest_rel}")
            print(f"       already present, {human_size(size)}, sha256={actual}")
            return size, actual
        if actual == expected_sha:
            print(f"  OK   {dest_rel}  (already present, sha matches)")
            return size, actual
        print(f"  WARN {dest_rel}  sha mismatch on disk, re-downloading")
        dest_abs.unlink()

    print(f"  PULL {repo}@{revision[:12]} :: {path_in_repo}")
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=path_in_repo,
            revision=revision,
            local_dir=str(STAGING_DIR),
        )
    )
    # `downloaded` is STAGING_DIR / path_in_repo (preserves subdir layout).
    # Move it to the flat ComfyUI-style location we actually want.
    shutil.move(str(downloaded), str(dest_abs))

    size = dest_abs.stat().st_size
    actual = sha256_file(dest_abs)
    print(f"       wrote {human_size(size)}, sha256={actual}")

    if expected_sha != TBD and expected_sha != actual:
        raise SystemExit(
            f"FATAL: {dest_rel} sha mismatch after download\n"
            f"  expected: {expected_sha}\n"
            f"  actual:   {actual}"
        )

    return size, actual


def main() -> int:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    if not LOCK_PATH.exists():
        print(f"ERROR: {LOCK_PATH} not found", file=sys.stderr)
        return 1

    with LOCK_PATH.open("rb") as f:
        lock = tomllib.load(f)

    entries = lock.get("model", [])
    if not entries:
        print("ERROR: no [[model]] entries in models.lock", file=sys.stderr)
        return 1

    VOLUME_ROOT.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Volume root   : {VOLUME_ROOT}")
    print(f"Lock file     : {LOCK_PATH}")
    print(f"Files to pull : {len(entries)}")
    print()

    results: list[tuple[dict, int, str]] = []
    try:
        for entry in entries:
            size, sha = fetch_one(entry)
            results.append((entry, size, sha))
    finally:
        # Remove the (now empty) staging tree
        if STAGING_DIR.exists():
            shutil.rmtree(STAGING_DIR, ignore_errors=True)

    print()
    print("=" * 72)
    print("Summary — paste these values into models.lock")
    print("=" * 72)
    total = 0
    for entry, size, sha in results:
        print(f"  dest    = {entry['dest']}")
        print(f"    bytes  = {size}")
        print(f"    sha256 = \"{sha}\"")
        print()
        total += size
    print(f"Total: {human_size(total)} across {len(results)} files")
    print()
    print("Next steps:")
    print("  1. Update models.lock with the bytes/sha256 values above.")
    print("  2. Commit and push models.lock.")
    print("  3. Terminate this pod — the weights live on the volume now.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

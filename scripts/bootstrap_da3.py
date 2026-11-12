"""Prepare the exact official DA3-BASE source and safetensors for offline inference."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uv_envs/da3/assets.json"
SOURCE = ROOT / "models/Depth-Anything-3"
ASSETS = ROOT / "checkpoints/da3"


def prepare() -> None:
    """Download missing pinned assets and reject existing assets with another identity.

    Raises:
        ValueError: Existing source or model bytes differ from the asset manifest.

    Example:
        >>> prepare()  # Downloads once; subsequent calls only verify the same assets.
    """
    from huggingface_hub import hf_hub_download

    manifest = json.loads(MANIFEST.read_text())
    source = manifest["source"]
    if not SOURCE.exists():
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", source["url"], str(SOURCE)], check=True)
        subprocess.run(["git", "-C", str(SOURCE), "checkout", "--detach", source["revision"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != source["revision"] or dirty:
        raise ValueError(f"DA3 source must be clean at {source['revision']}; observed {revision}: {dirty}")
    ASSETS.mkdir(parents=True, exist_ok=True)
    weights = manifest["weights"]
    for item in weights["files"]:
        path = ASSETS / item["filename"]
        downloaded = not path.exists()
        if downloaded:
            cached = Path(hf_hub_download(repo_id=weights["repo_id"], revision=weights["revision"], filename=item["filename"]))
            candidate = cached
        else:
            candidate = path
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"DA3 asset hash mismatch: {candidate}")
        if downloaded:
            temporary = path.with_suffix(path.suffix + ".part")
            with candidate.open("rb") as source_stream, temporary.open("wb") as destination:
                for chunk in iter(lambda: source_stream.read(1 << 20), b""):
                    destination.write(chunk)
            temporary.replace(path)
    print(f"Verified DA3-BASE source and local model assets: {ASSETS}")


if __name__ == "__main__":
    prepare()

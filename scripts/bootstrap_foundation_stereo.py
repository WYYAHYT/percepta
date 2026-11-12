"""Prepare pinned FoundationStereo research assets with the isolated runtime interpreter."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uv_envs/foundation-stereo/assets.json"
SOURCE = ROOT / "models/FoundationStereo"
ASSETS = ROOT / "checkpoints/foundation-stereo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"asset hash mismatch: {path}: {actual} != {expected}")


def prepare_source(manifest: dict) -> None:
    source = manifest["source"]
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", "--filter=blob:none", source["url"], str(SOURCE)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(SOURCE), "checkout", "--detach", source["revision"]], check=True
        )
    revision = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != source["revision"]:
        raise ValueError(f"existing source is at {revision}; expected {source['revision']}")
    patch = source["offline_dino_patch"]
    path = SOURCE / patch["path"]
    content = path.read_text()
    if patch["replacement"] not in content:
        if content.count(patch["original"]) != 1:
            raise ValueError(f"source differs from the expected offline patch input: {path}")
        path.write_text(content.replace(patch["original"], patch["replacement"]))
    require_hash(path, patch["sha256"])


def prepare_checkpoint(manifest: dict) -> None:
    import gdown

    ASSETS.mkdir(parents=True, exist_ok=True)
    for asset in manifest["checkpoint_assets"]:
        path = ASSETS / asset["filename"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            result = gdown.download(id=asset["google_drive_id"], output=str(temporary), quiet=False)
            if result is None:
                raise RuntimeError(f"official download failed: {asset['filename']}")
            require_hash(temporary, asset["sha256"])
            temporary.replace(path)
        require_hash(path, asset["sha256"])


def prepare_backbone(manifest: dict) -> None:
    from huggingface_hub import hf_hub_download

    asset = manifest["backbone"]
    cache = ASSETS / "hf_cache"
    for item in asset["files"]:
        path = Path(
            hf_hub_download(
                repo_id=asset["repo_id"],
                revision=asset["revision"],
                filename=item["filename"],
                cache_dir=str(cache),
            )
        )
        require_hash(path, item["sha256"])
    # timm requests the model's default revision; bind that alias to the verified snapshot.
    reference = cache / ("models--" + asset["repo_id"].replace("/", "--")) / "refs/main"
    reference.parent.mkdir(parents=True, exist_ok=True)
    if reference.exists() and reference.read_text().strip() != asset["revision"]:
        raise ValueError(f"existing backbone cache points at another revision: {reference}")
    reference.write_text(asset["revision"])


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    print("FoundationStereo research assets: non-commercial research use only.")
    prepare_source(manifest)
    prepare_checkpoint(manifest)
    prepare_backbone(manifest)
    print(f"Verified source and model assets. Runtime config: {ROOT / 'configs/local-foundation-stereo.yaml'}")


if __name__ == "__main__":
    main()

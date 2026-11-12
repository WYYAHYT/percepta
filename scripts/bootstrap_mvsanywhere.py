"""Prepare pinned MVSAnywhere HERO research assets without modifying existing unknown files."""

from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uv_envs/mvsanywhere/assets.json"
SOURCE = ROOT / "models/mvsanywhere"
ASSETS = ROOT / "checkpoints/mvsanywhere"


def require_hash(path: Path, expected: str) -> None:
    """Reject missing or changed deployment bytes before loading them.

    Args:
        path: Existing local asset.
        expected: Expected SHA256 hexadecimal digest.

    Raises:
        ValueError: The file differs from its pinned digest.

    Example:
        >>> require_hash(Path("checkpoint.ckpt"), manifest["checkpoint"]["sha256"])
        # Returns None for the exact official checkpoint.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"asset hash mismatch: {path}: {digest.hexdigest()} != {expected}")


def prepare_repository(path: Path, asset: dict) -> None:
    """Clone one pinned source revision and reject an existing different revision.

    Args:
        path: Destination for the ignored upstream checkout.
        asset: Official source URL and immutable Git revision.

    Example:
        >>> prepare_repository(SOURCE, manifest["source"])
        # Creates or verifies the pinned checkout.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", asset["url"], str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", asset["revision"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if actual != asset["revision"]:
        raise ValueError(f"source revision mismatch: {path}: {actual} != {asset['revision']}")


def prepare_source(manifest: dict) -> None:
    """Apply only the hash-pinned offline initialization changes.

    Args:
        manifest: Pinned source, DINO source and replacement records.

    Example:
        >>> prepare_source(json.loads(MANIFEST.read_text()))
        # Both initial execution and subsequent executions verify identical bytes.
    """
    prepare_repository(SOURCE, manifest["source"])
    prepare_repository(SOURCE / "dinov2", manifest["dinov2"])
    for patch in manifest["source"]["offline_patches"]:
        target = SOURCE / patch["path"]
        content = target.read_text()
        if hashlib.sha256(content.encode()).hexdigest() != patch["sha256"]:
            require_hash(target, patch["original_sha256"])
            for replacement in patch["replacements"]:
                if content.count(replacement["original"]) != replacement["count"]:
                    raise ValueError(f"offline patch input differs: {target}")
                content = content.replace(replacement["original"], replacement["replacement"])
            if hashlib.sha256(content.encode()).hexdigest() != patch["sha256"]:
                raise ValueError(f"offline patch output differs: {target}")
            target.write_text(content)
        require_hash(target, patch["sha256"])


def prepare_checkpoint(manifest: dict) -> None:
    """Download and verify the official research checkpoint atomically.

    Args:
        manifest: Checkpoint URL, filename and expected digest.

    Example:
        >>> prepare_checkpoint(json.loads(MANIFEST.read_text()))
        # An existing verified checkpoint is reused without network access.
    """
    ASSETS.mkdir(parents=True, exist_ok=True)
    asset = manifest["checkpoint"]
    target = ASSETS / asset["filename"]
    if not target.exists():
        temporary = target.with_suffix(target.suffix + ".part")
        if not temporary.exists():
            with urllib.request.urlopen(asset["url"], timeout=120) as source, temporary.open("wb") as output:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    output.write(chunk)
        require_hash(temporary, asset["sha256"])
        temporary.replace(target)
    require_hash(target, asset["sha256"])


def main() -> None:
    """Prepare the pinned research runtime; print a success message on verification.

    Example:
        >>> main()
        # Prints the research-only license notice and successful verification.
    """
    manifest = json.loads(MANIFEST.read_text())
    print("MVSAnywhere HERO: non-commercial research use only.")
    prepare_source(manifest)
    prepare_checkpoint(manifest)
    for asset in manifest["demo_assets"]:
        require_hash(SOURCE / asset["path"], asset["sha256"])
    print("Verified pinned MVSAnywhere source, local DINO source, checkpoint and official calibrated demo.")


if __name__ == "__main__":
    main()

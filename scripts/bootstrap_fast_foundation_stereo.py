"""Download and verify the pinned official Fast-FoundationStereo research release."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uv_envs/fast-foundation-stereo/assets.json"
SOURCE = ROOT / "models/Fast-FoundationStereo"
ASSETS = ROOT / "checkpoints/fast-foundation-stereo"


def require_hash(path: Path, expected: str) -> None:
    """Reject bytes differing from a manifest digest without modifying the file.

    Args:
        path: Existing asset to read.
        expected: Expected complete SHA256 hex digest.

    Raises:
        ValueError: The existing bytes differ from the release manifest.

    Example:
        >>> require_hash(Path("cfg.yaml"), "the digest from assets.json")  # doctest: +SKIP
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"asset hash mismatch: {path}: {digest.hexdigest()} != {expected}")


def prepare_source(manifest: dict) -> None:
    """Clone the pinned upstream checkout or validate the existing, unmodified source.

    Args:
        manifest: Asset manifest containing source URL, revision and file digests.

    Raises:
        ValueError: An existing checkout has a different revision or modified tracked files.

    Example:
        >>> prepare_source(json.loads(MANIFEST.read_text()))  # doctest: +SKIP
    """
    source = manifest["source"]
    SOURCE.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", source["url"], str(SOURCE)], check=True)
        subprocess.run(["git", "-C", str(SOURCE), "checkout", "--detach", source["revision"]], check=True)
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    if revision != source["revision"]:
        raise ValueError(f"existing source is at {revision}; expected {source['revision']}")
    changed = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed.strip():
        raise ValueError("existing Fast-FoundationStereo source has modified tracked files")
    for asset in source.get("files", []):
        require_hash(SOURCE / asset["filename"], asset["sha256"])


def prepare_checkpoint(manifest: dict) -> None:
    """Download missing official assets and publish them only after digest verification.

    Args:
        manifest: Manifest with official Google Drive IDs and complete SHA256 digests.

    Raises:
        RuntimeError: The official download failed.
        ValueError: Existing or downloaded bytes differ from the pinned release.

    Example:
        >>> prepare_checkpoint(json.loads(MANIFEST.read_text()))  # doctest: +SKIP
    """
    import gdown

    ASSETS.mkdir(parents=True, exist_ok=True)
    for asset in manifest["checkpoint_assets"]:
        path = ASSETS / asset["filename"]
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".part")
            if gdown.download(id=asset["google_drive_id"], output=str(temporary), quiet=False) is None:
                raise RuntimeError(f"official download failed: {asset['filename']}")
            require_hash(temporary, asset["sha256"])
            temporary.replace(path)
        require_hash(path, asset["sha256"])


def main() -> None:
    """Prepare the pinned upstream source and research assets without modifying other runtimes.

    Example:
        >>> main()  # doctest: +SKIP
    """
    manifest = json.loads(MANIFEST.read_text())
    print("Fast-FoundationStereo official research assets: non-commercial research only.")
    prepare_source(manifest)
    prepare_checkpoint(manifest)
    print(f"Verified assets. Configuration: {ROOT / 'configs/local-fast-foundation-stereo.yaml'}")


if __name__ == "__main__":
    main()

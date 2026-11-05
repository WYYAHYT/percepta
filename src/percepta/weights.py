"""Model asset management: where weights live, and whether they are the right ones.

The library does **not** manage a model's *dependencies* — that is the deployer's concern
(see ``docs/model-isolation.md``). It does manage the *contract* around model assets, because
those are the things that fail silently when they are wrong:

* **location** — weights are never in the repository; every tool receives a path;
* **integrity** — a truncated or wrong-version file can load "successfully" and then produce
  quietly wrong segments. An expected sha256 turns that into a startup error;
* **traceability** — the observed hash is recorded in ``Provenance.model_sha256`` either way,
  so a result can always be traced to the exact bytes that produced it.

Three config shapes are accepted per tool, so a deployment can be explicit without being
verbose:

.. code-block:: yaml

    tools:
      sam3:
        checkpoint: /models/sam3.pt          # path only: hash is recorded, not checked
        checkpoint_sha256: "abc123..."       # optional: hash is verified at load
      other:
        checkpoint: hf://facebook/sam3       # URI: passed through untouched (http backend)

The strictness knob is ``tools.<name>.verify_weights``:

``if-declared`` (default)
    Verify when ``checkpoint_sha256`` is configured, otherwise record only. This keeps
    read-only exploration usable while making production configs safe by default.

``always``
    Refuse to load a *local file* whose hash is not declared. Use this in CI/production.

``never``
    Skip verification entirely (escape hatch for very large weights on slow disks).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "WeightSpec",
    "WeightError",
    "WeightNotFound",
    "WeightMismatch",
    "VerifyMode",
    "resolve_weights",
    "file_sha256",
    "is_local_path",
]

log = logging.getLogger(__name__)

VerifyMode = Literal["if-declared", "always", "never"]

_VERIFY_MODES = ("if-declared", "always", "never")

#: URI schemes that mean "not a local file": the backend resolves them, we do not.
_REMOTE_SCHEMES = ("http://", "https://", "hf://", "s3://", "gs://")


class WeightError(RuntimeError):
    """Weights are unusable. Maps to Percept ``status="unavailable"`` (spec §16.4)."""


class WeightNotFound(WeightError):
    """The configured weights do not exist at the configured location."""


class WeightMismatch(WeightError):
    """The weights exist but are not the bytes the caller expected."""


@dataclass(frozen=True)
class WeightSpec:
    """A resolved model asset: where it is, what it should hash to, whether that was checked."""

    location: str
    """As configured: a local path or a remote URI."""

    path: Path | None
    """Absolute path when :attr:`location` is a local file, else ``None``."""

    expected_sha256: str | None
    observed_sha256: str | None
    verified: bool
    """``True`` only when an expected hash was actually compared."""

    def describe(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "path": None if self.path is None else str(self.path),
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "verified": self.verified,
            "remote": self.path is None,
        }


def is_local_path(location: str) -> bool:
    """True when ``location`` names a filesystem path rather than a remote URI."""
    lowered = str(location).strip().lower()
    return not lowered.startswith(_REMOTE_SCHEMES)


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (constant memory, so 10 GB weights are fine)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_mode(config: dict[str, Any]) -> VerifyMode:
    raw = config.get("verify_weights", "if-declared")
    mode = str(raw).strip().lower()
    if mode not in _VERIFY_MODES:
        raise WeightError(
            f"unknown verify_weights {raw!r}; expected one of {', '.join(_VERIFY_MODES)}"
        )
    return mode  # type: ignore[return-value]


def resolve_weights(
    location: str | Path | None,
    *,
    expected_sha256: str | None = None,
    verify: VerifyMode | str = "if-declared",
    config: dict[str, Any] | None = None,
    what: str = "model weights",
) -> WeightSpec:
    """Validate a model asset and return a :class:`WeightSpec`.

    ``location`` may be ``None`` for tools that ship no weights (e.g. the geometry tool), in
    which case a spec with no path is returned and nothing is checked — callers that *require*
    weights should reject ``None`` themselves before calling.

    Raises :class:`WeightNotFound` / :class:`WeightMismatch`; never returns a spec whose
    ``verified`` flag overstates what happened.
    """
    config = dict(config or {})
    if location is None or str(location).strip() == "":
        return WeightSpec(
            location="",
            path=None,
            expected_sha256=None,
            observed_sha256=None,
            verified=False,
        )

    text = str(location).strip()
    # NOTE: read the expected hash through an explicit ``is not None`` check, not ``or``.
    # YAML parses an all-digit hash (e.g. 64 zeros) as an *integer*, and ``int(0)`` is falsy —
    # with an ``or`` chain the expected hash would silently vanish and verification would be
    # skipped without a word. Coercing to str first turns that into a loud config error.
    raw_expected = (
        expected_sha256 if expected_sha256 is not None else config.get("checkpoint_sha256")
    )
    expected: str | None = None
    if raw_expected is not None:
        candidate = str(raw_expected).strip().lower()
        expected = candidate or None
    if expected is not None:
        if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise WeightError(
                f"{what}: expected sha256 must be 64 hex characters, got {expected!r} "
                "(quote the value in YAML so a numeric-looking hash stays a string)"
            )
    mode = _VERIFY_MODE_CACHE.get(verify) or _resolve_mode({"verify_weights": verify})

    if not is_local_path(text):
        # A URI is the backend's business (http/hf resolve it); we cannot hash it here.
        log.debug("%s: remote location %s passed through unverified", what, text)
        return WeightSpec(
            location=text,
            path=None,
            expected_sha256=expected,
            observed_sha256=None,
            verified=False,
        )

    path = Path(text).expanduser()
    if not path.is_file():
        raise WeightNotFound(
            f"{what} not found: {path}. Mount the checkpoint read-only and point "
            f"`checkpoint:` at it; weights are never distributed with percepta (spec §12)."
        )
    # Resolve symlinks so the recorded path is the real artifact.
    path = path.resolve()

    if mode == "never":
        return WeightSpec(
            location=text,
            path=path,
            expected_sha256=expected,
            observed_sha256=None,
            verified=False,
        )

    if mode == "always" and expected is None:
        raise WeightError(
            f"{what}: verify_weights='always' requires checkpoint_sha256 for {path}"
        )

    observed = file_sha256(path)
    verified = False
    if expected is not None:
        if observed != expected:
            raise WeightMismatch(
                f"{what} checksum mismatch for {path}:\n"
                f"  expected sha256 {expected}\n"
                f"  observed sha256 {observed}\n"
                "Refusing to load: a truncated or wrong-version checkpoint can load "
                "successfully and then produce silently wrong results."
            )
        verified = True

    return WeightSpec(
        location=text,
        path=path,
        expected_sha256=expected,
        observed_sha256=observed,
        verified=verified,
    )


_VERIFY_MODE_CACHE: dict[str, str] = {mode: mode for mode in _VERIFY_MODES}

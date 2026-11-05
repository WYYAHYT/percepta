"""The ``sam3`` segmenter tool — the default real segmenter (spec §5.4, appendix C).

Two properties matter here:

1. **Import safety.** ``import torch`` / ``import sam3`` live inside :class:`_Sam3Backend`,
   never at module import time, so ``ToolRegistry.discover()`` can import this module on a
   GPU-less CI box (spec §16.2). Without weights it reports ``ToolUnavailable``, which the
   pipeline surfaces as a Percept with ``status="unavailable"`` — never as ``not_found``.
2. **Backend seam.** All model access goes through a tiny backend object, so the prompt
   handling, score filtering, mask encoding and provenance logic *can* be unit-tested with a
   fake backend (``tests/unit/test_sam3_backend.py``) while the real inference path stays
   covered by the ``gpu``-marked smoke test (§14 S9).

Upstream shape, per appendix C: ``build_sam3_image_model(device, checkpoint_path,
load_from_HF=False, enable_inst_interactivity=True)``, ``Sam3Processor(resolution=1008)``,
``set_image`` once and then reuse the state for ``set_text_prompt`` /
``add_geometric_prompt(box=[cx, cy, w, h])``; outputs are ``masks (N,1,H,W) bool`` at the
original resolution plus ``boxes`` (xyxy) and ``scores``. If the upstream API moves, only
this file changes — that is exactly why the spec forbids leaking it into the schema.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..backends import BackendRequestError, BackendUnavailable, resolve_backend_kind
from ..backends.http_backend import HttpBackend
from ..backends.subprocess_backend import SubprocessBackend
from ..schema import Percept, Prompt, ToolCapabilities, ToolContext
from .base import (
    BaseTool,
    ToolUnavailable,
    apply_top_k,
    make_observation,
    make_percept,
)

__all__ = ["Sam3Tool", "Sam3BackendError"]

#: Warning emitted when a box prompt is handed to a build without geometric interactivity.
NO_GEOMETRIC_PROMPT = "sam3_geometric_prompt_unavailable"


class Sam3BackendError(RuntimeError):
    """The upstream SAM3 build could not satisfy the request (API drift, bad state)."""


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


class _Sam3Backend:
    """Real in-process adapter: heavy imports happen in ``__init__`` only (called from ``load()``).

    It satisfies :class:`percepta.backends.base.ModelBackend`, which is what lets the same
    tool run this object directly, or ask a worker in another virtualenv to run it
    (``docs/model-isolation.md``).
    """

    name = "sam3-inprocess"

    def __init__(self, checkpoint: str | Path, device: str, resolution: int) -> None:
        try:
            import sam3  # noqa: PLC0415
            import torch  # noqa: PLC0415 - deliberately lazy (spec §16.2)
        except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
            raise ToolUnavailable(
                "sam3 is not importable; install the 'sam3' extra and provide a checkpoint "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise ToolUnavailable(f"SAM3 checkpoint not found: {checkpoint_path}")

        self._torch = torch
        self.device = device
        self.resolution = int(resolution)
        self.checkpoint = str(checkpoint_path)
        self.model_sha256 = _sha256_file(checkpoint_path)

        self.model = sam3.build_sam3_image_model(
            device,
            str(checkpoint_path),
            load_from_HF=False,
            enable_inst_interactivity=True,
        )
        self.processor = sam3.Sam3Processor(resolution=self.resolution)
        self._state: Any = None
        self._supports_geometric = True

    # -- inference -------------------------------------------------------

    def set_image(self, image_rgb: np.ndarray) -> None:
        torch = self._torch
        with torch.inference_mode():
            self._state = self.processor.set_image(image_rgb)

    def segment(
        self, prompt: Prompt
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Return ``(masks (N,H,W) bool, boxes xyxy (N,4) or None, scores (N,))``."""
        torch = self._torch
        if self._state is None:
            raise Sam3BackendError("set_image() must be called before segment()")

        with torch.inference_mode():
            if prompt.type == "text":
                self._state = self.processor.set_text_prompt(
                    state=self._state, prompt=prompt.text
                )
            elif prompt.type == "box":
                if not self._supports_geometric:
                    raise Sam3BackendError(NO_GEOMETRIC_PROMPT)
                try:
                    self._state = self.processor.add_geometric_prompt(
                        state=self._state, box=[float(v) for v in (prompt.box or [])]
                    )
                except (AttributeError, TypeError) as exc:
                    self._supports_geometric = False
                    raise Sam3BackendError(NO_GEOMETRIC_PROMPT) from exc
            else:  # pragma: no cover - the tool rejects these before we get here
                raise Sam3BackendError(f"unsupported prompt type {prompt.type!r}")

            masks = self._state.get("masks")
            boxes = self._state.get("boxes")
            scores = self._state.get("scores")
            if masks is None:
                raise Sam3BackendError("sam3 returned no masks")

            mask_array = masks.detach().cpu().numpy() if hasattr(masks, "detach") else np.asarray(masks)
            if mask_array.ndim == 4:
                mask_array = mask_array[:, 0]
            score_array = (
                scores.detach().cpu().numpy().reshape(-1)
                if hasattr(scores, "detach")
                else np.asarray(scores if scores is not None else np.zeros(len(mask_array))).reshape(-1)
            )
            box_array = None
            if boxes is not None:
                box_array = (
                    boxes.detach().cpu().numpy()
                    if hasattr(boxes, "detach")
                    else np.asarray(boxes)
                ).reshape(-1, 4)
        return mask_array.astype(bool), box_array, score_array.astype(float)

    def close(self) -> None:
        self._state = None
        self.model = None
        self.processor = None
        torch = getattr(self, "_torch", None)
        if torch is not None and torch.cuda.is_available():  # pragma: no cover - GPU only
            torch.cuda.empty_cache()


class Sam3Tool(BaseTool):
    name = "sam3"
    version = "0.1.0"
    kind = "segmenter"

    def __init__(self, backend_factory: Callable[..., Any] | None = None) -> None:
        super().__init__()
        self._backend_factory = backend_factory
        self._backend: Any = None
        self._lock = threading.Lock()
        self._image_key: tuple[str, int] | None = None
        self._warned: set[str] = set()

    def capabilities(self) -> ToolCapabilities:
        return ToolCapabilities(
            kind="segmenter",
            prompts={"text", "box"},
            masks=True,
            boxes=True,
            points=False,  # official Sam3Processor has no point prompt (spec §19.1)
            depth=False,
            multi_view=False,
            geometry=False,
        )

    # -- lifecycle -------------------------------------------------------

    def load(self, config: dict[str, Any]) -> None:
        super().load(config)
        # ``backend_factory`` was the original CI seam (a fake adapter); keep it working and
        # give it priority so tests never spawn a process.
        if self._backend_factory is not None:
            self._backend = self._backend_factory(
                checkpoint=config.get("checkpoint"),
                device=config.get("device") or "cpu",
                resolution=int(config.get("resolution", 1008)),
            )
            self._image_key = None
            return

        kind = resolve_backend_kind(config)
        if kind == "inprocess":
            checkpoint = config.get("checkpoint")
            if not checkpoint:
                raise ToolUnavailable(
                    "sam3 tool requires tools.sam3.checkpoint (weights are never vendored)"
                )
            device = config.get("device") or self._default_device()
            self._backend = _Sam3Backend(
                checkpoint=checkpoint,
                device=device,
                resolution=int(config.get("resolution", 1008)),
            )
        elif kind == "subprocess":
            self._backend = SubprocessBackend(
                python=str(config.get("python") or "python3"),
                module=str(config.get("worker_module") or "percepta.backends.worker"),
                config={
                    "backend_factory": str(
                        config.get("worker_backend_factory")
                        or "percepta.tools.sam3:_Sam3Backend"
                    ),
                    "backend_args": {
                        "checkpoint": config.get("checkpoint"),
                        "device": config.get("device") or "cuda",
                        "resolution": int(config.get("resolution", 1008)),
                    },
                },
                timeout_s=float(config.get("timeout_s", 120.0)),
                cuda_visible_devices=config.get("cuda_visible_devices"),
            )
        elif kind == "http":
            endpoint = config.get("endpoint")
            if not endpoint:
                raise ToolUnavailable(
                    "sam3 backend 'http' requires tools.sam3.endpoint "
                    "(e.g. http://127.0.0.1:8801)"
                )
            self._backend = HttpBackend(
                endpoint=str(endpoint),
                timeout_s=float(config.get("timeout_s", 120.0)),
                token=config.get("token"),
            )
        else:  # pragma: no cover - resolve_backend_kind already validated
            raise ToolUnavailable(f"unsupported backend {kind!r}")

        self._image_key = None
        if config.get("warmup", False):  # pragma: no cover - GPU only
            self._warmup()

    @staticmethod
    def _default_device() -> str:
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - torch absent: the backend will report it
            return "cpu"

    def close(self) -> None:
        with self._lock:
            if self._backend is not None:
                try:
                    self._backend.close()
                finally:
                    self._backend = None
            self._image_key = None
            self._loaded = False

    def _warmup(self) -> None:  # pragma: no cover - GPU only
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        self._backend.set_image(dummy)
        self._backend.segment(Prompt(type="text", text="object"))
        self._image_key = None

    # -- inference -------------------------------------------------------

    def _model_meta(self) -> tuple[str | None, str | None]:
        backend = self._backend
        model = getattr(backend, "checkpoint", None) or self.config.get("checkpoint")
        sha = getattr(backend, "model_sha256", None) or self.config.get("model_sha256")
        return (
            None if model is None else str(model),
            None if sha is None else str(sha),
        )

    def perceive(self, ctx: ToolContext, prompts: Sequence[Prompt]) -> list[Percept]:
        if self._backend is None:
            raise ToolUnavailable("sam3 tool is not loaded")
        percepts: list[Percept] = []
        with self._lock:  # one model, one inference at a time (§11.2)
            for camera in sorted(ctx.images):
                image = ctx.images[camera]
                started = time.perf_counter()
                self._set_image_once(camera, image)
                for prompt in prompts:
                    percepts.append(self._perceive_one(ctx, camera, prompt, started))
        return percepts

    def _set_image_once(self, camera: str, image: np.ndarray) -> None:
        """``set_image`` is the expensive call: reuse the state for every prompt (§appendix C)."""
        key = (camera, int(hash(image.tobytes())) if image.size < 4096 else id(image))
        if self._image_key != key:
            self._backend.set_image(image)
            self._image_key = key

    def _perceive_one(
        self, ctx: ToolContext, camera: str, prompt: Prompt, started_at: float
    ) -> Percept:
        percept_id = str(uuid.uuid4())
        mask_b64 = ctx.options.masks
        try:
            masks, boxes, scores = self._backend.segment(prompt)
        except (Sam3BackendError, BackendRequestError) as exc:
            model, model_sha = self._model_meta()
            provenance = self._provenance(
                ctx,
                started_at,
                camera=camera,
                prompt=prompt,
                model=model,
                model_sha256=model_sha,
            )
            return make_percept(
                percept_id=percept_id,
                status="invalid_request",
                provenance=provenance,
                camera=camera,
                message=str(exc),
                extras={"code": "unsupported_prompt"},
            )
        except BackendUnavailable as exc:
            # A dead worker / unreachable service is "I could not look", which must never be
            # reported as ``not_found`` (spec §16.4).
            model, model_sha = self._model_meta()
            provenance = self._provenance(
                ctx,
                started_at,
                camera=camera,
                prompt=prompt,
                model=model,
                model_sha256=model_sha,
            )
            return make_percept(
                percept_id=percept_id,
                status="unavailable",
                provenance=provenance,
                camera=camera,
                message=str(exc),
                extras={"code": "tool_unavailable"},
            )

        order = np.argsort(-scores) if scores.size else np.array([], dtype=int)
        observations = []
        for rank, index in enumerate(order.tolist()):
            mask = masks[index] if index < len(masks) else None
            if mask is None or not mask.any():
                continue
            score = float(scores[index]) if index < scores.size else 0.0
            box = (
                [float(v) for v in boxes[index]]
                if boxes is not None and index < len(boxes)
                else None
            )
            observations.append(
                make_observation(
                    mask,
                    score,
                    version_percept_id=percept_id,
                    index=rank,
                    encoding=mask_b64,
                    label=(prompt.text if prompt.type == "text" else "box_prompt"),
                    box_xyxy=box,
                )
            )

        observations = apply_top_k(observations, prompt)
        model, model_sha = self._model_meta()
        provenance = self._provenance(
            ctx,
            started_at,
            camera=camera,
            prompt=prompt,
            model=model,
            model_sha256=model_sha,
        )
        status = "ok" if observations else "not_found"
        return make_percept(
            percept_id=percept_id,
            status=status,
            provenance=provenance,
            camera=camera,
            message=None if observations else "sam3 found no instance above min_score",
            observations=observations,
        )

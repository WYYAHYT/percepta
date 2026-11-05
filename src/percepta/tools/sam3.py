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

import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from ..backends import BackendRequestError, BackendUnavailable, resolve_backend_kind
from ..backends.http_backend import HttpBackend
from ..backends.subprocess_backend import SubprocessBackend
from ..schema import Percept, Prompt, ToolCapabilities, ToolContext
from ..weights import WeightError, resolve_weights
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




def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor (or anything array-like) to numpy, coping with bf16.

    ``Tensor.numpy()`` raises ``TypeError: Got unsupported ScalarType BFloat16`` — SAM3 runs in
    bf16 autocast, so its ``boxes``/``scores`` come back as bfloat16. There is no numpy bf16
    dtype, so we upcast to float32 first. Bool masks pass through unchanged.
    """
    if value is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        # bfloat16 has no numpy dtype; upcast before converting.
        if str(tensor.dtype) == "torch.bfloat16":
            tensor = tensor.float()
        return tensor.numpy()
    return np.asarray(value)


class _Sam3Backend:
    """Real in-process adapter: heavy imports happen in ``__init__`` only (called from ``load()``).

    It satisfies :class:`percepta.backends.base.ModelBackend`, which is what lets the same
    tool run this object directly, or ask a worker in another virtualenv to run it
    (``docs/model-isolation.md``).
    """

    name = "sam3-inprocess"

    def __init__(
        self,
        checkpoint: str | Path,
        device: str,
        resolution: int,
        checkpoint_sha256: str | None = None,
        verify_weights: str = "if-declared",
        bpe_path: str | Path | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        try:
            import sam3  # noqa: PLC0415
            import torch  # noqa: PLC0415 - deliberately lazy (spec §16.2)

            # ``sam3/__init__.py`` re-exports only ``build_sam3_image_model``; the processor
            # lives in a submodule. Importing it lazily keeps this module import-safe while
            # still failing with a clear message when the package is absent.
            from sam3.model.sam3_image_processor import (  # noqa: PLC0415
                Sam3Processor,
            )
        except Exception as exc:  # noqa: BLE001 - any import failure means "unavailable"
            raise ToolUnavailable(
                "sam3 is not importable; install the 'sam3' extra and provide a checkpoint "
                f"({type(exc).__name__}: {exc})"
            ) from exc

        # Validate BEFORE importing the framework path is used, so a wrong checkpoint fails
        # with a clear message instead of an obscure torch load error.
        try:
            spec = resolve_weights(
                checkpoint,
                expected_sha256=checkpoint_sha256,
                verify=verify_weights,
                what="SAM3 checkpoint",
            )
        except WeightError as exc:
            raise ToolUnavailable(str(exc)) from exc
        if spec.path is None:
            raise ToolUnavailable(
                f"SAM3 checkpoint must be a local file, got {spec.location!r}"
            )

        self._torch = torch
        self.device = device
        self.resolution = int(resolution)
        self.checkpoint = str(spec.path)
        # Record the observed hash in provenance; ``weights_verified`` says whether it was
        # actually compared against an expected value (never overstate this).
        self.model_sha256 = spec.observed_sha256
        self.weights_verified = spec.verified
        self._weights_spec = spec

        # NOTE on the upstream API (verified against sam3 0.1.4 source, not just the docs):
        #   build_sam3_image_model(bpe_path=None, device=..., eval_mode=True,
        #                          checkpoint_path=None, load_from_HF=True,
        #                          enable_segmentation=True,
        #                          enable_inst_interactivity=False, compile=False)
        # The first positional argument is ``bpe_path``, NOT ``device`` — passing positionally
        # silently treated the device string as a tokenizer path. All of these are keyword.
        self.model = sam3.build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            eval_mode=True,
            checkpoint_path=str(spec.path),
            load_from_HF=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )
        # ``Sam3Processor.__init__(self, model, resolution=1008, device="cuda",
        # confidence_threshold=0.5)`` — ``model`` is required and positional.
        self.processor = Sam3Processor(
            self.model,
            resolution=self.resolution,
            device=device,
            confidence_threshold=float(confidence_threshold),
        )
        self._state: Any = None
        self._supports_geometric = True

    # -- inference -------------------------------------------------------

    def set_image(self, image_rgb: np.ndarray, depth: np.ndarray | None = None) -> None:
        """Prepare a frame. ``depth`` is accepted for protocol conformance and ignored:
        SAM3 is RGB-only (declared in ``capabilities().depth = False``).

        Input format is a real trap. ``Sam3Processor.set_image`` feeds its argument to
        ``torchvision.transforms.v2.functional.to_image``, which expects a **CHW** tensor or a
        PIL image. Given a plain HWC numpy array it does not complain — it reads
        ``shape[-2:]`` as ``(H, W)`` for bookkeeping while ``to_image`` treats the axes as
        ``(C, H, W)``. The result is silent nonsense: a ``(480, 640, 3)`` frame becomes
        C=480, H=640, W=3, and every prompt returns zero instances.

        We therefore hand over a PIL image, which is unambiguous.
        """
        _ = depth
        torch = self._torch
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise Sam3BackendError(
                f"expected RGB image (H, W, 3), got shape {image.shape}"
            )
        pil = PILImage.fromarray(image.astype(np.uint8, copy=False), mode="RGB")
        with torch.inference_mode():
            self._state = self.processor.set_image(pil)

    def segment(
        self, prompt: Prompt
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Return ``(masks (N,H,W) bool, boxes xyxy (N,4) or None, scores (N,))``."""
        torch = self._torch
        if self._state is None:
            raise Sam3BackendError("set_image() must be called before segment()")

        with torch.inference_mode():
            if prompt.type == "text":
                # Verified signature: set_text_prompt(self, prompt: str, state: Dict)
                self._state = self.processor.set_text_prompt(
                    prompt=prompt.text, state=self._state
                )
            elif prompt.type == "box":
                if not self._supports_geometric:
                    raise Sam3BackendError(NO_GEOMETRIC_PROMPT)
                try:
                    # Verified signature: add_geometric_prompt(self, box: List, label: bool,
                    # state: Dict) — ``label`` is REQUIRED and means positive/negative
                    # example, which is exactly Prompt.label's meaning (spec §3.2).
                    self._state = self.processor.add_geometric_prompt(
                        box=[float(v) for v in (prompt.box or [])],
                        label=bool(prompt.label),
                        state=self._state,
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

            mask_array = _to_numpy(masks)
            if mask_array.ndim == 4:
                mask_array = mask_array[:, 0]
            score_array = _to_numpy(scores).reshape(-1)
            box_array = None if boxes is None else _to_numpy(boxes).reshape(-1, 4)
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

        # Resolve weights in the PARENT for every backend: a subprocess/http worker keeps the
        # model in another process, so this is the only place the parent can record (and
        # optionally verify) the exact bytes behind a result.
        if kind != "inprocess" and config.get("checkpoint"):
            try:
                spec = resolve_weights(
                    config.get("checkpoint"),
                    expected_sha256=config.get("checkpoint_sha256"),
                    verify=str(config.get("verify_weights", "if-declared")),
                    what="sam3 checkpoint",
                )
                self._resolved_checkpoint = str(spec.path or spec.location)
                self._resolved_model_sha256 = spec.observed_sha256
            except WeightError as exc:
                raise ToolUnavailable(str(exc)) from exc

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
                checkpoint_sha256=config.get("checkpoint_sha256"),
                verify_weights=str(config.get("verify_weights", "if-declared")),
                # The PyPI wheel ships no tokenizer vocab; build_sam3_image_model() looks for
                # ../assets/bpe_simple_vocab_16e6.txt.gz and fails if it is absent. Point it at
                # a real file (e.g. the vocab.json/merges.txt pair from the HF repo).
                bpe_path=config.get("bpe_path"),
                confidence_threshold=float(
                    config.get("confidence_threshold", config.get("min_score", 0.5))
                ),
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
                        "checkpoint_sha256": config.get("checkpoint_sha256"),
                        "verify_weights": str(config.get("verify_weights", "if-declared")),
                        "bpe_path": config.get("bpe_path"),
                        "confidence_threshold": float(
                            config.get("confidence_threshold", config.get("min_score", 0.5))
                        ),
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
        # NOTE: there used to be a ``warmup`` flag here that ran a 32x32 all-zero frame. It was
        # worse than useless: a tiny black image exercises different kernel shapes than a real
        # frame (so the expensive autotuning still happened on the first real request), it
        # tripled the checkpoint read cost by forcing a second set_image, and it could not be
        # observed by anyone. Warm-up now belongs to the service layer
        # (:mod:`percepta.warmup`), which sends a realistic frame through the whole pipeline
        # and reports the outcome on /healthz.

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

    # -- inference -------------------------------------------------------

    def _model_meta(self) -> tuple[str | None, str | None]:
        backend = self._backend
        model = (
            getattr(backend, "checkpoint", None)
            or self.config.get("checkpoint")
            or getattr(self, "_resolved_checkpoint", None)
        )
        # A subprocess/http backend keeps the model in *another* process, so its
        # ``model_sha256`` is not visible here. The parent resolved the hash itself at load
        # time (``_resolved_model_sha256``), which is what provenance must report — otherwise
        # a remote worker's results would carry no traceable weight identity.
        sha = (
            getattr(backend, "model_sha256", None)
            or self.config.get("model_sha256")
            or getattr(self, "_resolved_model_sha256", None)
        )
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

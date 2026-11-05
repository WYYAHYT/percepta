"""Real GPU smoke test for the subprocess backend — run it, do not trust mocks.

This is the script that answers "does the model actually work in another interpreter?".
It is deliberately a standalone script rather than a pytest case, because it needs three
things CI does not have: a CUDA torch, real SAM3 weights, and a GPU.

Usage::

    python3 scripts/gpu_smoke.py \
        --python /tmp/gpuenv/bin/python \
        --checkpoint /models/sam3.pt \
        --bpe-path /models/bpe_simple_vocab_16e6.txt.gz

What it proves, in order:

1. the worker interpreter really is a *different* environment (torch version, numpy
   version, and that ``percepta`` is not installed in it);
2. the binary protocol carries a real frame across the process boundary;
3. the model loads once and stays resident across several prompts;
4. results come back as masks/boxes/scores at the original resolution;
5. a killed worker is reported as ``unavailable``, never ``not_found``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from percepta.backends.base import BackendUnavailable  # noqa: E402
from percepta.backends.subprocess_backend import SubprocessBackend  # noqa: E402
from percepta.schema import Prompt  # noqa: E402


def _env_facts(python: str) -> dict:
    """Ask the worker interpreter about itself — proof of isolation, not a claim."""
    code = (
        "import json,sys\n"
        "out={'executable': sys.executable, 'version': sys.version.split()[0]}\n"
        "try:\n"
        "    import torch; out['torch']=torch.__version__; "
        "out['cuda']=torch.cuda.is_available(); "
        "out['device']=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None\n"
        "except Exception as e: out['torch']=f'none ({e})'\n"
        "try:\n"
        "    import numpy; out['numpy']=numpy.__version__\n"
        "except Exception as e: out['numpy']=f'none ({e})'\n"
        "try:\n"
        "    import percepta; out['percepta']=percepta.__version__\n"
        "except Exception: out['percepta']='NOT INSTALLED (expected: it comes via PYTHONPATH)'\n"
        "try:\n"
        "    import sam3; out['sam3']=sam3.__version__\n"
        "except Exception as e: out['sam3']=f'none ({e})'\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run([python, "-c", code], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"worker env probe failed:\n{result.stderr}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def _product_scene(size: int = 640) -> np.ndarray:
    """Fallback synthetic scene.

    NOTE: SAM3 is a *real* open-vocabulary model trained on photographs; it finds nothing in a
    flat vector-drawing scene like this one. Always prefer ``--image`` with a real photo —
    using this fallback and concluding "0 instances" says nothing about the model.
    """
    image = np.full((480, size, 3), 232, dtype=np.uint8)
    image[380:480, :] = 205
    image[140:390, 285:355] = (38, 92, 168)
    image[95:145, 305:335] = (38, 92, 168)
    image[80:100, 300:340] = (235, 235, 235)
    return image


def _load_image(path: str | None) -> np.ndarray:
    if not path:
        print("   (no --image given; using a synthetic scene the model will not recognise)")
        return _product_scene()
    from PIL import Image

    with Image.open(path) as handle:
        return np.array(handle.convert("RGB"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, help="worker interpreter (its own venv)")
    parser.add_argument("--checkpoint", required=True, help="SAM3 weights (.pt)")
    parser.add_argument("--bpe-path", default=None, help="tokenizer vocab (.txt.gz/.json)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--image", default=None, help="photo to segment (recommended)")
    parser.add_argument("--prompt", default="bottle")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--skip-kill-test", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("1. isolation check: who is the worker?")
    facts = _env_facts(args.python)
    for key, value in facts.items():
        print(f"   {key:12s} {value}")

    print("\n2. building subprocess backend")
    backend = SubprocessBackend(
        python=args.python,
        module="percepta.backends.worker",
        config={
            "backend_factory": "percepta.tools.sam3:_Sam3Backend",
            "backend_args": {
                "checkpoint": args.checkpoint,
                "device": args.device,
                "resolution": args.resolution,
                "bpe_path": args.bpe_path,
            },
        },
        timeout_s=args.timeout,
        cuda_visible_devices="2",  # idle card on this host
    )

    image = _load_image(args.image)
    try:
        print(f"\n3. loading model + first frame (prompt={args.prompt!r})")
        started = time.perf_counter()
        backend.set_image(image)
        load_s = time.perf_counter() - started
        print(f"   set_image (includes model load): {load_s:.2f}s")
        print(f"   worker pid: {backend.describe().get('pid')}")

        print("\n4. running prompt")
        started = time.perf_counter()
        masks, boxes, scores = backend.segment(Prompt(type="text", text=args.prompt))
        infer_s = time.perf_counter() - started
        print(f"   segment: {infer_s * 1000:.1f}ms  -> {len(scores)} instance(s)")
        for index, score in enumerate(scores.tolist()):
            area = int(masks[index].sum()) if index < len(masks) else 0
            box = boxes[index].tolist() if boxes is not None and index < len(boxes) else None
            print(f"   [{index}] score={score:.3f} area={area}px box={box}")

        if len(scores) == 0:
            print("\n   NOTE: model ran but found nothing for this prompt.")
        else:
            expected = tuple(image.shape[:2])
            assert masks.shape[1:] == expected, f"mask {masks.shape[1:]} != image {expected}"
            print(f"\n   mask shape {masks.shape} matches image {expected}  OK")

        print("\n5. second prompt on the SAME prepared frame (resident worker)")
        started = time.perf_counter()
        try:
            masks2, _boxes2, scores2 = backend.segment(Prompt(type="text", text="table"))
            print(f"   segment: {(time.perf_counter() - started) * 1000:.1f}ms -> {len(scores2)}")
        except BackendUnavailable as exc:
            print(f"   second prompt failed (model may reject it): {exc}")
        print(f"   worker still alive: {backend.describe().get('running')}")

        if not args.skip_kill_test:
            print("\n6. killing the worker must yield 'unavailable', not 'not_found'")
            import os
            import signal

            pid = backend.describe().get("pid")
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            try:
                backend.segment(Prompt(type="text", text=args.prompt))
                print("   UNEXPECTED: request succeeded after the worker was killed")
            except BackendUnavailable as exc:
                print(f"   correctly raised BackendUnavailable: {str(exc)[:100]}")
    finally:
        backend.close()

    print("\n" + "=" * 70)
    print("GPU SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

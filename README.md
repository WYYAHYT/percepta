# percepta

A pluggable **observation layer** for robot agents: one serialisable observation object for
many perception models (open-vocabulary segmentation, detection, depth, VLM grounding,
multi-view geometry), consumable identically by a code policy, a VLA, or a toolchain.

`percepta` answers one question — *"where is the thing I asked about, and how sure am I?"* —
and answers it in metres when calibration allows, in pixels when it does not, but never by
guessing.

* **In-process SDK** for low latency, **loopback HTTP service** for multi-machine setups.
* **Prompts**: text (noun phrase), box, point. **Outputs**: instance masks (RLE/PNG), boxes,
  scores, 2D statistics, and calibrated 3D position/size/yaw with an uncertainty.
* **Pluggable tools**: SAM3 is only the default segmenter. CI runs on a pure-numpy `fake`
  segmenter — no GPU, no weights, no network.
* **Evidence**, not vibes: every Percept carries provenance (model hash, image hash,
  calibration id, latency, cache hit) and `verify()` compares before/after observations.

Spec: [`percepta-SPEC.md`](percepta-SPEC.md) (mirror at [`docs/spec.md`](docs/spec.md)).
Decisions taken where the spec was ambiguous: [`docs/decisions.md`](docs/decisions.md).

## Quickstart (no GPU, no weights)

```bash
python3 -m pip install -e .            # core: pydantic, numpy, pillow, pyyaml
python3 -m pip install -e ".[service]" # + fastapi, uvicorn
python3 -m pytest -q                   # full contract suite, GPU tests skipped

# serve with the synthetic segmenter
percepta serve --config configs/ci.yaml

# or call it
percepta perceive --image cam_head=scene.png --prompt text:red\ block --json
```

Python:

```python
import numpy as np
from percepta import Prompt, PerceptaClient

image = np.zeros((480, 640, 3), dtype=np.uint8)
image[200:280, 280:360] = (255, 0, 0)          # the thing to find

client = PerceptaClient("http://127.0.0.1:8790")
resp = client.perceive(
    images={"cam_head": image},
    prompts=[Prompt(type="text", text="red block")],
)
for p in resp.percepts:
    for o in p.observations:
        print(p.camera, o.score, o.area_px, o.box_xyxy, o.mask.encoding)
print(resp.fused.geometry.centroid if resp.fused else None)
```

`client.perceive_local(...)` runs the same pipeline in-process, without HTTP.

## Install extras

| extra | brings | for |
|---|---|---|
| *(none)* | `pydantic`, `numpy`, `pillow`, `pyyaml` | schema, masks, fake tool, SDK |
| `service` | `fastapi`, `uvicorn` | `percepta serve` |
| `sam3` | `torch`, `sam3` | the real segmenter (weights supplied at runtime) |
| `dev` | `pytest`, `httpx`, `ruff` | tests and lint |

`torch` is deliberately **not** a default dependency: the contract suite must run on a
GPU-less machine.

## HTTP API

| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | `{"ok":true,"uptime_s":…,"tools":[{"name","version","status"}]}` |
| GET | `/v1/info` | schema version, package version, tool capabilities, device, model hashes |
| GET | `/v1/tools` | tool list with capabilities |
| POST | `/v1/perceive` | `PerceiveRequest` → `PerceiveResponse` |
| GET | `/metrics` | optional Prometheus text |

Error mapping: `400` invalid request · `401` missing/wrong bearer · `413` body over
`max_body_bytes` · `503` queue full / tool unavailable · `504` timeout · `500` internal.
**`not_found` is a Percept status, not an HTTP error** — it arrives inside a `200`.

Binding is loopback-only by default (`127.0.0.1:8790`); put an SSH tunnel in front of it
rather than exposing it:

```bash
ssh -N -L 8790:127.0.0.1:8790 user@robot-host
```

Optional shared token: `export PERCEPTA_TOKEN=…` on the server, `PerceptaClient(token=…)`
on the client. No TLS, no auth system (out of scope).

## Configuration

See [`configs/example.yaml`](configs/example.yaml) (production, SAM3) and
[`configs/ci.yaml`](configs/ci.yaml) (synthetic, no weights). Every field of spec §11.1 is
supported; unknown keys are rejected so typos fail loudly.

```yaml
pipeline:
  default: ["sam3", "multi_view_3d"]
tools:
  sam3:
    checkpoint: /models/sam3.pt   # gated weights, never vendored
    resolution: 1008
```

## Tools

| tool | kind | needs | purpose |
|---|---|---|---|
| `fake` | fake | — | colour threshold + connected components; makes CI meaningful |
| `sam3` | segmenter | GPU + gated weights | text/box prompted open-vocabulary masks |
| `multi_view_3d` | geometry | calibration (optional depth) | multi-view fusion → 3D centroid/bbox/yaw |

A tool declares `capabilities()`; the pipeline validates prompts against those *before*
calling, so a box-only tool handed a text prompt yields a Percept with
`status="invalid_request"`, `message="unsupported_prompt"` — never a request-level failure.

Adding a tool: subclass `percepta.tools.base.BaseTool`, keep heavy imports inside `load()`,
register an entry point in the `percepta.tools` group (or `percepta.tools.<name>` module).

## Geometry and coordinates

Frozen, and the most common source of bugs:

* Pixels: origin top-left, `u` = column, `v` = row; `box = [x0, y0, x1, y1]` with `x1`/`y1`
  **exclusive**.
* Camera frame: OpenCV (x right, y down, z forward).
* World: right-handed metres; `T_world_cam` is **camera→world**, `p_world = T_world_cam @ p_cam_h`.
* **No calibration ⇒ no `Geometry`.** Metres are never inferred from pixels alone.
* Every geometry carries `frame="world"` and a non-null `uncertainty_m`; single-view results
  are explicitly marked (`uncertainty_m >= 0.03` at worst).

Fusion priority: depth (point cloud, percentile bbox, PCA yaw) → multi-view silhouette prism
(height search per `docs/decisions.md` D4) → single-view 2D + coarse height.

## Verification

```python
from percepta.verification import compare

result = compare(before, after, action={"ee_delta_px": [0.0, -18.0]})
assert result.checks["lifted"] and result.ok
```

`followed` / `lifted` / `stable` / `visible`; a `not_found` observation never counts as a pass.

## Repository layout

```
src/percepta/
  schema.py       frozen data model (spec §3)
  masks.py        RLE (appendix B), PNG, IoU, area/centroid
  geometry.py     project/unproject, planes, bboxes, PCA yaw
  fusion.py       multi-view 3D (§9)
  verification.py before/after evidence (§10)
  prompts.py      noun-phrase extraction, disambiguation (§8)
  registry.py     entry-point discovery, per-tool locks
  pipeline.py     orchestration shared by HTTP and SDK
  cache.py        LRU + TTL (§11.4)
  config.py       YAML config (§11.1)
  service/        FastAPI app, routes, logging, metrics
  client.py, cli.py
  tools/          base, fake, sam3, multi_view_3d
tests/            unit/, contract/, golden/, gpu/ (skipped by default)
docker/, systemd/, configs/, docs/
```

## Deploying

```bash
# container
docker build -f docker/Dockerfile -t percepta:0.1 .
docker run --rm -p 127.0.0.1:8790:8790 -v /models:/models:ro percepta:0.1

# systemd
sudo cp systemd/perceptad.service /etc/systemd/system/
sudo systemctl enable --now perceptad
```

Mount weights read-only; they are never built into the image.

## Testing

```bash
python3 -m pytest -q                 # everything except GPU (no GPU/network/weights needed)
python3 -m pytest -m gpu -q          # SAM3 smoke test, needs weights + CUDA
python3 -m ruff check .
```

Acceptance scenarios S1–S9 (spec §14) map onto `tests/unit`, `tests/contract` and
`tests/gpu`; see [`docs/decisions.md`](docs/decisions.md) for the choices made where the
spec left room.

## License

Apache-2.0. Model weights are **not** distributed here: SAM3 weights are gated
(`facebook/sam3`) and Ultralytics-based tools are AGPL-3.0, hence optional and off by default.

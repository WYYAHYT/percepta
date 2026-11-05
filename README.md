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
Release history: [`CHANGELOG.md`](CHANGELOG.md).

## Quickstart (no GPU, no weights)

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). `uv.lock` is committed and
pins Python 3.10 plus every transitive dependency, so the environment is reproducible.

```bash
uv sync                  # creates .venv from uv.lock (includes the dev group)
uv run pytest -q         # full contract suite, GPU tests skipped
uv run ruff check .

# serve with the synthetic segmenter
uv run percepta serve --config configs/ci.yaml

# or call it
uv run percepta perceive --image cam_head=scene.png --prompt text:red\ block --json
```

`uv sync --no-dev` gives a minimal runtime environment (no pytest/httpx/ruff).
Prefer pip? `pip install -e ".[service]"` still works — `pyproject.toml` stays
PEP 621 compliant — but `uv` is the supported path.

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

### Waiting for the model to be warm

The first request to a real model pays a one-time cost (SAM3: ~16–70 s depending on GPU load,
versus 137–530 ms afterwards). The service warms itself in the background at startup and
reports progress on `/healthz`, so a client should wait for it rather than send a request into
the cold path:

```python
client = PerceptaClient("http://127.0.0.1:8790")
if not client.wait_until_ready(timeout_s=180):   # waits for warm-up, not just the port
    raise SystemExit("service did not become ready")
print(client.warmup_state())        # {'status': 'ok', 'duration_ms': 15219.6}

resp = client.perceive(images={"cam_head": photo}, prompts=[Prompt(type="text", text="bottle")])
```

`wait_until_ready` returns ``True`` once warm-up has *settled*, including when it failed —
the service is up either way, and a broken model is reported separately in
`/healthz.tools[].status`. Pass ``warm=False`` to wait only for the port.

## Install extras

| extra | brings | for |
|---|---|---|
| *(none)* | `pydantic`, `numpy`, `pillow`, `pyyaml` | schema, masks, fake tool, SDK |
| `service` | `fastapi`, `uvicorn` | `percepta serve` |
| `sam3` | nothing — see `uv_envs/sam3/` | the real segmenter (weights supplied at runtime) |
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

## Model isolation (dependency conflicts)

Sooner or later two models demand incompatible dependencies — say `sam3` needs `torch>=2.7`
and a depth model is stuck on `torch==2.6`. One Python process can only hold **one** torch,
so the conflict cannot be configured away; the models must run in separate interpreters.

`percepta` solves this **without containers**, by making "where the model runs" a
configuration choice rather than code:

```yaml
tools:
  sam3:
    backend: subprocess              # inprocess | subprocess | http
    python: uv_envs/sam3/.venv/bin/python   # this model's own virtualenv
    cuda_visible_devices: "0"
  depth:
    backend: subprocess
    python: uv_envs/depth/.venv/bin/python  # a different torch, a different venv
    cuda_visible_devices: "1"
```

| backend | when | cost |
|---|---|---|
| `inprocess` | no conflict (default) | none — direct call |
| `subprocess` | two models disagree on versions | 0.01 s start, then a resident worker |
| `http` | separate host / independent restart | one loopback round trip |
| container | CUDA or system-library conflict | GB-scale images |

The subprocess backend sends frames over a binary pipe (raw `uint8` pixels, not JSON — a
640×480 frame is 921 KB raw vs ~15 MB as text) and loads the model once, so per-request
overhead is milliseconds. A dead or unavailable worker maps to Percept
`status="unavailable"`, never `not_found`.

Setting up an isolated model environment is the deployer's job, not the library's — you own
the interpreter, percepta only needs to be told where to find it. Each model ships its own
locked project under `uv_envs/`, so one command rebuilds it exactly:

```bash
uv sync --project uv_envs/sam3 --python 3.10    # -> uv_envs/sam3/.venv
```

`uv_envs/sam3/pyproject.toml` and its `uv.lock` are committed (that is what makes the runtime
reproducible); only the generated `.venv/` is ignored. The directory is named `uv_envs`, not
`envs`, because "envs" reads as simulator environments in robotics.

The worker needs only `numpy` + `pydantic` beyond the model's own framework — it imports
percepta's protocol and schema, nothing heavier. Each model gets its own GPU via
`cuda_visible_devices` so workers never fight over VRAM.

Full design, wire protocol and the checklist for adding a model:
[`docs/model-isolation.md`](docs/model-isolation.md).

### Adding a model — checklist

1. Does it conflict with an existing model's `torch`/`numpy`? No → `inprocess`; yes → `subprocess`.
2. Declare `backend` and `cuda_visible_devices` in the config.
3. Tool code depends only on the backend protocol — never `import torch` at module level.
4. Ship a fake adapter so the logic is testable without a GPU (see
   [`backends/reference.py`](src/percepta/backends/reference.py)).
5. Weights and their checksum come from config — never from the repository.
6. `/healthz` must report an uninstalled model as `error`, never `ok`.


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
  backends/       where a model runs: inprocess / subprocess / http
  client.py, cli.py
  tools/          base, fake, sam3, multi_view_3d
tests/            unit/, contract/, gpu/ (skipped by default)
systemd/, configs/, docs/
```

## Deploying

```bash
sudo cp systemd/perceptad.service /etc/systemd/system/
sudo systemctl enable --now perceptad
```

Weights are mounted read-only from the host (`checkpoint: /models/sam3.pt`); they are never
copied into the repository. Packaging this as a container image is left to the deployer —
`percepta` itself stays dependency-light, and a model's framework belongs to that model's own
environment (see [Model isolation](#model-isolation-dependency-conflicts)).

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

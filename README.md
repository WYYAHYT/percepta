# percepta

Documentation revised **2026-11-12**; previous wording and reasons are recorded in
[`docs/documentation-changes.md`](docs/documentation-changes.md).

A pluggable **observation layer** for robot agents: one serialisable observation object for
many perception models (open-vocabulary segmentation, detection, depth, VLM grounding,
multi-view geometry), consumable identically by a code policy, a VLA, or a toolchain.

`percepta` answers one question — *"where is the thing I asked about, and how sure am I?"* —
and reports metres when the calibration and tool support them, or pixels otherwise.
Coarse estimates state their assumptions and warnings; missing usable evidence does not
justify inventing geometry.

* **In-process SDK** for low latency, **loopback HTTP service** for multi-machine setups.
* **Prompts**: text (noun phrase), box, point. **Outputs**: instance masks (RLE/PNG), boxes,
  scores, 2D statistics, and calibrated 3D position/size/yaw with an uncertainty.
* **Pluggable tools**: SAM3 is only the default segmenter. CI runs on a pure-numpy `fake`
  segmenter — no GPU, no weights, no network.
* **Evidence**, not vibes: every Percept carries provenance (model hash, image hash,
  calibration id, latency, cache hit) and `percepta.verification.compare()` compares before/after observations.

Spec: [`percepta-SPEC.md`](percepta-SPEC.md) (mirror at [`docs/spec.md`](docs/spec.md)).
Decisions taken where the spec was ambiguous: [`docs/decisions.md`](docs/decisions.md).
Release history: [`CHANGELOG.md`](CHANGELOG.md).
Current deployment evidence and limits: [`docs/model-isolation.md`](docs/model-isolation.md).
The original M0–M5 are complete; the broader work in the
[`implementation plan`](docs/Percepta_RoboDojo_Agent_Implementation_Plan.md) remains in development.

## Quickstart (no GPU, no weights)

Dependencies are managed with [`uv`](https://docs.astral.sh/uv/). `.python-version` selects
Python 3.10, and the committed `uv.lock` pins package versions for a reproducible environment.

```bash
uv sync                  # creates .venv from uv.lock (includes the dev group)
uv run pytest -q         # default CPU suite; GPU tests skipped, optional OpenCV tests may skip
uv run ruff check .

# serve with the synthetic segmenter
uv run percepta serve --config configs/ci.yaml

# or call it
uv run percepta perceive --image cam_head=scene.png --prompt text:red\ block --json
```

`uv sync --no-dev` gives a minimal runtime environment (no pytest/httpx/ruff).
The `service` extra adds HTTP serving to a runtime installed with `--no-dev`.

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

The first request to a real model pays a one-time cost (SAM3's 2026-11-05 H20 measurements:
~16–70 s depending on GPU load, versus 137–530 ms afterwards). With warm-up enabled, the
service runs it in the background and reports progress on `/healthz`, so a client should
wait for it rather than send a request into the cold path:

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
The stereo/triview deployment configs disable this single-image warm-up. Use their paired or
three-image smoke commands below to exercise inference; a listening port alone does not prove it ran.

## Install extras

| extra | brings | for |
|---|---|---|
| *(none)* | `pydantic`, `numpy`, `pillow`, `pyyaml` | schema, masks, fake tool, SDK |
| `service` | `fastapi`, `uvicorn` | `percepta serve` |
| `stereo` | `opencv-python-headless` | stereo rectification and SGBM |

`dev` is a dependency group, installed by `uv sync` by default; it includes `pytest`,
`httpx`, `ruff`, `fastapi` and `uvicorn`. It is not an install extra.

`torch` is deliberately **not** a default dependency: the contract suite must run on a
GPU-less machine. SAM3 and neural depth models use separate locked projects under `uv_envs/`.

## HTTP API

| Method | Path | Notes |
|---|---|---|
| GET | `/healthz` | `{"ok":true,"uptime_s":…,"tools":[{"name","version","status"}]}` |
| GET | `/v1/info` | schema version, package version, tool capabilities, device, model hashes |
| GET | `/v1/tools` | tool list with capabilities |
| POST | `/v1/perceive` | `PerceiveRequest` → `PerceiveResponse` |
| GET | `/metrics` | optional Prometheus text |

Error mapping: `400` invalid request · `401` missing/wrong bearer · `413` body over
`max_body_bytes` · `503` queue full · `504` timeout · `500` internal.
An unavailable model returns HTTP `200` with Percept `status="unavailable"` and `tool_unavailable` in warnings.
**`not_found` is a Percept status, not an HTTP error** — it arrives inside a `200`.

Binding is loopback-only by default (`127.0.0.1:8790`); put an SSH tunnel in front of it
rather than exposing it:

```bash
ssh -N -L 8790:127.0.0.1:8790 user@robot-host
```

Optional shared token: `export PERCEPTA_TOKEN=…` on the server, `PerceptaClient(token=…)`
on the client. No TLS, no auth system (out of scope).

## Configuration

See [`configs/example.yaml`](configs/example.yaml) (SAM3 subprocess template; edit asset paths) and
[`configs/ci.yaml`](configs/ci.yaml) (synthetic, no weights). Every field of spec §11.1 is
supported; unknown section fields are rejected. Entries under `tools.<name>` are tool-specific
dictionaries, so validate their settings against the chosen tool's documentation.

```yaml
pipeline:
  default: ["sam3", "multi_view_3d"]
tools:
  sam3:
    backend: subprocess
    python: uv_envs/sam3/.venv/bin/python
    checkpoint: /models/sam3.pt   # gated weights, never vendored
    bpe_path: /models/bpe_simple_vocab_16e6.txt.gz
    resolution: 1008
```

Build the SAM3 runtime with `uv sync --project uv_envs/sam3 --locked`; there is no root `sam3`
extra. Each other deployed model has its own `configs/local-*.yaml` and locked runtime below.

## Tools

| tool | kind | needs | purpose |
|---|---|---|---|
| `fake` | fake | — | colour threshold + connected components; makes CI meaningful |
| `sam3` | segmenter | GPU + gated weights | text/box prompted open-vocabulary masks |
| `multi_view_3d` | geometry | calibration (optional depth) | multi-view fusion → 3D centroid/bbox/yaw |
| `stereo_3d` | geometry | exactly two calibrated RGB views; OpenCV | SGBM, FoundationStereo or Fast-FoundationStereo → visible-surface geometry |
| `triview_3d` | geometry | exactly three calibrated RGB views | pairwise SGBM, MVSAnywhere HERO or DA3-BASE → visible-surface geometry |

SAM3 supports text and box prompts; the schema's point prompt is supported by `fake`, not
the official SAM3 processor. The pairwise triview route also requires the `stereo` extra;
the neural triview routes keep their model dependencies in their own runtimes.

A tool declares `capabilities()`; the pipeline validates prompts against those *before*
calling, so a box-only tool handed a text prompt yields a Percept with
`status="invalid_request"`, `message="unsupported_prompt"` — never a request-level failure.

`options.tools` selects a serial pipeline. `multi_view_3d` is moved to its end; if several
geometry stages succeed, the last successful stage replaces the earlier `fused` result.
A later error leaves that previous success in place and remains visible in `percepts`.
Enabled stages run in resolved order: this is not an automatic fallback chain. `fuse=False` disables
the extra fusion used when no geometry tool ran. The pipeline still calls explicitly selected
geometry tools: `multi_view_3d` itself honors this flag, while `stereo_3d` and `triview_3d`
can still populate `fused`.

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
  stereo_3d:
    stereo_backend: foundation_stereo
    backend: subprocess
    python: uv_envs/foundation-stereo/.venv/bin/python
    cuda_visible_devices: "1"
```

This excerpt shows interpreter/GPU isolation only; the complete model configs below also
declare source, checkpoint and auxiliary assets. SAM3 supports all three backend forms;
the deployed neural stereo/triview adapters use resident subprocess workers.

| backend | when | cost |
|---|---|---|
| `inprocess` | no conflict (default) | none — direct call |
| `subprocess` | two models disagree on versions | process startup + model load once, then a resident worker |
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
uv sync --project uv_envs/sam3 --locked    # -> uv_envs/sam3/.venv
```

`uv_envs/sam3/pyproject.toml` and its `uv.lock` are committed (that is what makes the runtime
reproducible); only the generated `.venv/` is ignored. The directory is named `uv_envs`, not
`envs`, because "envs" reads as simulator environments in robotics.

The worker needs only `numpy` + `pydantic` beyond the model's own framework — it imports
percepta's protocol and schema, nothing heavier. Select GPUs explicitly with
`cuda_visible_devices` or the shell's `CUDA_VISIBLE_DEVICES`; a blank config value inherits
the shell environment. There is no automatic free-GPU selection or VRAM reservation.

Full design, wire protocol and the checklist for adding a model:
[`docs/model-isolation.md`](docs/model-isolation.md).

### FoundationStereo research runtime

FoundationStereo is **not** Fast-FoundationStereo: use `stereo_backend: foundation_stereo`.
Fast-FoundationStereo has its own `ffs` deployment below. The official 23-51-11 checkpoint is restricted to
**non-commercial research**. Source, checkpoint and auxiliary backbone hashes are pinned in
[`uv_envs/foundation-stereo/assets.json`](uv_envs/foundation-stereo/assets.json).

```bash
uv sync --extra stereo --locked
uv sync --project uv_envs/foundation-stereo --locked
uv_envs/foundation-stereo/.venv/bin/python scripts/bootstrap_foundation_stereo.py
uv run --extra stereo percepta models check --config configs/local-foundation-stereo.yaml
# Select an available GPU in the shell before running either command.
uv run --extra stereo python scripts/foundation_stereo_smoke.py --http
uv run --extra stereo percepta serve --config configs/local-foundation-stereo.yaml
```

The dedicated service listens on loopback port 8791; the default pipeline and SAM3 runtime
are unchanged. Send two RGB images with matching calibration and
`context={"stereo_left": "left", "stereo_right": "right"}`. Set
`options={"tools": ["stereo_3d"]}` when selecting this tool in a configured multi-tool service.
The API still requires a prompt, but this geometry-only tool does **not** segment that prompt:
without an original-left-image `stereo_mask`, the output summarizes the whole valid scene.
Its bbox and centroid describe visible points, not an inferred complete object.

See [`docs/model-isolation.md`](docs/model-isolation.md) §11 for verified timings,
offline initialization, failure semantics and runtime settings.

### Other deployed stereo and three-view models

Each model has a pinned source/asset manifest, an independent `pyproject.toml`/`uv.lock`,
an asset bootstrap and a real GPU smoke. Run commands from the repository root.

| Model | Runtime directory | Configuration | Selector | License |
|---|---|---|---|---|
| Fast-FoundationStereo 23-36-37 | `uv_envs/fast-foundation-stereo` | `configs/local-fast-foundation-stereo.yaml` | `stereo_backend: ffs` | non-commercial research |
| MVSAnywhere HERO | `uv_envs/mvsanywhere` | `configs/local-mvsanywhere.yaml` | `triview_method: mvs`, `triview_backend: mvsanywhere` | non-commercial research |
| DA3-BASE | `uv_envs/da3` | `configs/local-da3.yaml` | `triview_method: mvs`, `triview_backend: da3` | Apache-2.0 |

```bash
uv sync --project uv_envs/fast-foundation-stereo --locked
uv_envs/fast-foundation-stereo/.venv/bin/python scripts/bootstrap_fast_foundation_stereo.py
CUDA_VISIBLE_DEVICES=1 uv run --extra stereo python scripts/fast_foundation_stereo_smoke.py --http --upstream

uv sync --project uv_envs/mvsanywhere --locked
uv_envs/mvsanywhere/.venv/bin/python scripts/bootstrap_mvsanywhere.py
CUDA_VISIBLE_DEVICES=5 uv run python scripts/multiview_smoke.py --config configs/local-mvsanywhere.yaml --http

uv sync --project uv_envs/da3 --locked
uv_envs/da3/.venv/bin/python scripts/bootstrap_da3.py
CUDA_VISIBLE_DEVICES=4 uv run python scripts/multiview_smoke.py --config configs/local-da3.yaml --http

# Start the selected service; choose an available GPU for this machine.
CUDA_VISIBLE_DEVICES=4 uv run percepta serve --config configs/local-da3.yaml
```

Send exactly three RGB images and their pixel intrinsics / camera-to-world calibration to
`triview_3d`. DA3 requires non-collinear camera centres for metric pose alignment; MVSAnywhere
requires distinct centres. The outputs describe whole-scene visible surfaces; text prompts
do not segment targets. Confidence is not a calibrated positioning error.

DA3 rejects extreme image aspect ratios that would leave less than one processed patch;
these requests return `invalid_request` and preserve the loaded worker. Camera poses are
rebased in float64 before model input, while output world geometry uses the original calibration.
FFS and MVSAnywhere manifests are packaged in the wheel. A crashed worker is reported as
`error` in tool health, including when it exits between requests.

The deployed Fast-FoundationStereo checkpoint supports `max_disp <= 416`; 640 was tested and
fails inside its fixed positional encoding. Full reproducibility and validation notes:
[`docs/model-isolation.md`](docs/model-isolation.md) §12.

### Adding a model — checklist

1. Does it conflict with an existing model's `torch`/`numpy`? No → `inprocess`; yes → `subprocess`.
2. Declare `backend` and `cuda_visible_devices` in the config.
3. Tool code depends only on the backend protocol — never `import torch` at module level.
4. Ship a fake adapter so the logic is testable without a GPU (see
   [`backends/reference.py`](src/percepta/backends/reference.py)).
5. Checkpoint paths and hashes belong in deployment config; pinned asset manifests may also
   be versioned in the repository. Weight files themselves must never be tracked.
6. `/healthz` reports enabled tools whose attempted load failed, or whose resident worker
   failed, as `error`. Disabled tools are omitted; a lazy, unattempted tool is not proof of a loaded model.


## Geometry and coordinates

Frozen, and the most common source of bugs:

* Pixels: origin top-left, `u` = column, `v` = row; `box = [x0, y0, x1, y1]` with `x1`/`y1`
  **exclusive**.
* Camera frame: OpenCV (x right, y down, z forward).
* World: right-handed metres; `T_world_cam` is **camera→world**, `p_world = T_world_cam @ p_cam_h`.
* **No calibration ⇒ no `Geometry`.** Metres are never inferred from pixels alone.
* Every geometry carries `frame="world"` and a non-null `uncertainty_m`. The current
  single-view table estimate has a height-uncertainty floor of 0.05 m; this is an assumption,
  not a guaranteed error bound.

Fusion priority: depth (point cloud, percentile bbox, PCA yaw) → multi-view silhouette prism
(D35: sampled vertical columns and the peak of the complete height/mean-IoU curve) → coarse
single-view 3D using an explicitly known table plane and assumed object height. Without that
table information, the single-view no-depth path returns no geometry.

The 2026-11-12 coordinate fixes (D39) map hang polygons from world XY into grid pixels before
rasterization and compute lateral uncertainty using the full camera-centre vector. The repair
history for `DOC-20261112-01` and `DOC-20261112-02` is in the
[`modification record`](docs/documentation-changes.md). Hang remains an experimental union of
per-view candidates accepted only when its IoU improves; uncertainty values remain estimates,
not strict error bounds.

## Verification

```python
from percepta.verification import compare

result = compare(before, after, action={"ee_delta_px": [0.0, -18.0]})
print(result.ok, result.checks)   # inspect the evidence; pixel displacement alone does not establish a lift
```

`followed` / `lifted` / `stable` / `visible`; a `not_found` observation never counts as a pass.

## Repository layout

```
src/percepta/
  schema.py       frozen data model (spec §3)
  masks.py        RLE (appendix B), PNG, IoU, area/centroid
  geometry.py     project/unproject, planes, bboxes, PCA yaw
  hull.py         silhouette column test for a 3D footprint
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
  tools/          base, fake, sam3, multi_view_3d, stereo_3d, triview_3d
tests/            unit/, contract/, gpu/ (skipped by default)
uv_envs/          sam3/, foundation-stereo/, fast-foundation-stereo/, mvsanywhere/, da3/
scripts/          bootstrap and smoke tools, lock portability check, geometry analysis
systemd/, configs/, docs/
```

Each model runtime contains `.python-version`, `pyproject.toml` and `uv.lock`; the four
source-based depth runtimes also contain `assets.json`. Generated `.venv/`, upstream
checkouts under `models/`, and weights under `checkpoints/` are ignored by Git.

## Deploying

```bash
sudo cp systemd/perceptad.service /etc/systemd/system/
sudo systemctl enable --now perceptad
```

Edit the unit's user, working directory and `/etc/percepta/config.yaml` for the host before
enabling it. Give that user read access to the model runtime, weights and auxiliary assets.
Weights may live at an external read-only path (`checkpoint: /models/sam3.pt`) or in an
ignored local `checkpoints/` directory; they are never tracked or included in the package.
Packaging this as a container image is left to the deployer —
`percepta` itself stays dependency-light, and a model's framework belongs to that model's own
environment (see [Model isolation](#model-isolation-dependency-conflicts)).

## Testing

```bash
uv run pytest -q                     # default CPU suite, no GPU/network/weights
uv run --extra stereo pytest -m "not gpu" -q  # include optional OpenCV geometry
uv run ruff check .
uv run python scripts/e2e_smoke.py    # real loopback HTTP agrees with in-process
uv run python scripts/check_lock_portability.py --list
```

Real GPU tests require each selected model's environment and local assets. Run them
explicitly, choosing an available GPU before each model; these commands do not select a free
GPU automatically. SAM3 additionally needs a real photograph containing flowers for its test.

```bash
export CUDA_VISIBLE_DEVICES=0   # example only: choose a GPU available on your host
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

PERCEPTA_SAM3_GPU="$CUDA_VISIBLE_DEVICES" PERCEPTA_SAM3_PHOTO=/path/to/flower-photo.jpg \
  uv run pytest -m gpu tests/gpu/test_sam3.py -q
uv run --extra stereo pytest -m gpu tests/gpu/test_foundation_stereo.py -q
uv run --extra stereo pytest -m gpu tests/gpu/test_fast_foundation_stereo.py -q
uv run pytest -m gpu tests/gpu/test_mvsanywhere.py tests/gpu/test_multiview_tools.py -k mvsanywhere -q
uv run pytest -m gpu tests/gpu/test_da3.py tests/gpu/test_multiview_tools.py -k da3 -q
```

Acceptance scenarios S1–S9 (spec §14) map onto `tests/unit`, `tests/contract` and
`tests/gpu`; see [`docs/decisions.md`](docs/decisions.md) for the choices made where the
spec left room. Later stereo/triview tests extend those original scenarios. Historical H20
measurements are dated in [`docs/model-isolation.md`](docs/model-isolation.md) §§10–12;
they establish deployment behavior for those runs, not guarantees for every input or host.

## License

The percepta library is Apache-2.0. Model weights are **not** distributed here: SAM3 weights
are gated (`facebook/sam3`); FoundationStereo, Fast-FoundationStereo and MVSAnywhere HERO
weights are limited to non-commercial research. DA3-BASE source and weights are Apache-2.0.
See each runtime's asset manifest and upstream license before redistributing model assets.

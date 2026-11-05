# AGENTS.md — collaboration rules for `percepta`

Derived from `percepta-SPEC.md` §0 / §16. **Read the spec first**; this file only adds the
things a parallel implementation needs: file ownership, frozen seams, and the decisions we
took where the spec is ambiguous (spec §19).

## 1. Current state

M0–M5 are **done and green** under `uv`.

```bash
uv sync                              # creates .venv (Python 3.10) from uv.lock
uv run pytest -q                     # 277 passing, 2 skipped (gpu), no GPU/network/weights
uv run ruff check .                  # clean
uv run python scripts/e2e_smoke.py   # real uvicorn; HTTP and in-process agree
```

**Dependencies are managed by `uv`.** `uv.lock` is committed and generated against the
official PyPI index, so it resolves anywhere; `.python-version` pins Python 3.10.
`torch` / `sam3` live behind the `sam3` extra and are **never** part of a default sync
(spec §13). `uv sync` installs the `dev` group by default, so a fresh clone can run the
test suite immediately; `uv sync --no-dev` gives a minimal runtime environment.

```bash
uv sync --extra service    # + fastapi/uvicorn (also in dev, so usually unnecessary)
uv sync --extra sam3       # + torch>=2.7, sam3==0.1.4  (multi-GB, GPU machine only)
```

Local network note for *this* host: PyPI official takes ~23 s per request through the
`127.0.0.1:8118` proxy while aliyun takes 0.15 s. For interactive installs set
`UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple` in the shell, but **do not
regenerate `uv.lock` under a mirror** — it bakes mirror URLs into every download entry and
breaks machines that cannot reach that mirror. If you must re-lock, do it against
`https://pypi.org/simple` (through the proxy).

If you need another package, stop and ask the Lead: spec §16.3 forbids new default
dependencies. Add it to the right extra, then `uv lock`.

## 2. File ownership (hard boundaries, one owner per file)

| Owner | Files |
|---|---|
| **Lead** | `pyproject.toml`, `src/percepta/{__init__,schema,masks,registry,config,prompts}.py`, `src/percepta/tools/{__init__,base,fake}.py`, `src/percepta/tools/sam3.py`, `tests/{conftest,synth}.py`, `AGENTS.md`, `README.md`, `docs/**`, `configs/**` |
| **service** | `src/percepta/service/*`, `pipeline.py`, `client.py`, `cli.py`, `tests/contract/**`, `tests/unit/test_client.py`, `tests/unit/test_pipeline.py` |
| **geometry** | `src/percepta/{geometry,fusion}.py`, `src/percepta/tools/multi_view_3d.py`, `tests/unit/test_{geometry,fusion}.py`, `tests/synth_geom.py` |
| **state** | `src/percepta/{cache,verification}.py`, `tests/unit/test_{cache,verification}.py`, `systemd/**` |

Never edit a file you do not own — send the owner a message instead. `tests/synth.py` is
shared and **read-only** for everyone but the Lead (add geometry fixtures to
`tests/synth_geom.py`).

## 3. Frozen seams (do not change without the Lead)

* `schema.py` — every model, field name and enum in spec §3. Additions are optional fields
  only, and must be recorded in `docs/decisions.md`.
* Coordinates (spec §4): pixel `u`=column/`v`=row, origin top-left, box `[x0,y0,x1,y1]`
  with `x1`/`y1` **exclusive**; camera frame OpenCV; `T_world_cam` is **camera→world**
  (`p_world = T_world_cam @ p_cam_h`).
* `masks.py` — RLE is the appendix-B format (`sum(counts) == H*W`, starts with the 0-run).
* `tools/base.py` — `PerceptionTool` ABC (`name`/`version`/`api_version`,
  `capabilities()`/`load()`/`perceive()`/`close()`) and the helpers `make_observation`,
  `make_percept`, `build_provenance`, `apply_top_k`.
* `registry.py` — `ToolRegistry.instance(name, config)` returns a loaded singleton and
  raises `UnknownTool` / `ToolLoadError`; `lock_for(name)` gives the per-tool inference lock.

## 4. Module seams that other agents code against

These are contracts; implement exactly, even if your module is the only user today.

```python
# src/percepta/pipeline.py            (owner: service)
class PipelineRunner:
    def __init__(self, registry: ToolRegistry, config: AppConfig, cache: Any = None) -> None: ...
    def perceive(self, request: PerceiveRequest) -> PerceiveResponse: ...
    def close(self) -> None: ...

# src/percepta/cache.py               (owner: state)
class PerceptCache:
    def __init__(self, max_entries: int = 256, ttl_s: float = 300.0) -> None: ...
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, value: Any) -> None: ...
    @property
    def stats(self) -> dict[str, float]: ...
    @staticmethod
    def make_key(image_sha256: dict[str, str], prompt: Prompt, tool: str,
                 options: PerceiveOptions | None = None) -> str: ...
def build_cache(config: CacheConfig) -> PerceptCache: ...   # NullCache when disabled

# src/percepta/verification.py        (owner: state)
def compare(before: Percept, after: Percept, action: dict, options: dict | None = None) -> VerificationResult: ...

# src/percepta/geometry.py            (owner: geometry)
def project_points(points_cam: np.ndarray, K) -> np.ndarray: ...          # (N,3)->(N,2)
def unproject_pixels(uv, depth, K) -> np.ndarray: ...                     # (N,2)+(N,)->(N,3) camera
def mask_to_world_points(mask, depth, K, T_world_cam) -> np.ndarray: ...  # (M,3) world
def bbox3d_from_points(points) -> tuple[list[float], list[float]]: ...
def yaw_from_points(points_xy: np.ndarray) -> float: ...

# src/percepta/fusion.py              (owner: geometry)
def fuse(percepts: list[Percept], calibration: dict[str, Calibration], context: dict, options: PerceiveOptions | None = None) -> Percept: ...
```

`ToolContext.extras["upstream"]` is how the pipeline hands earlier percepts to later
tools (`list[Percept]`), per spec §5.3.

## 5. Error semantics (spec §16.4, appendix A)

* Missing model/weights/device → `ToolUnavailable` → Percept `status="unavailable"`.
* Tool genuinely looked and found nothing → Percept `status="not_found"`, HTTP **200**.
* Bad request payload → `invalid_request` → HTTP **400**.
* Unsupported prompt type → Percept `status="invalid_request"` with
  `message="unsupported_prompt"` and `extras["code"]="unsupported_prompt"` (see
  `docs/decisions.md` D1). Never swallow an exception into `not_found`.

## 6. Definition of done for any change

1. `python3 -m pytest -q` still green (GPU tests carry `@pytest.mark.gpu`).
2. `python3 -m ruff check .` clean.
3. New behaviour has a test that would fail without it.
4. Coordinates: docstring states input/output frames and the formula.
5. No new default dependency; no `import torch` at module import time.

The Lead additionally runs `python3 scripts/e2e_smoke.py`, which starts a real uvicorn server
on an ephemeral port, drives it through `PerceptaClient`, and asserts the in-process and HTTP
paths agree pixel-for-pixel (Lead-owned; not part of pytest).

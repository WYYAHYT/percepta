# Changelog

All notable changes to `percepta`. Versions follow [SemVer](https://semver.org/); the
`schema_version` of the wire format (`percepta/v1`) is versioned separately and has not
changed. Every design decision referenced as `D<n>` lives in [`docs/decisions.md`](decisions.md).

---

## 0.2.1 — 2026-11-05

Engineering hardening after 0.2.0. No API changes; the wire format is untouched.

### Changed

* **Model runtimes are separate locked projects under `uv_envs/<model>/`** (D33). The root
  `pyproject.toml`'s `sam3` extra is **gone** — it was broken by construction:

  1. resolving `torch` from PyPI yields the **CPU build** (2.14.1, no `+cu128`), so
     `uv sync --extra sam3` produced an environment with no GPU support;
  2. sam3 0.1.4 imports `psutil` at module scope without declaring it, so `import sam3` failed;
  3. it silently disagreed with the environment that actually runs the model
     (torch 2.10.0+cu128) — two different "SAM3 environments" for one repository.

  Each runtime now rebuilds exactly with one command:

  ```bash
  uv sync --project uv_envs/sam3 --python 3.10    # -> uv_envs/sam3/.venv
  ```

  `pyproject.toml` + `uv.lock` are committed (that is what makes the runtime reproducible);
  only the generated `.venv/` is ignored. Named `uv_envs`, not `envs`, because "envs" reads as
  simulator environments in a robotics codebase.

  Side effect: the root lockfile dropped from 93 to **33 packages**, since the extra's CUDA
  dependencies are no longer resolved into the library environment.

* **`configs/local-sam3.yaml` no longer pins a GPU index.** This host has 8 shared H20s and no
  permanently free card, so a hardcoded index made warm-up time depend on whatever else landed
  there (15 s quiet versus 45–73 s shared). Fixing that exposed a real bug: an empty
  `cuda_visible_devices` was exported verbatim, and `CUDA_VISIBLE_DEVICES=""` means *hide every
  GPU* (D32). Empty and whitespace-only now mean "inherit".

### Added

* **`.github/workflows/ci.yml`** — the repository's first CI: lockfile portability check, both
  `uv lock --check` runs, `uv sync --locked`, ruff, pytest and the end-to-end smoke test.
* **`scripts/check_lock_portability.py`** (D34) — whitelist check on every committed `uv.lock`,
  with 17 tests of its own. The rule "don't lock against a mirror" had been in `AGENTS.md` and
  was violated **twice**; the root lockfile reached this release carrying **631 aliyun URLs**.
  A rule broken twice is a rule that should not rely on memory.
  `uv lock --check` alone cannot catch this: it validates that the lock matches
  `pyproject.toml`, not that the URLs inside are reachable from anywhere.

### Fixed

* Root `uv.lock` regenerated against the official index (631 mirror URLs → 0).

---

## 0.2.0 — 2026-11-05

First version that runs a **real** perception model, end to end, over both entry points.

### Verified, not assumed

Real SAM3 (3.2 GiB checkpoint, `sha256 9999e234…`, NVIDIA H20) through the subprocess backend:

| | |
|---|---|
| real photo, `prompt="flower"` | 4 instances, top score **0.973**, area 58,717 px, masks at original resolution |
| first request, cold | 16.3 s → **137–343 ms** after warm-up |
| warm inference | 253–530 ms; 127 ms for a second prompt on the same frame |
| warm-up itself | 15 s idle GPU, 45–73 s when the GPU is shared |

Acceptance scenarios S1–S9 from the spec all pass; S9 (SAM3 smoke) is now automated under
`pytest -m gpu` and requires a real photograph.

### Added

* **Model isolation without containers** (`src/percepta/backends/`, `docs/model-isolation.md`).
  A model's dependencies no longer constrain the library: `tools.<name>.backend` selects
  `inprocess`, `subprocess` (its own virtualenv, binary pipe) or `http`. Two models may now
  require incompatible `torch`/`numpy` versions.
* **Model asset contract** (`percepta/weights.py`). `checkpoint`, `checkpoint_sha256`,
  `verify_weights` (`if-declared`/`always`/`never`), plus `percepta models check` as a
  pre-flight. A wrong or truncated checkpoint is refused instead of loading and quietly
  producing wrong segments (D27).
* **Startup warm-up** (`percepta/warmup.py`). One throwaway request in a background thread;
  state reported on `/healthz` as `warmup: {status, duration_ms, error?}` (D29).
* **`PerceptaClient.wait_until_ready`** now waits for warm-up, not just for the port, and
  `warmup_state()` exposes the raw state (D30).
* **`configs/local-sam3.yaml`**: a config that actually runs real SAM3 on this host.
* **uv** dependency management: `uv.lock` resolved against the official index,
  `.python-version` pinning 3.10, dev tooling in a PEP 735 group (D22).

### Changed

* Dependency constraints corrected: `torch>=2.1` → `>=2.7,<3` (the spec's version was simply
  wrong for sam3 0.1.4), `sam3` → `==0.1.4`, `numpy` gained sam3's `<2.4` ceiling (D23).
* `percepta/__init__` imports only `schema` eagerly; everything else is lazy, so a model worker
  in a slim virtualenv can import the protocol without pyyaml/pillow (D26).
* `ToolEntry.describe()` reports `version`/`kind` from the tool *class*, so `/healthz`,
  `/v1/tools` and the CLI agree for tools that are registered but not loaded. This removed a
  duplicated helper in `service/routes.py` that had been reporting `0.0.0` (D31).
* Failed tool states remain HTTP 200 with Percept `status="unavailable"` (D17, a deliberate
  deviation from appendix A's 503 column — still open for review).

### Removed

* `requirements/` and `docker/`: a library should not maintain a manifest of what a *model*
  needs. Model frameworks are installed by the deployer; the library only constrains assets
  (D27).
* `tools.sam3.warmup`, which warmed a 32×32 all-zero frame — wrong kernel shapes, a second
  full checkpoint read, and invisible to operators.
* `tests/golden/` and `examples/` were dropped from the spec's layout rather than shipped
  half-done.

### Fixed (found only by running the real model)

These four are upstream traps; all are documented in `docs/model-isolation.md` §10.

1. `Sam3Processor.set_image` accepts HWC numpy **without error** but treats it as CHW, turning
   `(480, 640, 3)` into C=480/H=640/W=3 and returning **zero instances for every prompt**.
   Passing a PIL image fixes it. A mocked backend can never catch this.
2. `build_sam3_image_model`'s first positional argument is `bpe_path`, not `device`;
   `Sam3Processor` requires `model`; `add_geometric_prompt` requires `label`.
3. bf16 `boxes`/`scores` cannot go straight to numpy — upcast first.
4. The PyPI wheel ships no BPE vocabulary; the official file lives under `sam3/assets/` in the
   GitHub repo (not the repo-root `assets/`).

Also fixed here: the parent process now records the checkpoint hash itself (a worker keeps the
model elsewhere, so provenance was blank); `SubprocessBackend.segment()` raises `unavailable`
when a worker has died instead of silently respawning it and blaming the caller; and an empty
`cuda_visible_devices` means "inherit", not "hide every GPU" (D32).

### Known limitations

* One worker per model, serialised by a lock. Concurrent requests queue (`max_queue`); real
  parallelism needs several workers on different GPUs, which is documented but not automated.
* Prism-based multi-view geometry assumes a convex object on a table plane; `bbox3d` is an
  upper bound when there is no depth.
* SAM3 point prompts are unsupported (the upstream processor has no such API).
* `not_found` on a *synthetic* image is normal: SAM3 is trained on photographs.

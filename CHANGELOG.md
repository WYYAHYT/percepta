# Changelog

All notable changes to `percepta`. Versions follow [SemVer](https://semver.org/); the
`schema_version` of the wire format (`percepta/v1`) is versioned separately and has not
changed. Every design decision referenced as `D<n>` lives in [`docs/decisions.md`](docs/decisions.md).

---

## Unreleased

### Audit fixes — 2026-11-12

* Isolate SAM3 image state per request and text/box state per prompt; enforce worker startup
  and full RPC deadlines, strict frame validation and terminal cleanup (D40).
* Keep HTTP admission and inflight accounting until real inference finishes, serialize
  singleton loading, close failed instances, and preserve failed geometry-stage classification.
* Validate physical calibration, geometry numeric options and SGBM image dimensions; require
  world geometry for lift verification and report triview participation consistently (D41).
* Isolate cached response objects, restore duplicate prompts linearly, bound SDK readiness IO,
  close replaced local runners, return CLI failure codes and report model health/hash correctly (D42).
* Apply Options filtering as defaults while preserving explicitly provided Prompt values,
  including explicit `0.2` and `1`, as confirmed by the user (D43).
* Preserve the `a761656` audit baseline, original behavior and red/green regression evidence
  in [`docs/documentation-changes.md`](docs/documentation-changes.md). Package/schema/protocol
  versions and default dependencies are unchanged.

### Documentation — 2026-11-12

* Align the specification, README, deployment notes, configuration guidance and source
  documentation with implementation checkpoint `534066a`. Record the previous wording,
  replacement and reason in [`docs/documentation-changes.md`](docs/documentation-changes.md).
  Keep historical decisions and withdrawn plan annotations with dated corrections.
* Clarify geometry-stage ordering, per-tool `fuse=False` behavior, independent model
  environments, current SAM3 API and the D35 geometry path. Document two newly confirmed
  silhouette-geometry coordinate issues as pending fixes; this documentation change does
  not alter their calculations or claim they are resolved.

### Added

* Isolated **FoundationStereo 23-51-11** runtime and the explicit
  `stereo_backend="foundation_stereo"` route in `stereo_3d` (D36). Pinned source, official
  research checkpoint/config and auxiliary backbone hashes; repeatable asset bootstrap and
  offline loading. Research-only license.
* Deploy **Fast-FoundationStereo 23-36-37**, **MVSAnywhere HERO**, and **DA3-BASE** in separate
  locked uv projects, with hash-pinned official assets, offline bootstrap, resident workers,
  GPU smoke tests and local service configurations (D37).
* `stereo_backend="ffs"` now selects Fast-FoundationStereo; `triview_method="mvs"` selects
  deployed MVSAnywhere or DA3 metric depth. Three-view RPC reuses the existing bounded worker
  lifecycle; model paths and factories remain trusted deployment configuration.
* CI checks every model lockfile and runs an additional CPU suite with the stereo extra.
* Dedicated two-RGB/float-disparity worker protocol with bounded IO deadlines, resident reuse,
  worker cleanup and explicit unavailable errors. SAM3's protocol and root dependencies are unchanged.
* `configs/local-foundation-stereo.yaml` and real GPU / loopback HTTP smoke coverage, including
  comparison with the upstream demo, empty geometry and missing/crashed worker behavior.

### Fixed

* Correct silhouette hang rasterization by converting world XY polygons to plane-grid pixel
  coordinates before filling them. Candidate selection retains its existing IoU improvement gate.
* Compute silhouette lateral uncertainty using the full camera centre rather than its x
  component, preserving the estimate when the world origin changes. Both fixes (D39,
  2026-11-12) close the issues recorded in the earlier documentation audit, with coordinate
  translation regressions.
* DA3 rejects image sizes that would leave fewer than one processed patch before sending
  data to its worker; a rejected request leaves the resident model usable.
* DA3 rebases camera poses in float64 before upstream float32 conversion, preserving metric
  depth when the caller changes the world origin, including million-metre coordinates.
* FFS and MVSAnywhere runtime manifests are included as wheel resources, so model loading
  also works outside an editable source checkout.
* Resident worker failures and idle process exits update tool health to `error` without
  waiting for the inference lock or restarting the failed worker.
* Stereo now actually remaps both RGB images and the original-left mask, uses the correct
  camera-relative translation and OpenCV +Z-forward unprojection. Triview's pairwise route shares
  these corrections. Explicit left/right ordering is honoured; reversed/vertical pairs and
  out-of-support correspondences cannot silently become valid geometry.
* Stereo/triview outputs explicitly describe visible-surface statistics and assumed disparity
  uncertainty, not a complete object reconstruction or measured positioning accuracy.
* Reject duplicate camera centres and degenerate DA3 pose alignment before model inference;
  invalid requests cannot poison an otherwise healthy multiview worker.

Measured on one H20 with the official 960×540 pair: 0.828–0.831 s warm tool calls,
643–644 ms GPU forward, 3.70 GiB peak allocated memory; see `docs/model-isolation.md` §11.

---

## 0.3.0 — 2026-11-09

The no-depth geometry path was measuring the wrong thing. Accuracy improved by a factor of
three in the worst rig and two silent failures were removed.

### Fixed

* **The prism footprint was the object's *shadow*, not its footprint** (D35). The cross-section
  was the cone slice at mid-height — where each camera's silhouette rays cross that plane — which
  is larger than the object by roughly ``h/2·tan(θ)``. Measured on a 10 cm cube with a known
  ground truth: **178–293 cm² reported against a true 100 cm² (1.8–2.9×)**.

  It is now the discrete silhouette-restricted visual hull: a plane cell belongs to the object iff
  its **whole vertical column** lies inside every mask (`src/percepta/hull.py`). Computed on a
  ground-plane grid rather than as a convex polygon, because a three-view hull is generally
  non-convex and a polygon would discard exactly the information that matters.

* **The height was systematically underestimated.** Spec D4 asked for the *smallest* height whose
  IoU reached the threshold, which stops far too early: the curve crosses 0.5 at h≈0.04 while
  peaking at 0.10 for a 10 cm cube. The answer is now the **peak**; the threshold only gates
  confidence. This also removes the half-height error it injected into ``top_z`` and the
  centroid's z.

* **A 15 cm object measured 2 cm².** The column test sampled its ends (fractions 0.0 and 1.0), and
  a mask covers the object's *interior* — the base and top edges are exactly where the interior
  stops. Every correct column therefore read "outside" and the answer collapsed. Sampling is now
  at interior fractions.

* **A coarse `prism_step_m` collapsed the search to one sample**, which cannot have a peak.
  ``MIN_PRISM_STEPS`` floors it.

### Added

* **Split uncertainty** (D35). ``Geometry`` gains ``lateral_uncertainty_m``,
  ``height_uncertainty_m``, ``view_disagreement_m`` and ``method`` — all optional, so nothing
  existing changes shape. ``uncertainty_m`` keeps its frozen meaning (the worst axis), so no
  existing caller silently becomes more confident.

  The distinction matters because the two numbers differ by an order of magnitude: one mask pixel
  moves the footprint a few millimetres while leaving the height *guessed*. A caller that descends
  onto a surface must read ``height_uncertainty_m``; a caller that only needs a lateral target can
  read ``lateral_uncertainty_m`` and ignore the rest. ``view_disagreement_m`` is the third,
  independent signal: the average of a good view and an occluded one looks perfectly confident, and
  only the per-view spread reveals it.

* `scripts/footprint_accuracy.py` — the measurement harness these numbers come from, kept so the
  next change can be compared against the same ground truth.
* 18 tests in `tests/unit/test_footprint.py`, including one per defect above.

### Verified

| rig | centroid error | height error |
|---|---|---|
| 0/120/240 | 1.03–1.52 cm → **0.33–1.04 cm** | 0–3 cm → **0–2 cm** |
| 0/180 | 0.5–2.0 cm → **0.5–1.0 cm** | 1–4 cm → **1–2 cm** |
| 0/90/180/270 | 0–1.5 cm → **0.01–1.5 cm** | 0–3 cm → **0–3 cm** |

401 tests pass (18 new), 1 skipped; ruff clean; both lockfiles portable.

---

## 0.2.2 — 2026-11-05

### Fixed

* **Appendix A contradicted itself on `tool_unavailable`** (D17, now resolved). The row said
  "503 (Percept.status=`unavailable`)" — but a 503 response body is not a `PerceiveResponse`,
  and a `Percept` only exists inside one. The table was corrected to **200**, so the spec and
  the implementation finally agree instead of the code permanently being "a deviation awaiting
  review".

  `POST /v1/perceive` returns **200** with the affected Percepts carrying
  `status="unavailable"`, `extras.code="tool_unavailable"`, a human-readable `message`, and a
  top-level `warnings` entry. `/healthz` still reports the tool as `error`.

  The deciding argument: this endpoint is a **batch** operation (one request may span several
  cameras × prompts × tools). Answering 503 because one tool is down discards both the results
  the other tools produced and the information about which tool failed.

### Added

* **Five contract tests for the unavailable path** — it previously had **none**, which is how
  the disagreement survived so long. They cover: HTTP 200 rather than 503, the Percept status
  and its `unavailable` ≠ `not_found` distinction, the top-level warning, `/healthz` reporting
  `error`, and — most importantly — a healthy and a broken tool in the same pipeline where the
  healthy tool's observations must survive.

  Verified by mutation: forcing the endpoint to answer 503 makes three of them fail
  immediately, so they genuinely pin the decision rather than passing vacuously.

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

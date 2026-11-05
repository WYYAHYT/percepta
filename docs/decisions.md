# Decisions taken where the spec was ambiguous (spec §19)

Spec §0.5 says: "有歧义就在 §19 记录问题并选一个保守实现，不要阻塞。" These are those
decisions. **Any agent may read this file; only the Lead edits it.** Each entry names the
spec clause it resolves.

## D1 — `unsupported_prompt` is not a `Status` value

*Conflict:* §3.1 freezes `Status = ok|not_found|occluded|unavailable|timeout|invalid_request`,
but appendix A lists `unsupported_prompt` with "400（Percept.status）", and S8 accepts
`status=invalid_request/unsupported_prompt`.

*Decision:* keep the frozen enum. An unsupported prompt yields a Percept with
`status="invalid_request"`, `message="unsupported_prompt"`, and
`extras["code"]="unsupported_prompt"`, inside an HTTP **200** response for the rest of the
request. Rationale: one unsupported prompt must not fail a request whose other prompts
succeeded, and the enum is frozen while `extras` is free-form. `message` is stable, so
tests can assert on it.

## D2 — `ToolContext.extras`

*Conflict:* §5.3 passes upstream percepts via `ctx.extras["upstream"]`, but the §5.1
`ToolContext` definition has no `extras` field.

*Decision:* add `extras: dict[str, Any] = {}` to `ToolContext` as an **optional additive**
field (§3 allows additions that are optional) and pass upstream percepts as
`ctx.extras["upstream"]`, matching §5.3 literally.

## D3 — `ToolContext` as a pydantic model with ndarrays

*Conflict:* §5.1 declares `ToolContext(BaseModel)` but it holds `np.ndarray`s, a logger and
a cache handle.

*Decision:* keep it a `BaseModel` with `arbitrary_types_allowed=True`; `logger` and `cache`
are typed `Any`. This keeps the frozen shape while allowing duck-typed collaborators.

## D4 — Silhouette/prism height search (§9, algorithm 2) is underspecified

*Conflict:* §9 says "取匹配最好且高度最小的解" without stating the discretisation, the
match metric's granularity, or what to do when nothing matches.

*Decision:* scan heights from `table_z` to `table_z + prism_max_height_m` (default 0.5 m) in
steps of `prism_step_m` (default 0.01 m). For each candidate height, build the prism whose
cross-section is the back-projected mask outline at that height, re-project it into every
other calibrated view, and score it by mean mask IoU. Pick the **smallest** height whose
best score is ≥ `prism_iou_threshold` (default 0.5); if no height reaches the threshold,
pick the global arg-max and add the warning `geometry_low_confidence`.
`uncertainty_m` = `prism_step_m * (1 + 1/views_used)` plus a margin proportional to the
IoU shortfall. This makes S5's "centroid error < 2 cm" reproducible.

## D5 — `object_id` is only stable within a percept

*Conflict:* §3.4 calls `sha1(percept_id + index)` a "稳定 id", but `percept_id` is a
`uuid4`, so ids differ across requests.

*Decision:* implement exactly `sha1(f"{percept_id}:{index}")`. It is stable for the lifetime
of a Percept (and across its serialisation), which is what `verify` needs. Cross-request
tracking is explicitly out of scope for v0.1 (§1.2). Recorded here so nobody "fixes" it into
a hash of the mask.

## D6 — One tool call per request, not per camera

*Conflict:* §5.3 says "并行多相机由服务层调度（每相机独立调用工具）", but
`ToolContext.images` is a `dict[camera -> ndarray]` and `multi_view_3d` needs all views in
one call.

*Decision:* the pipeline calls each tool **once per request** with every camera in
`ctx.images`, and tools iterate cameras internally (one Percept per camera×prompt). The
service layer guards concurrency with the registry's per-tool lock (§11.2) instead of
parallelising cameras. This keeps `multi_view_3d` coherent and avoids per-camera model
re-entry.

## D7 — `config.py` uses pydantic + pyyaml, not `pydantic-settings`

*Conflict:* §13's file layout mentions `pydantic-settings`, but its dependency list does not
include it.

*Decision:* plain pydantic models + `yaml.safe_load`. No new dependency. `logging.json` is
exposed as `logging.json_output` with the YAML alias `json`, because a pydantic field named
`json` shadows `BaseModel.json`.

## D8 — Client uses stdlib `urllib`, not `httpx`

*Decision:* `httpx` is only in the `dev` extra (spec §13 marks it "(测试)"). `PerceptaClient`
uses `urllib.request` so the runtime package needs no HTTP client beyond the stdlib.
`httpx` remains the test client for the contract suite.

## D9 — Default pipeline vs. CI

*Decision:* the code default stays faithful to §5.3 (`["sam3", "multi_view_3d"]`), while
`configs/ci.yaml` and `configs/example.yaml` pin `["fake", "multi_view_3d"]` so the service
is runnable without weights. `percepta serve` with no `--config` uses the code default and
reports `sam3` as `error`/`unavailable` in `/healthz` — that is expected and must not crash.

## D10 — RLE includes no COCO compression

*Decision:* appendix B's alternating run lengths, stored as a plain `list[int]`. `sum(counts)`
must equal `H*W` (validated in `Mask`), and the decoder also accepts a leading `0` run for a
first pixel of 1. No string/byte compression in v0.1.

## D11 — `mask_iou` on two empty masks is `0.0`

*Decision:* IoU is undefined for `0/0`; returning `0.0` prevents `verification.stable` from
treating two missed detections as a match.

## D12 — `PipelineRunner` and `pipeline.py` are additive

*Conflict:* §13's layout lists no orchestrator module, yet both the HTTP service and
`client.perceive_local()` need identical pipeline semantics.

*Decision:* add `src/percepta/pipeline.py` with `PipelineRunner` (see `AGENTS.md` §4) and
have both callers use it, rather than duplicating orchestration.

## D13 — `VerificationResult.status` precedence

*Conflict:* §10 defines `status: Status` but never says what it means for the comparison.

*Decision:* `ok` when every required check passed; otherwise `not_found` if either percept
is `not_found`; otherwise `invalid_request` when `action`/`options` are malformed; otherwise
propagate `after.status` (`occluded`/`unavailable`/`timeout`/`invalid_request`). When
`after.status == "ok"` and only a substantive check failed (say the object rose 4 cm instead
of 10), `status` stays `"ok"` and the verdict lives in `ok`/`checks` — the percept was fine,
the action was not. `details` carries numeric markers (`geometry_available`,
`masks_available`, `frame_mismatch`, `ee_delta_available`, …) instead of prose, because the
frozen `VerificationResult` has no `message` field.

## D14 — `stable` needs a third observation

*Decision:* `checks["stable"]` is only computed when `options["stable_with"]` is supplied (a
`Percept`, `Observation`, `Mask`, or bool ndarray). `required` defaults to "every applicable
check" and may be narrowed, but narrowing can never make `not_found` pass. Asking for
`required=["stable"]` without `stable_with` is a malformed request: `ok=False`,
`checks["stable"]=False`, `status="invalid_request"`.

## D15 — A geometry stage supplies `fused`; it is never duplicated

*Conflict:* §5.3 puts `multi_view_3d` in the default pipeline, while §9 describes fusion as
a step of the response. Doing both computes geometry twice and emits a camera-less,
observation-less Percept inside `percepts`.

*Decision:* if a pipeline tool reports `capabilities().kind == "geometry"`, its single ok
Percept becomes `response.fused` and is **excluded from `percepts`**. `fusion.fuse()` is only
called when `options.fuse` is true *and* no geometry-kind stage ran (e.g.
`options.tools=["fake"]`). A geometry stage that yields nothing gives `fused=None` +
`geometry_unavailable`, not a second fusion attempt. A geometry Percept with status
`unavailable`/`invalid_request` stays visible in `percepts`, because that is a real failure
the caller must see (§16.4).

## D16 — Prism cross-section is the multi-view visual hull at mid-height

*Conflict:* §9.2/D4 fix the *height search* but not how the cross-section (hence the centroid,
bbox and yaw) is taken.

*Decision:* the height comes from D4 verbatim (smallest `h` reaching the IoU threshold, else
arg-max + `geometry_low_confidence`). The reported footprint is the **intersection of all
views' cone slices at mid-height**, not one source view's slice: a single view's slice is a
"shadow" that biased the centre by 5–12 cm at 90–140° view separation, versus 0.2–1.5 cm
when intersected. Consequences, documented in `fusion.py`: the prism `bbox3d` is a
visual-hull **upper bound** (no depth ⇒ no surface to fit), the height search exits at the
first passing height (~25× faster, same answer), `table_plane` is honoured by its height at
the object and extrusion is along world `+z` (a tilted plane triggers
`table_plane_tilted`), and single-view geometry assumes `single_view_rough_height_m`
(default 0.10 m) with `uncertainty_m >= 0.03`.

## D17 — An unavailable tool stays HTTP 200 (explicit deviation from appendix A)

*Conflict:* appendix A lists `tool_unavailable` with HTTP `503`, but annotates it
"(Percept.status=`unavailable`)" — a 503 body cannot be a `PerceiveResponse`, and in a
multi-tool pipeline one dead tool must not discard what the others produced.

*Decision:* `POST /v1/perceive` answers **200** with the affected Percepts carrying
`status="unavailable"` plus a `tool_unavailable` warning; `/healthz` reports the tool as
`error`. This is a deliberate, recorded deviation from appendix A's HTTP column and is
flagged for user review — the alternative (503 carrying a `PerceiveResponse`) is a one-line
change in `service/routes.py` if preferred.

## D18 — `capabilities().prompts == ∅` means "prompt-agnostic"

*Decision:* a tool that declares an empty prompt set ignores `prompts` entirely (that is how
`multi_view_3d` consumes `ctx.extras["upstream"]`). It must not be reported as
`unsupported_prompt` for every camera×prompt. Tools that declare a non-empty set still reject
anything outside it (S8).

## D19 — Disabled tools are absent from health/introspection

*Decision:* `config.tools.<name>.enabled: false` removes the tool from `/healthz`,
`/v1/tools` and `/v1/info`; absence is the honest representation of "not part of this
deployment". The frozen vocabulary `ok|loading|error` is preserved, and `lazy` (imported,
loads on demand) still maps to `ok`. Listing a disabled tool as `ok` would tell operators
SAM3 is available when it was never attempted.

## D20 — Cache key covers only result-affecting options; first stage only

*Decision:* the key is `v1:` + `sha256` over the sorted per-camera image hashes, the prompt's
pydantic JSON, the tool name, and — appended only when supplied — `masks`, `min_score`,
`top_k`. `tools`, `fuse`, `timeout_ms`, `cache` and `context` are excluded (§19.6). Caching
applies only to the first pipeline stage: later stages depend on upstream Percepts, which the
key cannot represent.

## D21 — Unknown tool name is a 400; an unloadable one is a 200 Percept

*Decision:* a name the registry does not know is a caller error (`invalid_request`, HTTP 400,
same class as the §4.6 size mismatch). A tool that is configured but fails to load, or loads
and then cannot serve, produces `status="unavailable"` Percepts inside HTTP 200 + a
`tool_unavailable` warning. Neither case may ever be reported as `not_found`.

## D22 — Dependency management is uv, with a portable lock file

*Context:* the project previously used `pip install --user` with unpinned ranges, which was
neither reproducible nor isolated (the host has no `python3-venv` and no passwordless sudo).

*Decision:* `uv` with a committed `uv.lock` resolved against the **official** PyPI index, and
`.python-version` pinning 3.10. Dev tooling moved to a PEP 735 `dependency-groups.dev`
installed by default, because a fresh clone must be able to run the test suite.

The lock file must never be generated under a mirror: doing so bakes mirror URLs into every
download entry and breaks any machine that cannot reach that mirror. This host is an example —
PyPI official takes ~23 s per request through its proxy while a domestic mirror takes 0.15 s,
so it is tempting to re-lock under the mirror and quietly make the lock non-portable.

## D23 — Two dependency constraints were wrong and were corrected

*Decision:* `torch>=2.1` → `torch>=2.7,<3` (sam3 0.1.4 requires 2.7+ and does not work on
2.1), `sam3` → `sam3==0.1.4` (its 0.1.x constructor/processor API still moves, so an open
range guarantees breakage), and `numpy` gained the `<2.4` ceiling sam3 declares but the
project had left unconstrained. These were latent bugs found by reading the real package
rather than the spec's description of it.

## D24 — Model isolation without containers: a backend abstraction

*Context:* two perception models may require mutually exclusive `torch`/`numpy` versions. A
single Python process can hold only one of each, so this cannot be configured away.

*Decision:* "where the model runs" becomes configuration (`tools.<name>.backend`) rather than
code, behind a small protocol (`set_image` / `segment` / `close` / `describe`):

* `inprocess` — same environment; default, no IPC;
* `subprocess` — the model lives in its own virtualenv, driven over a binary pipe;
* `http` — a separate service, possibly on another host.

Containers remain the last resort, reserved for CUDA/system-library conflicts that process
isolation cannot fix. Measured on this host: two venvs with `numpy` 1.26.4 and 2.2.6 coexist,
a parent process drives a model in the other venv, and interpreter cold start is 0.01 s.
Design and wire protocol: `docs/model-isolation.md`.

## D25 — The wire protocol is binary, not JSON

*Decision:* request/response frames are length-prefixed binary with magic numbers and a
version field; only the prompt and status travel as JSON. Rationale: a 640×480×3 frame is
921 KB raw but ~15 MB as JSON text, and a magic check turns stream desynchronisation (a
crashed worker, a stray `print()` on stdout) into an immediate error instead of garbage
pixels. The worker logs to stderr only — stdout is the protocol.

A killed or unresponsive worker maps to `status="unavailable"`, never `not_found` (the same
rule as D17/§16.4: "I could not look" must stay distinguishable from "I looked and found
nothing").

## D26 — `percepta/__init__` must import cheaply

*Context:* a worker running in an isolated virtualenv imports
`percepta.backends.protocol` and `percepta.schema`. An eager `from .config import …` in the
package `__init__` dragged in pyyaml, and `.masks` dragged in pillow — so a worker in a slim
environment failed with `ModuleNotFoundError: No module named 'yaml'` before it ever reached
the model.

*Decision:* only `schema` (pydantic + stdlib) is imported eagerly; every other public name
(`AppConfig`, `ToolRegistry`, `encode_rle`, `PerceiveClient`, `compare`, …) is resolved
lazily via module `__getattr__`. The consequence is a hard rule for worker environments: a
worker needs exactly `numpy` + `pydantic` beyond its own framework, which is what makes the
isolation boundary thin enough to be practical.

## D27 — The library does not manage model dependencies, but does police model assets

*Context (dependency side):* an earlier iteration shipped a `requirements/` directory with
one pinned dependency file per model, plus a `docker/` directory. Both were removed on the
grounds that a *library* should not maintain a list of what a *model* needs: it would have to
track every upstream model's release cadence, and different deployments have different
CUDA/driver combinations for which no universal answer exists.

*Decision (dependency side):* the repository ships no model dependency manifests. Model
frameworks are installed by the deployer into whatever interpreter they like, and percepta is
told where via `tools.<name>.backend` / `python`. What the library does guarantee is
narrow and durable: it **finds** third-party tools (the `percepta.tools` entry-point group),
**calls** them through the backend protocol, and **constrains their assets**.

*Context (asset side):* weights are the failure mode that hurts most, because it is silent —
a truncated or wrong-version checkpoint can load successfully and then produce quietly wrong
segments. The previous code only *recorded* a hash; it never compared one.

*Decision (asset side):* every tool that needs weights declares
`checkpoint` (a path or a remote URI), an optional `checkpoint_sha256`, and a
`verify_weights` mode:

| mode | behaviour | purpose |
|---|---|---|
| `if-declared` (default) | verify when a hash is configured, otherwise record only | exploration stays usable; production configs become safe by default |
| `always` | refuse a local file with no declared hash | CI / production |
| `never` | skip verification | escape hatch for huge weights on slow disks |

`percepta models check` performs the same check before the service starts, and reports
existence, observed hash, and whether it was *actually compared* — `verified: false` means
"recorded only" and is never overstated to `true`.

One trap is worth recording: YAML parses an all-digit hash (e.g. 64 zeros) as an **integer**,
and `int(0)` is falsy — an `or`-chain then dropped the expected hash and skipped verification
*without a word*. The truncated-checkpoint case could therefore load silently. The code now
requires an explicit `is not None` and rejects a numeric-looking hash with a message telling
the operator to quote it in YAML. A test pins both halves: mismatch fails loudly, and an
unquoted numeric hash is rejected rather than silently ignored.

## D28 — Real SAM3 verified over the subprocess backend; four upstream traps recorded

*Context:* SAM3 was finally run for real (3.2 GiB checkpoint, hash
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, on an H20 via the
`subprocess` backend in `.venvs/sam3`). This settles the question the spec left open at S9.

*Decision:* the subprocess backend is **validated for real**, not just in theory. Against a
real photo (640×427) it returned correct masks/boxes/scores at the original resolution, with
`model_sha256` recorded in provenance. Numbers are in `docs/model-isolation.md` §10:
20 s one-time model load, 253–520 ms per prompt, 127 ms for a second prompt on the same frame.

Four upstream traps were found and are now documented (§10.2–10.6). The most dangerous is
that `Sam3Processor.set_image` accepts HWC numpy **without complaining** while internally
treating it as CHW — a `(480, 640, 3)` frame becomes C=480/H=640/W=3 and every prompt silently
returns zero instances. The fix is to pass a PIL image. This is exactly the kind of defect
that a mocked backend can never catch, which is why the "ship a fake adapter *and* run the
real thing once" rule exists.

Also fixed as a result: `percepta.tools.sam3` now resolves and records the weight hash in the
**parent** process (a subprocess worker keeps the model elsewhere, so provenance would
otherwise be blank), and `SubprocessBackend.segment()` raises `unavailable` when the worker
has died instead of silently respawning it and blaming the caller for not calling
`set_image()`.

Testing note (D28.1): the GPU test **requires a real photograph** and skips without one.
A synthetic colour-block scene is not evidence — SAM3 legitimately finds nothing in it, and
letting that "pass" would manufacture false confidence.

## D29 — Warm-up runs in the background, through the whole pipeline

*Context:* the first SAM3 call costs ~16 s (worker spawn + 3.2 GiB checkpoint read + CUDA
init + cuDNN autotuning); subsequent calls are 137-520 ms. An Agent Policy sending its first
request into that cold path either times out or has to branch on "first call is slow".

*Decision:* the service runs one throwaway request at startup, in a **background thread**, and
reports the outcome on `/healthz` as `warmup: {status, duration_ms, error?}`.

Measured on this host with real SAM3: first real request **16339 ms → 137 ms** (119×), with
identical results.

Four details that matter:

1. **Background, not blocking startup.** A service that refuses connections for 16 s looks
   broken to a supervisor, a container healthcheck, or a load balancer. The port binds
   immediately; `/healthz` distinguishes "listening but cold" (`warmup.status: running`) from
   "wrong port".
2. **Through the real pipeline**, not a bare model call — the cost being removed spans image
   decode, prompt handling, the forward pass and result assembly.
3. **Never fatal.** A failed warm-up is recorded (`status: error`, with the reason) and the
   service keeps serving. `ok` on `/healthz` stays true: a broken model is already reported by
   `tools[].status` (D9), and liveness must not depend on a model loading.
4. **`not_found` counts as success.** The warm-up frame is synthetic; the model correctly
   reports no instances. Only `unavailable`/`invalid_request`/`timeout` count as failures.

The previous `tools.sam3.warmup` flag was removed: it ran a 32×32 all-zero frame, which
exercised different kernel shapes than a real frame (so the expensive autotuning still landed
on the first real request), forced a second full checkpoint read, and was invisible to
operators.

## D30 — `wait_until_ready` waits for warm-up, not just the port

*Context:* the client already had a ``wait_until_ready`` that returned as soon as
``/healthz`` answered ``ok: true``. With warm-up added (D29) that became actively misleading:
``/healthz`` reports ``ok: true`` the moment the process listens — deliberately, so a
supervisor never sees a dead window — while the model is still cold. A client calling
``wait_until_ready`` then immediately firing a request gets the full cold cost anyway.

*Decision:* ``wait_until_ready`` also waits for ``warmup.status`` to settle
(``ok``/``error``/``disabled``; ``unknown`` covers servers that predate warm-up, keeping
backward compatibility). ``warm=False`` restores the old port-only behaviour. A *failed*
warm-up counts as settled: the service is up, the failure is already reported by
``tools[].status``, and blocking forever would be wrong.

Measured from a genuine cold start, real SAM3:

| stage | time |
|---|---|
| initial ``warmup_state()`` | ``running`` |
| ``wait_until_ready`` | blocked **39.7 s**, returned ``True`` |
| warm-up itself | 45.6 s (GPU 2 was busy; 15 s on an idle card) |
| first real request afterwards | **343 ms** |

Without the fix that first request would have paid the 45.6 s itself. The default timeout was
raised from 10 s to 120 s for the same reason: 10 s cannot cover a cold model load, so the
old default would have reported "not ready" on a perfectly healthy server.

Also observed: warm-up cost is **not constant** — 15 s on an idle H20 versus 45–73 s when
another workload shares the GPU. The timeout therefore has to be generous, and
``/healthz.warmup.duration_ms`` exists so operators can see the real number rather than guess.

## D31 — Introspection reads metadata from the tool class, not the instance

*Context:* `/healthz` and `/v1/tools` carried their own ``entry_version`` helper that returned
the literal ``0.0.0`` whenever ``entry.instance is None``, while the CLI went through the
registry and showed the real version. The same deployment therefore reported *different*
versions depending on which surface you asked — and a tool that is registered but not loaded
(no GPU, no weights) looked like a broken plugin rather than a known one.

*Decision:* ``version`` and ``kind`` are class attributes, so ``ToolEntry.describe()`` reads
them without instantiating anything, and it is the single source of truth for `/healthz`,
`/v1/tools` and `percepta tools list`. The duplicate helper in ``service/routes.py`` was
deleted. ``describe()`` returns ``{name, version, status, kind, loaded}``.

## D32 — An empty `cuda_visible_devices` means inherit, not "hide every GPU"

*Context:* this host has 8 H20s shared with other workloads and no permanently free card, so
pinning an index in the config made the worker's speed depend on whatever else landed there
(warm-up measured at 15 s on a quiet card versus 45–73 s when shared). Leaving the value blank
to mean "auto-select" then exposed a real bug: the backend did
``if cuda_visible_devices is not None: env["CUDA_VISIBLE_DEVICES"] = ...``, and
``CUDA_VISIBLE_DEVICES=""`` means **hide every GPU** to CUDA — a worker that cannot see any
device at all.

*Decision:* empty and whitespace-only mean "inherit the parent environment"; an explicit value
is stripped and exported; unset inherits. The rule lives in ``SubprocessBackend._worker_env()``
so it can be tested directly rather than by asserting an attribute.

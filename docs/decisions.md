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

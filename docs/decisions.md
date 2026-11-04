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

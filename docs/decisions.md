# Decisions taken where the spec was ambiguous (spec §19)

Spec §0.5 says: "有歧义就在 §19 记录问题并选一个保守实现，不要阻塞。" These are those
decisions. **Any agent may read this file; only the Lead edits it.** Each entry names the
spec clause it resolves.

> **现行说明与修订记录（2026-11-12，核对代码 `534066a`）**：各决策原文保留为历史记录；
> 后续决策替代了早期实现时，以下带日期的说明指出替代关系。D4/D16 的高度和轮廓算法以 D35
> 为准，D27 的依赖清单边界以 D33 为准，D36 的部署进度以 D37/D38 为准。
> 本次只统一说明，不改变代码、接口或功能范围。
> 本轮各处原文、修订内容和原因汇总于[文档修订记录](documentation-changes.md)。

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

> **2026-11-12 修订说明（由 D35 替代算法部分）**：上文原定“smallest height whose best score
> is ≥ … threshold”保留为历史。当前 `fusion.py` 搜索 IoU 峰值；阈值只用于低置信度判断，
> 不在首次达标处停止。原因及旧规则的高度偏差见 D35。

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

> **2026-11-12 术语更新**：上文“`dev` extra”保留旧称。当前 `httpx` 位于 PEP 735 的
> `dependency-groups.dev`，由默认 `uv sync` 安装；不是根项目的可选 extra，也不进入最小运行时。

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

> **2026-11-12 现行补充**：上文“a geometry stage that yields nothing gives `fused=None`”
> 只适用于此前没有成功几何结果的情况。`PipelineRunner._take_geometry()` 在当前阶段的成功
> geometry Percept 中选择最后一个，后一个成功阶段覆盖前一个阶段的 `fused`；当前阶段没有
> 成功结果则保留已有 `fused`，错误仍在 `percepts` 中。代码没有“第一个成功结果生效”规则，
> 也不自动比较方法质量或调用备用工具。只有未执行任何 geometry 阶段且 `options.fuse=true`
> 时才调用通用 `fusion.fuse()`。`options.fuse=false` 不让 pipeline 跳过显式 geometry
> 阶段；`multi_view_3d` 自身调用通用融合并遵守该选项，当前 `stereo_3d` / `triview_3d`
> 则不读取该开关，仍可填充 `fused`。这是顺序执行与结果保留语义，不是自动 fallback。
> 这里的顺序是 `AppConfig.resolve_pipeline()` 解析后的顺序：它会把启用且已选择的
> `multi_view_3d` 移到末尾，不一定保留请求列表的原序。因此当其成功时，可能覆盖此前
> `stereo_3d` / `triview_3d` 的结果；这也不是按质量选择方法。

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

> **2026-11-12 修订说明（由 D35 替代）**：上文“cone slices at mid-height”和“first passing
> height”是保留的旧方案。现行 footprint 对平面网格的整根竖直柱体取轮廓约束，柱内采样高度
> 为 0.15/0.5/0.85；高度取 IoU 峰值。改动解决中高切片将阴影当 footprint、首次过阈值
> 低估高度的问题。坐标、标定约束及单视角粗略高度的原则继续有效。

## D17 — An unavailable tool stays HTTP 200 (explicit deviation from appendix A)

*Conflict:* appendix A lists `tool_unavailable` with HTTP `503`, but annotates it
"(Percept.status=`unavailable`)" — a 503 body cannot be a `PerceiveResponse`, and in a
multi-tool pipeline one dead tool must not discard what the others produced.

*Decision:* `POST /v1/perceive` answers **200** with the affected Percepts carrying
`status="unavailable"` plus a `tool_unavailable` warning; `/healthz` reports the tool as
`error`.

**Resolved in v0.2.2 — appendix A was the thing that was wrong.** The row read
"503 (Percept.status=`unavailable`)", which contradicts itself: a 503 response body is not a
`PerceiveResponse`, and a Percept only exists *inside* one. Rather than keep an implementation
that permanently disagreed with its own spec, the table was corrected to 200 and D17 is no
longer recorded as a deviation.

The deciding argument is that this endpoint is inherently a **batch** operation (one request
may span several cameras × prompts × tools). Answering 503 because one tool is unavailable
discards two things at once: the results other tools already produced, and the information
about *which* tool failed. Callers should read `percepts[].status` and `warnings`, not just the
status code — and a supervisor reading `/healthz` still sees the tool as `error`.

Locked down by five contract tests, including one that runs a healthy and a broken tool in the
same pipeline and asserts the healthy tool's observations survive. Verified by mutation: forcing
the endpoint to 503 makes three of them fail immediately.

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

> **2026-11-12 修订说明（依赖部分由 D33 替代）**：上文“the repository ships no model
> dependency manifests”不再描述当前仓库。现在每个已部署模型在 `uv_envs/<model>/` 中提交
> 独立的 `pyproject.toml`、`uv.lock` 和 Python 版本；这些模型框架依赖仍不进入根默认依赖。
> 部署方使用独立解释器，库通过 backend 调用；本条关于权重路径、hash 和自检的资产契约继续有效。

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

## D33 — Model runtimes are separate locked projects under `uv_envs/`, not extras

*Context:* the root `pyproject.toml` carried `[project.optional-dependencies] sam3 =
["torch>=2.7,<3", "sam3==0.1.4"]`. It was broken by construction, in three separate ways:

1. resolving `torch` from PyPI yields the **CPU build** (2.14.1, no ``+cu128``), so
   `uv sync --extra sam3` installs an environment with no GPU support;
2. sam3 0.1.4 imports ``psutil`` at module scope without declaring it, so `import sam3` fails;
3. it silently **disagreed** with `uv_envs/sam3/.venv` — the environment that actually runs
   the model (torch 2.10.0+cu128) — giving two different "SAM3 environments" for one repo.

The root cause is structural: a model runtime needs its own package index (PyTorch's CUDA
wheels), its own pins, and its own lock. That cannot be expressed as an extra of the library.

*Decision:* each model runtime is a standalone uv project with a committed
`pyproject.toml` + `uv.lock` under `uv_envs/<model>/`, rebuilt with
`uv sync --project uv_envs/<model>`. The generated `.venv/` is git-ignored; the definition is
committed, because reproducibility of a model runtime matters as much as the library's own.

The directory is named `uv_envs`, not `envs`: in a robotics codebase "envs" reads as
simulator environments, and the ambiguity is not worth the shorter name.

The root `pyproject.toml` therefore has **no** `sam3` extra. The library still constrains
model *assets* (`checkpoint_sha256`, D27) — dependencies are the deployer's, assets are the
library's contract.

## D34 — Lockfile portability is enforced by CI, not by documentation

*Context:* `uv` writes the absolute download URL of every package into `uv.lock`. Generating a
lock while `UV_DEFAULT_INDEX` points at a mirror therefore bakes that mirror into ~600 entries,
and the lock works only on machines that can reach it. This repository shipped a lockfile with
**631 aliyun URLs** — twice. Both times the rule "do not lock against a mirror" was already
written in `AGENTS.md`; both times it was violated anyway, because the mirror is 150× faster
and the mistake is invisible until someone else clones the repo.

*Decision:* a whitelist checker runs as the **first** CI step
(`scripts/check_lock_portability.py`, decision enforced by
`.github/workflows/ci.yml`). Only `pypi.org`, `files.pythonhosted.org` and PyTorch's two index
hosts are accepted. Unknown hosts fail too, with a message pointing at `ALLOWED_HOSTS` — adding
a host should be a deliberate, reviewable act rather than a silent slide.

Two details that matter:

* **Whitelist, not blacklist.** A new unknown host is more likely a mistake or a supply-chain
  risk than an intentional choice, so it must be justified explicitly.
* **It runs before `uv sync`.** A non-portable lockfile fails *every* download, so checking it
  first converts a screenful of "not found" errors into one actionable message naming the
  mirror and the fix.

`uv lock --check` is *not* sufficient on its own: it validates that the lock matches
`pyproject.toml`, not that the URLs inside it are reachable from anywhere. Both are now run.

## D35 — The prism footprint is a column test, and the uncertainty is split in three

*Context:* two defects in the no-depth path, both found by measuring against a synthetic cube with
a known ground truth (`scripts/footprint_accuracy.py`).

**The footprint was the object's shadow.** The cross-section was the cone slice at mid-height: where
each camera's silhouette rays cross that plane. For an oblique camera that polygon is larger than
the object by roughly ``h/2 · tan(θ)``, and intersecting several such shadows cancels only part of it
because they disagree in different directions. Measured on a 10 cm cube: **178–293 cm² against a true
100 cm² (1.8–2.9×)**.

**One uncertainty hid two unrelated numbers.** ``uncertainty_m`` could not say whether the lateral
position or the height was the weak one — and they differ by more than an order of magnitude: one
mask pixel moves the footprint a few millimetres while leaving the height *guessed*.

*Decision (footprint).* A plane cell belongs to the object **iff its whole vertical column** lies
inside every mask, sampled at interior heights (0.15, 0.5, 0.85 of the column). This is the
discrete silhouette-restricted visual hull, and for a box it is exact. Computed on a ground-plane
grid (`src/percepta/hull.py`) rather than as a convex polygon, because a three-view hull is
generally non-convex and any polygon representation would discard exactly the information that
matters.

*Decision (height).* The height is the **peak** of the IoU curve, not the first height that clears
the threshold. Spec D4 asked for the smallest qualifying height, which stops far too early: on a
10 cm cube the curve crosses 0.5 at h≈0.04 and peaks at 0.10 (IoU 0.96), so the old rule reported a
4 cm object and dragged ``top_z`` and the centroid's z down with it. The threshold now only gates
confidence.

*Decision (uncertainty).* ``Geometry`` gains three optional fields (additive, per spec §3):
``lateral_uncertainty_m``, ``height_uncertainty_m`` and ``view_disagreement_m``. ``uncertainty_m``
keeps its frozen meaning (the worst axis) so no existing caller silently becomes more confident.
``view_disagreement_m`` reports the spread between per-view fits, which neither uncertainty can
express: the average of a good and an occluded view looks perfectly confident.

Two further defects surfaced while fixing the above and are now pinned by tests:

* sampling the column at fractions **0.0 and 1.0** made every column fail. A mask covers the
  object's interior, and its base and top edges *are* the boundary — so the correct columns read
  "outside" and a 15 cm cube collapsed to a 2 cm² footprint with a zero score.
* a coarse ``prism_step_m`` collapsed the height search to a single sample, which cannot have a
  peak. ``MIN_PRISM_STEPS`` floors it.

Result (`scripts/footprint_accuracy.py`, cube side 10 cm):

| rig | centroid err before → after | height err before → after |
|---|---|---|
| 0/120/240 | 1.03–1.52 cm → **0.33–1.04 cm** | 0–3 cm → **0–2 cm** |
| 0/180 | 0.5–2.0 cm → **0.5–1.0 cm** | 1–4 cm → **1–2 cm** |
| 0/90/180/270 | 0–1.5 cm → **0.01–1.5 cm** | 0–3 cm → **0–3 cm** |

The footprint ratio moved from 1.78–2.93× to 0.65–2.18×. It is still an *upper bound* on a convex
body — that is what a silhouette hull is — so the remaining excess is expected, not a bug.

> **2026-11-12 审计限制（计算尚未修复）**：上文原文保留为 D35 的方法说明和当时结果，
> 不代表当前所有轮廓分支或不确定度计算都已正确。本轮另确认两项既有问题：
> `fusion._hang_cells()` 把世界 XY polygon 直接传给像素 rasterizer，未转换成 grid 像素坐标，
> hang 候选可能为空或错位；`fusion._lateral_uncertainty()` 使用
> `norm(hit[0] - origin[0])`，其中 `origin[0]` 是光心的 x 标量，导致结果依赖任意世界原点。
> 详见[文档修订记录中的 DOC-20261112-01／02](documentation-changes.md)。本轮只更新说明和登记
> 限制，没有修正这两处计算，也不把现有测试通过解释成它们已经解决。

## D36 — FoundationStereo is an isolated disparity backend, not a segmenter

The first deployed neural stereo model is FoundationStereo 23-51-11 (research license),
selected by `stereo_backend="foundation_stereo"`. It is distinct from Fast-FoundationStereo
(`ffs`, still unavailable). Its Python 3.11 / torch 2.4.1 runtime has its own locked project;
root dependencies, SAM3's binary protocol and the public schema are unchanged.

The dedicated worker transports two rectified uint8 RGB images and one float32 pixel-disparity
map. Startup and pipe IO have enforced deadlines; worker failure is `unavailable`, never a
silent SGBM fallback. Interpreter/source/checkpoint paths come only from trusted deployment
configuration, not request context. Checkpoint hashes go into provenance.

Geometry uses OpenCV +Z forward. For camera-to-world rotations/centres `Ra,Ca,Rb,Cb`,
OpenCV stereo rectification receives `R=Rb.T@Ra`, `t=Rb.T@(Ca-Cb)`. RGB and the optional
original-left mask are actually remapped; both cameras' valid support and the right-image
correspondence must be valid. Rectified points return through `R1.T` before camera-to-world
transformation. Explicit left/right order is honoured; unsupported reverse/vertical pairs
are rejected rather than silently swapping cameras. Triview's pairwise route shares these
geometry corrections, but gains no neural model in this deployment.

The result summarizes visible points: without `stereo_mask` it describes the scene, not the
object named by a prompt. Disparity-noise uncertainty remains an assumption, not a measured
accuracy claim. Real official-image, worker and HTTP smoke checks establish executability and
coordinate consistency, not task-level accuracy or ground-truth object localization.

The asset manifest pins upstream source, the official checkpoint/config and timm's EdgeNeXt
backbone. One hash-pinned source patch makes DINO load bundled code instead of fetching a
remote torch.hub repository. Inference uses the prepared local cache with HF offline mode.

> **2026-11-12 修订说明（部署进度由 D37/D38 更新）**：上文“`ffs`, still unavailable”和
> “gains no neural model in this deployment”记录首轮 FoundationStereo 部署时的状态。
> 当前 FFS 已接入 `stereo_3d`，MVSAnywhere HERO 和 DA3-BASE 已接入 `triview_3d` 的 `mvs`
> 路线，均使用独立运行时。D36 的双目坐标、显式失败和可见表面语义保持有效；本条不表示
> 已实现目标分割、自动配对或模型间 fallback。

## D37 — Deploy the remaining three depth models in independent runtimes

Fast-FoundationStereo 23-36-37, MVSAnywhere HERO and DA3-BASE each receive a separate
locked uv project, source/asset manifest, bootstrap, adapter and real GPU smoke. No model
framework enters the root dependencies or module-level tool imports. Source, checkpoint
paths and adapter factories are trusted deployment configuration; requests carry RGB and
calibration only. Existing SAM3 defaults and the public schema remain unchanged.

Fast-FoundationStereo uses `stereo_backend="ffs"` and the existing stereo worker protocol.
Its official checkpoint is a structurally pruned serialized model; only the exact
manifest-pinned official bytes may be unpickled in the isolated worker. The unchanged
checkpoint has 26 positional tokens and supports at most 416 disparity pixels. Actual
640-pixel inference fails; reject it explicitly instead of interpolating model weights or
claiming the planned near-range coverage was achieved.

`triview_method="mvs"` with `triview_backend="mvsanywhere"` or `"da3"` uses calibrated
three-view RGB and returns metric camera +Z depth on the input pixel grid. A bounded
three-view binary protocol reuses the stereo worker lifecycle, IO deadlines and cleanup.
Protocol validation rejects nonfinite arrays, invalid rigid calibration and coincident
camera pairs before inference. Worker errors produce `unavailable`; missing depth produces
no Geometry. There is no substitute-model fallback.

MVSAnywhere jointly conditions each reference on the other two images. The official
in-memory API already uses OpenCV camera-to-world poses. Its model consumes only relative
camera transforms; rebase poses to the first camera in float64 before float32 conversion
to avoid world-origin cancellation. The world Geometry still uses the caller's original
`T_world_cam`. Sky probability masks sky; it is not advertised as depth confidence.
Offline source patches suppress redundant backbone downloads, then the full trained HERO
state loads strictly. The official checkpoint hash is checked before trusted deserialization.

DA3-BASE uses local safetensors and known world-to-camera poses, with
`align_to_input_ext_scale=True`. Three non-collinear camera centres are required for its
Sim(3) alignment; no relative depth is silently interpreted as metres. Safetensors' strict
`load_model` honors the checkpoint's shared LayerNorm aliases. Raw confidence remains an
uncalibrated model score. Shared tool Geometry uses an explicit 0.03 m uncertainty assumption,
with an uncalibrated warning; this value is not measured positioning accuracy.

These models summarize visible surfaces of the supplied scene. Model deployment does not
implement target segmentation, object completion, automatic camera selection or RoboDojo
Policy evaluation. Fast-FoundationStereo and HERO checkpoints are research-only; DA3-BASE
source and weights are Apache-2.0.

## D38 — Preserve workers on invalid DA3 sizes, and package runtime metadata

DA3's image processor can round an extremely short edge to one pixel, which violates the
model's 14-pixel patch requirement. Validate the processed dimensions before IPC and at
the direct adapter boundary. A rejected request is `invalid_request`; the resident worker
and its next valid request remain usable. Runtime/protocol failures remain terminal.

Rebase DA3 camera-to-world poses to the first camera in float64 before the upstream path
casts poses to float32: `T_local_cam = inv(T_world_cam[0]) @ T_world_cam`. Feed the inverse
local poses to DA3 and keep the original calibration for final world Geometry. This prevents
world-origin translation from erasing centimetre-scale baselines; regression checks include
common translations of 1e6 and 1e7 metres while holding images and relative poses fixed.

The wheel includes the existing FFS and MVSAnywhere asset manifests as package resources.
Editable source checkouts retain the uv project manifests as their single source. Verify a
built wheel outside the checkout, and also build a wheel from the source distribution.

Resident backends expose a nonblocking `health_error` snapshot. Registry descriptions record
terminal failures and idle worker exits as tool `error`, without acquiring the inference lock
or changing singleton/restart semantics. Service health and introspection share that record.

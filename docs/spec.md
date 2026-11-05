# percepta — 功能规格与开发说明书（v0.1 draft）

> 本文档是给**实现者（人或 Agent）**看的规格书，不是用户手册。目标是把"一套可插拔的机器人感知层"定义到**接口无歧义、验收可自动判定**的程度。实现时不要自行发明字段、端点或语义；确需变更，先改本文档并升版本。

---

## 0. 怎么用这份文档（给 Agent）

1. 先读 §1–§5（目标、数据模型、坐标约定），再读你要负责的模块与 §16 验收。
2. **§4/§5/§6/§7 是冻结接口**：字段名、枚举值、JSON 形状、HTTP 路径不得改动；新增只能"加可选字段"，且需在 §18 记录。
3. 任何"模型相关"的实现都必须能被 **fake 工具**替换，保证 CI 无 GPU 可跑。
4. 完成定义（DoD）：代码 + 单测 + 验收场景通过 + 文档更新 + 不新增未声明的第三方依赖。
5. 有歧义就在 §19 记录问题并选一个保守实现，不要阻塞。

---

## 1. 目标与非目标

### 1.1 目标

`percepta` 是**感知层（observation layer）**：把多个感知模型（开放词表分割、检测、深度、VLM 定位、几何）统一成**同一种可序列化的观测对象**，让 Agent Policy（代码策略/VLA/工具链）用一致的方式消费，并支持：

- 一张或多张 RGB（多相机）+ 可选深度；
- 文本 / 框 / 点 三种提示；
- 输出实例 mask、框、分数、2D 统计；
- 用标定做**多视角几何 → 3D 位置/尺寸/朝向**；
- 动作前后观测对比（验证/证据）；
- 工具插件化：SAM3 只是默认实现，可替换/叠加（检测器、深度、VLM grounding）。
- 既可进程内调用（Python SDK），也可作为 loopback HTTP 服务（多机部署）。

### 1.2 非目标（v0.1 明确不做）

- 不做规划/控制/VLA 推理；不输出动作。
- 不做训练/微调。
- 不做视频跟踪与跨帧实例 ID（可在 v0.2 加）。
- 不做通用 VLM 对话；VLM 只作为"grounding 工具"。
- 不做鉴权体系/TLS/公网暴露（loopback + 可选共享 token 即可）。
- 不内置相机驱动/仿真接口；输入是图像/标定/上下文。
- 不绑定任何 benchmark（RoboDojo/其它）。

---

## 2. 术语

| 术语 | 含义 |
|---|---|
| **Percept** | 一次感知结果：某工具在某相机/某提示下产出的观测集合 + 来源信息 |
| **Observation** | 单个实例：mask、box、score、2D 统计 |
| **Prompt** | 提示：text / box / point（可带正负标签） |
| **Calibration** | 相机内参 K、外参 `T_world_cam`、尺寸；可选深度 |
| **Tool** | 感知工具插件（segmenter/detector/depth/vlm/geometry/verifier） |
| **Pipeline** | 工具编排：如 `[sam3, multi_view_3d]`；顺序即数据流 |
| **Fusion** | 多视角/多提示结果合并成 3D 观测 |
| **Status** | 结果状态：ok / not_found / occluded / unavailable / timeout / invalid_request |
| **Evidence** | 带 provenance 的观测，可用于动作前后对比 |

---

## 3. 数据模型（冻结）

使用 **pydantic v2** 定义；字段名即 JSON 名（snake_case）。所有模型必须可 `model_dump_json()` / `model_validate_json()` 往返。

### 3.1 枚举

```python
Status            = "ok" | "not_found" | "occluded" | "unavailable" | "timeout" | "invalid_request"
PromptType        = "text" | "box" | "point"
MaskEncoding      = "rle" | "png" | "none"
ImageEncoding     = "png" | "jpeg" | "raw"
GeometryFrame     = "pixel" | "camera" | "world"
ToolKind          = "segmenter" | "detector" | "depth" | "vlm_grounding" | "geometry" | "verifier" | "fake"
```

### 3.2 Prompt

```python
class Prompt(BaseModel):
    type: PromptType
    text: str | None = None          # type=text 时必填；名词短语，非句子
    box: list[float] | None = None   # type=box 时必填；归一化 [cx, cy, w, h]，0..1
    point: list[int] | None = None   # type=point 时必填；像素 [u, v] = [col, row]
    label: bool = True               # 几何提示的正/负样本
    min_score: float = 0.2           # 该提示的过滤阈值
    top_k: int = 1                   # 该提示最多取几个实例
```

### 3.3 ImageRef / Calibration

```python
class ImageRef(BaseModel):
    encoding: ImageEncoding
    data: str                        # base64；raw 时为 uint8 RGB 字节
    shape: tuple[int, int, int] | None = None   # raw 必填 (H, W, 3)
    camera: str | None = None        # 可选，冗余标注

class DepthRef(BaseModel):
    encoding: Literal["raw_f32"] = "raw_f32"
    data: str                        # base64 float32，单位米
    shape: tuple[int, int]           # (H, W)

class Calibration(BaseModel):
    camera: str
    width: int
    height: int
    K: list[list[float]]             # 3x3 内参
    T_world_cam: list[list[float]]   # 4x4 camera->world
    depth: DepthRef | None = None
    id: str | None = None            # 标定版本/来源
```

### 3.4 Observation / Geometry / Provenance / Percept

```python
class Mask(BaseModel):
    encoding: MaskEncoding
    size: tuple[int, int] | None = None    # (H, W)
    counts: list[int] | None = None        # encoding=rle
    data: str | None = None                # encoding=png 时 base64 PNG

class Observation(BaseModel):
    object_id: str                   # 稳定 id：sha1(percept_id + index)
    score: float
    box_xyxy: list[float] | None     # 像素 [x0, y0, x1, y1]
    mask: Mask | None
    area_px: int | None
    centroid_uv: list[float] | None  # 像素 [u, v]
    label: str | None                # 产生它的概念（text prompt 或 detector label）
    attributes: dict[str, float | str | bool] = {}

class Geometry(BaseModel):
    frame: GeometryFrame = "world"
    centroid: list[float] | None = None      # [x, y, z] 米
    bbox3d_min: list[float] | None = None
    bbox3d_max: list[float] | None = None
    top_z: float | None = None
    yaw_rad: float | None = None
    uncertainty_m: float | None = None
    views_used: int | None = None

class Provenance(BaseModel):
    tool: str
    tool_version: str
    kind: ToolKind
    model: str | None = None
    model_sha256: str | None = None
    prompt: Prompt | None = None
    camera: str | None = None
    image_sha256: dict[str, str] = {}        # camera -> sha256
    calibration_id: str | None = None
    latency_ms: float
    created_at: datetime
    cache_hit: bool = False

class Percept(BaseModel):
    schema_version: Literal["percepta/v1"] = "percepta/v1"
    percept_id: str                  # uuid4
    status: Status
    message: str | None = None       # 失败原因（人可读）
    camera: str | None = None
    observations: list[Observation] = []
    geometry: Geometry | None = None
    provenance: Provenance
    extras: dict[str, Any] = {}
```

### 3.5 请求/响应

```python
class PerceiveOptions(BaseModel):
    tools: list[str] | None = None   # None = 默认 pipeline
    fuse: bool = True                # 多视角 -> 3D
    masks: MaskEncoding = "rle"
    min_score: float = 0.2
    top_k: int = 1
    timeout_ms: int = 120_000
    cache: bool = True

class PerceiveRequest(BaseModel):
    request_id: str | None = None
    images: dict[str, ImageRef]              # camera -> image（至少 1 个）
    calibration: dict[str, Calibration] = {}
    prompts: list[Prompt]
    context: dict[str, Any] = {}             # instruction/target/history 等，工具只读
    options: PerceiveOptions = PerceiveOptions()

class PerceiveResponse(BaseModel):
    request_id: str
    percepts: list[Percept]          # 每 camera×prompt×tool 一条
    fused: Percept | None = None     # options.fuse 且几何可用时
    timings_ms: dict[str, float] = {}
    warnings: list[str] = []
```

### 3.6 JSON 示例（单相机、text 提示、RLE）

```json
{
  "request_id": "req-1",
  "images": {"cam_head": {"encoding": "png", "data": "<b64>"}},
  "calibration": {"cam_head": {"camera": "cam_head", "width": 640, "height": 480,
    "K": [[288.13,0,320],[0,288.13,240],[0,0,1]],
    "T_world_cam": [[1,0,0,0],[0,1,0,-0.41],[0,0,1,1.308],[0,0,0,1]],
    "id": "station-arx-x5"}},
  "prompts": [{"type": "text", "text": "bottle", "min_score": 0.2, "top_k": 1}],
  "options": {"masks": "rle", "fuse": false}
}
```

---

## 4. 坐标与几何约定（冻结，最容易实现错）

1. **像素坐标**：原点左上；`u` 向右（列），`v` 向下（行）；`point=[u,v]`；`box=[x0,y0,x1,y1]` 浮点像素，`x1/y1` 为开区间外沿。
2. **相机坐标**：OpenCV 约定（x 右、y 下、z 向前）。
3. **世界坐标**：右手系，米；由 `T_world_cam` 唯一确定（**camera→world**，即 `p_world = T_world_cam @ p_cam_h`）。
4. **内参**：`K` 3x3，像素单位；不做畸变（v0.1 假定已校正）。
5. **深度**：`DepthRef` 为 float32 米，与 RGB 同分辨率；可选。没有深度时几何只能靠轮廓/平面假设。
6. **尺寸/分辨率校验**：图像 `(H,W)` 必须与 `Calibration.height/width` 一致，否则 `invalid_request`。
7. **mask 尺寸**：与原图同 `(H,W)`；`area_px = sum(mask)`；`centroid_uv` 为 mask 像素均值。

RLE 规范见附录 B。

---

## 5. 工具插件协议（冻结）

### 5.1 能力声明 + 生命周期

```python
class ToolCapabilities(BaseModel):
    kind: ToolKind
    prompts: set[PromptType]
    masks: bool
    boxes: bool
    points: bool
    depth: bool
    multi_view: bool
    geometry: bool

class ToolContext(BaseModel):
    request_id: str
    images: dict[str, np.ndarray]        # 已解码 RGB uint8 (H,W,3)
    calibration: dict[str, Calibration]
    context: dict[str, Any]
    options: PerceiveOptions
    logger: Any
    cache: Any

class PerceptionTool(Protocol):
    name: str
    version: str
    api_version: Literal["percepta/tool/v1"] = "percepta/tool/v1"
    def capabilities(self) -> ToolCapabilities: ...
    def load(self, config: dict[str, Any]) -> None: ...     # 进程内只调一次
    def perceive(self, ctx: ToolContext, prompts: list[Prompt]) -> list[Percept]: ...
    def close(self) -> None: ...
```

### 5.2 注册

- 通过 Python entry point 组 `percepta.tools` 注册；包内工具用 `percepta.tools.<name>` 导入。
- 配置文件按名字启用/配置；未知工具名 → `invalid_request`。
- 工具必须声明 `capabilities()`；不支持的 prompt 类型返回 `unsupported_prompt`（见附录 A）。

### 5.3 编排（pipeline）

- 默认 pipeline：`[<默认 segmenter>, multi_view_3d]`（`multi_view_3d` 是内置 geometry 工具）。
- 工具顺序即数据流：前一个工具的 `Percept` 作为后一个的输入（通过 `ctx.extras["upstream"]` 传递）。
- v0.1 只要求**串行**；并行多相机由服务层调度（每相机独立调用工具，工具内部须线程安全或由服务层加锁）。

### 5.4 必做工具（v0.1）

| 工具 | 类型 | 要求 |
|---|---|---|
| `fake` | fake | 纯 numpy 合成：按"颜色阈值 + 连通域"输出 mask/box，用于 CI；无外部依赖 |
| `sam3` | segmenter | 官方 `sam3` 包；支持 text / box；输出 mask/box/score；加载一次、模型 sha256 入 provenance |
| `multi_view_3d` | geometry | 多视角 mask + 标定 → Geometry（见 §9） |

可选（v0.2）：`yoloe`/`omdet`（detector，box 提示来源）、`depth_anything`（单目深度）、`vlm_grounding`（VLM 出框）。

---

## 6. HTTP 服务（冻结）

- 框架：FastAPI + uvicorn；pydantic 模型即 §3。
- 绑定默认 `127.0.0.1:8790`；**不提供 0.0.0.0 的默认值**。
- 端点：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | `{"ok":true,"uptime_s":…,"tools":[{"name","version","status"}]}` |
| GET | `/v1/info` | schema 版本、包版本、工具能力、GPU/设备、模型 sha |
| GET | `/v1/tools` | 工具列表 + capabilities |
| POST | `/v1/perceive` | 入参 `PerceiveRequest`，出参 `PerceiveResponse` |
| GET | `/metrics` | 可选 Prometheus 文本 |

- 限制：请求体 ≤ `max_body_bytes`（默认 32MiB）；单请求超时 `timeout_ms`；进程内队列 `max_queue`（默认 8），满则 **503 busy**。
- 错误映射（HTTP）：400 invalid_request / 401 unauthorized / 413 payload_too_large / 503 busy / 504 timeout / 500 internal；**`not_found` 是 Percept 的 status，不是 HTTP 错误**。
- 可选鉴权：`Authorization: Bearer $PERCEPTA_TOKEN`（未设 token 则不校验）。
- 响应必须可 JSON 序列化；不允许 NaN/Inf（几何缺失用 null）。

---

## 7. Python SDK 与 CLI（冻结）

```python
class PerceptaClient:
    def __init__(self, endpoint="http://127.0.0.1:8790", token=None, timeout=180.0): ...
    def health(self) -> dict: ...
    def info(self) -> dict: ...
    def tools(self) -> list[dict]: ...
    def perceive(self, images: dict[str, np.ndarray | str], prompts: list[Prompt],
                 calibration: dict[str, Calibration] | None = None,
                 context: dict | None = None, options: PerceiveOptions | None = None
                 ) -> PerceiveResponse: ...
```

- `images` 值可为 numpy RGB 或本地路径；客户端负责 PNG 编码与 base64。
- 客户端对 503/5xx 做有界重试（默认 2 次，指数退避）；`not_found` 不重试。
- 客户端提供 `perceive_local()`：同进程直接调 registry（不经过 HTTP），用于单机低延迟。

CLI：

```
percepta serve --config config.yaml
percepta perceive --image cam_head=img.png --prompt text:bottle [--calib calib.json] [--json]
percepta tools list
percepta probe [--config config.yaml]
```

---

## 8. 提示生成与多实例消歧

1. `percepta.prompts.from_instruction(instruction, target=None)`：
   - 若给了 `target` → 直接用；
   - 若没有 → 从指令里抽取**名词短语**（规则：取 "pick up/ grasp/ move the X" 的 X；失败则返回空并发 `not_found` 警告）；
   - **可选** LLM hook（接口预留 `prompt_llm`，v0.1 不实现）。
2. 一次请求可带多个 `Prompt`；每个 prompt 独立出 `Percept`。
3. 多实例消歧（默认策略，可配置）：
   - 先按 `score` 排序；
   - 若给了 calib：保留"在桌面高度带 ± `table_band_m`（默认 0.15）内"的实例；
   - 若给了 `context["ee_position_world"]`：优先最近者；
   - 仍多解 → 全部返回，`warnings` 标记 `ambiguous_instances`。
4. text prompt 必须是**名词短语**（不含动词/关系）；实现可在发送前做轻量校验，失败给 `invalid_request` 警告但不阻断。

---

## 9. 多视角 3D（`multi_view_3d`）

输入：同一次请求的多相机 `Percept`（mask + box + score）+ `Calibration`。

算法（v0.1，按优先级）：

1. **有深度**：取 mask 内深度像素 → 反投影到相机 → 世界点云 → 中位数中心、分位数包围盒、顶部 z、PCA/最小外接矩形得 yaw。
2. **无深度、多视角**：轮廓几何（silhouette/prism）：
   - 对每个相机，把 mask 的轮廓射线打到"桌面平面"（`context["table_plane"]: [a,b,c,d]` 或默认 z=table_z）上；
   - 在候选高度 h 上，把"以该轮廓为投影的立体"重新投影到其它相机，与其它 mask 做 IoU/F1 匹配；
   - 取匹配最好且高度最小的解 → 中心/顶部 z/半径；`views_used` 记录参与视角数；`uncertainty_m` 由视角数与高度区间估计。
3. **单视角无深度**：只返回 2D + 粗略高度（若 `table_plane` 可用），`geometry.uncertainty_m` ≥ 0.03；否则 `Geometry` 为 null 且 `warnings += ["geometry_unavailable"]`。

约束：

- 无 `calibration` 时**禁止**输出 `Geometry`（不得用像素猜米）。
- 所有几何结果必须带 `frame="world"` 与 `uncertainty_m`。
- 至少 2 视角才允许 `views_used>=2`；单视角必须显式标注。

---

## 10. 验证与证据（`verify`）

```python
class VerificationResult(BaseModel):
    ok: bool
    checks: dict[str, bool]          # followed / lifted / stable / visible
    details: dict[str, float]        # mask_iou, centroid_shift_px, lift_m, ...
    status: Status
```

`percepta.verification.compare(before: Percept, after: Percept, action: dict, options) -> VerificationResult`：

- `followed`：after 的 mask 质心位移与 `action["ee_delta_px"]` 方向/幅度一致（阈值可配）；
- `lifted`：`after.geometry.centroid.z - before.geometry.centroid.z >= required_lift_m`（默认 0.10）；
- `stable`：连续两次 after 的 mask IoU ≥ 阈值（可选）；
- `visible`：after 的 `status == ok` 且 mask 面积 ≥ `min_area_px`。
- 任一必要检查失败 → `ok=false`；不得把 `not_found` 当作通过。

---

## 11. 服务化、配置、并发、可观测

### 11.1 配置（YAML，示例）

```yaml
server:
  host: 127.0.0.1
  port: 8790
  max_body_bytes: 33554432
  max_queue: 8
  request_timeout_ms: 120000
  token_env: PERCEPTA_TOKEN      # 可选
device:
  cuda_visible_devices: "0"
pipeline:
  default: ["sam3", "multi_view_3d"]
tools:
  sam3:
    enabled: true
    checkpoint: /models/sam3.pt
    resolution: 1008
    min_score: 0.2
  fake:
    enabled: false
cache:
  enabled: true
  max_entries: 256
  ttl_s: 300
logging:
  level: info
  json: true
metrics:
  enabled: false
```

### 11.2 并发

- 每工具一把进程内锁（模型推理串行）；服务层 `BoundedSemaphore` 控制排队；超时/满 → 503/504。
- 多 GPU：一个进程绑一张卡（`CUDA_VISIBLE_DEVICES`）；多实例多端口。

### 11.3 可观测

- 结构化日志（JSON）：`request_id, tool, camera, status, latency_ms, cache_hit`。
- 指标（可选）：`percepta_requests_total{tool,status}`、`percepta_latency_ms{tool}`、`percepta_inflight`、`percepta_cache_hits_total`。
- `/healthz` 必须反映工具加载状态（`ok|loading|error`）。

### 11.4 缓存

- key：`sha256(image) + prompt + tool + options(相关子集)`；LRU + TTL；命中时 `provenance.cache_hit=true`（不改变结果内容）。

---

## 12. 安全与许可

- 默认只绑 loopback；跨机用 SSH 隧道；文档给出 `ssh -N -L` 与 systemd 常驻写法。
- 可选 bearer token；不实现 TLS。
- 请求体大小、超时、队列上限必须生效。
- 不接收文件路径（只收内联图像）；不做 pickle/任意反序列化。
- 仓库许可建议 **Apache-2.0**；**不得 vendor** 任何模型权重或违反上游许可的代码。
  - SAM3 权重：SAM License（gated），运行时由用户提供路径。
  - Ultralytics 系（YOLOE/YOLO-World）：AGPL-3.0，作为**可选工具**，默认不启用、不随仓库分发。
  - GroundingDINO/OWLv2/OmDet-Turbo/SAM2：Apache-2.0（以各自仓库为准）。

---

## 13. 仓库结构与依赖

```
percepta/
├── pyproject.toml            # hatchling; extras: [service], [sam3], [dev]
├── README.md
├── AGENTS.md                 # 从本文档 §0/§18 派生的协作规约
├── configs/example.yaml
├── docs/spec.md              # 本文档
├── src/percepta/
│   ├── __init__.py           # 公共 API 导出
│   ├── schema.py             # §3 全部模型
│   ├── masks.py              # RLE 编解码、IoU、面积/质心
│   ├── geometry.py           # project/unproject、平面、包围盒、PCA/yaw
│   ├── fusion.py             # §9 multi_view_3d
│   ├── verification.py       # §10
│   ├── prompts.py            # §8
│   ├── registry.py           # entry points 加载、能力校验
│   ├── config.py             # pydantic-settings + YAML
│   ├── cache.py
│   ├── service/{app.py,routes.py,metrics.py,logging.py}
│   ├── client.py             # PerceptaClient + perceive_local
│   ├── cli.py
│   └── tools/{base.py,fake.py,sam3.py,multi_view_3d.py}
├── src/percepta/backends/     # 模型跑在哪：inprocess / subprocess / http（v0.1.3 新增）
│   └── {base,protocol,inprocess,subprocess_backend,http_backend,worker,reference}.py
├── tests/
│   ├── unit/                 # schema/masks/geometry/prompts/verification/backends/weights
│   ├── contract/             # HTTP 契约（fake 工具，无 GPU）
│   └── gpu/                  # pytest -m gpu，默认跳过
└── systemd/perceptad.service
```

依赖（v0.1，最小集）：`pydantic>=2`、`numpy`、`pillow`、`pyyaml`、`fastapi`、`uvicorn`、`pytest`、`httpx`（测试）；`sam3` 走 extras（含 torch）。**不得**把 torch 设为默认依赖。

依赖由 **uv** 管理（`uv.lock` + `.python-version`，见 D22）。**本仓库不维护任何模型框架的依赖清单**：
模型依赖由部署方装进独立虚拟环境，通过 `tools.<name>.backend` / `python` 指过去（见 `docs/model-isolation.md`）。
库只约束模型**权重的路径与完整性**（`checkpoint` / `checkpoint_sha256` / `verify_weights`，D27）。

---

## 14. 验收场景（必须自动化，除标注 gpu）

| 编号 | 场景 | 判据 |
|---|---|---|
| S1 | fake 工具端到端（进程内 + HTTP） | 合成图（中心红方块）→ 1 个 observation，area/box/centroid 符合误差 <2px |
| S2 | RLE 往返 | 随机 mask encode→decode 完全相等；空 mask/全 1 mask 正确 |
| S3 | 投影/反投影往返 | 合成 K/T，world→pixel→(depth)world 误差 <1e-6（无深度用平面假设误差 <1e-3） |
| S4 | HTTP 契约 | 200 正常；400 坏字段；401 无 token；413 超限；503 队列满；not_found 仍在 200 的 status 里 |
| S5 | 多视角融合 | 两个合成视角（已知 K/T + 正方体）→ centroid 误差 <2cm，views_used=2，uncertainty 非空 |
| S6 | 验证 | 合成 before/after（物体上移 0.12m）→ lifted=true；not_found 时 ok=false |
| S7 | 缓存 | 同图同 prompt 第二次命中，cache_hit=true，延迟下降 |
| S8 | 能力校验 | 对只支持 box 的工具发 text prompt → status=invalid_request/unsupported_prompt |
| S9 | SAM3 冒烟（手动，gpu） | 真实商品图 + text=bottle → found，mask shape=(H,W)，延迟记录到台账 |

`pytest -m "not gpu"` 必须在无 GPU、无网络、无权重的情况下全绿。

---

## 15. 里程碑（可并行）

| 里程碑 | 内容 | DoD |
|---|---|---|
| **M0 骨架** | pyproject、schema.py、masks.py、registry、fake 工具、pytest 骨架 | S2 绿；`pip install -e .` 可用 |
| **M1 契约** | service（FastAPI）+ client + CLI + 错误映射 | S1/S4 绿；`percepta serve` 可起 |
| **M2 SAM3** | `tools/sam3.py`（text/box）、provenance、加载/释放、warmup | S9 手动通过；无 GPU 环境不导入 |
| **M3 几何** | geometry.py + fusion.py（深度 & 轮廓） | S3/S5 绿 |
| **M4 验证/缓存** | verification.py、cache.py、指标、日志 | S6/S7 绿 |
| **M5 加固** | 队列/超时/大小限制、Docker、systemd、文档、性能基线 | 全部验收绿；README 有部署示例 |

建议并行拆分（文件所有权互斥）：
- A：`schema.py` + `masks.py` + 单元测试（M0/M1 依赖）
- B：`service/*` + `client.py` + `cli.py` + HTTP 契约测试
- C：`tools/sam3.py` + `tools/fake.py` + `registry.py`
- D：`geometry.py` + `fusion.py` + `verification.py` + 几何测试
- E：`config.py` + `cache.py` + Docker/systemd + 文档

---

## 16. Agent 协作规约

1. **接口优先**：任何跨模块改动先改 `schema.py`（及本文档），再改实现；不得在工具里私自加顶层字段。
2. **可替换**：所有外部模型调用必须藏在 tool 后面；CI 用 fake 工具，GPU 测试用 `-m gpu` 标记。
3. **不新增依赖**：默认依赖清单见 §13；需要新依赖先在 §19 记录并征得同意。
4. **错误语义**：不许把异常吞成 `not_found`；`unavailable`（服务/模型不可用）和 `not_found`（模型说没有）必须分开。
5. **坐标**：任何几何函数必须写明输入/输出 frame，并在 docstring 里写公式。
6. **测试**：新功能必须带单测；修复 bug 必须先有失败用例。
7. **提交**：Conventional Commits；一个 PR 一个里程碑子项；CI 必须跑 `ruff check`、`pytest -m "not gpu"`。
8. **文档**：公共 API 改动同步 `docs/spec.md` 与 `README.md`。

---

## 17. 性能目标（v0.1，单卡 3090/H20 量级，仅参考）

| 项 | 目标 |
|---|---|
| fake 工具端到端（HTTP，640×480×3） | < 50ms |
| RLE 编解码 640×480 | < 5ms |
| 多视角融合（2 视角，无深度） | < 100ms（不含模型） |
| SAM3 text 分割（1008 分辨率） | p50 < 1.5s，p90 < 3s（首帧另计 warmup） |
| 服务空载内存 | < 1GB（不含模型） |

---

## 18. 变更记录

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 | 2026-11-04 | 首版：schema、坐标、工具协议、HTTP、几何、验证、里程碑 |
| v0.1.1 | 2026-11-04 | 新增 §19.1 实现取向（D1–D12）：`unsupported_prompt` 映射、`ToolContext.extras`、棱柱高度搜索离散化、单次工具调用、`pipeline.py` 等 |
| v0.1.2 | 2026-11-04 | 补充 D13–D21：验证结果状态优先级、geometry 工具与 `fused` 去重、visual-hull 横截面、`tool_unavailable` 的 HTTP 200 取向（偏离附录 A，待复核）、prompt 无关工具、健康检查隐藏未启用工具、缓存 key 的 options 子集 |
| v0.1.3 | 2026-11-05 | 补充 D22–D26：改用 uv 管理依赖、修正 torch/sam3/numpy 版本约束、**模型隔离后端抽象**（`inprocess`/`subprocess`/`http`，不用容器）、二进制分帧协议、包导入开销约束。新增 `docs/model-isolation.md` 与 `src/percepta/backends/` |
| v0.1.4 | 2026-11-05 | 补充 D27：明确库与模型依赖/资产的边界 —— **不维护模型依赖清单**（删除 `requirements/`、`docker/`），但**强制约束权重资产**（`checkpoint` / `checkpoint_sha256` / `verify_weights`，新增 `percepta models check`）。同时删除 §13 中未交付的 `tests/golden/` 与 `examples/` |
| v0.2.0 | 2026-11-05 | 补充 D28–D32：**真实 SAM3 端到端验证通过**（S9 关闭，实测记录见 `docs/model-isolation.md` §10）、启动预热、`wait_until_ready` 等待热态、introspection 统一读类属性、空 `cuda_visible_devices` 语义修正。包版本升至 0.2.0，新增 `CHANGELOG.md` |

---

## 19. 待决问题（实现时可先按保守方案）

1. 是否支持 point 提示？官方 `Sam3Processor` 未暴露，需要自定义 `forward_grounding` 或引入 SAM2；v0.1 建议：schema 保留 `point`，SAM3 工具返回 `unsupported_prompt`，由 `fake`/未来工具实现。
2. 多相机并行调用工具是否安全？v0.1 由服务层串行化，文档记录。
3. 是否内置单目深度工具（如 Depth Anything）？列为 v0.2 可选。
4. 是否支持视频/跟踪？v0.2。
5. 与具体 Agent Policy（如 PeerAgent）的适配层是否进本仓库？建议**不进**，在策略侧写 adapter。
6. 缓存 key 是否包含 `context`？v0.1 不包含（避免调试困惑），在 §11.4 注明。

### 19.1 实现取向（v0.1 已决）

下列条目是 §19 的落地裁定：实现时按此执行，不得各自发明。完整理由见
[`docs/decisions.md`](docs/decisions.md)（同编号）。

| # | 歧义 | v0.1 取向 |
|---|---|---|
| D1 | `unsupported_prompt` 不在 §3.1 的 `Status` 枚举内，但附录 A 与 S8 都用到 | **保持冻结枚举不变**：该 Percept 取 `status="invalid_request"`、`message="unsupported_prompt"`、`extras["code"]="unsupported_prompt"`，整个请求仍返回 HTTP 200（一条 prompt 不支持不得拖垮其余 prompt） |
| D2 | §5.3 用 `ctx.extras["upstream"]` 传递上游 Percept，但 §5.1 的 `ToolContext` 无 `extras` | 给 `ToolContext` **增加可选字段** `extras: dict[str, Any] = {}`（§0.2 允许"加可选字段"），按 `extras["upstream"]` 传递 |
| D3 | `ToolContext` 声明为 pydantic 模型却含 `ndarray`/`logger`/`cache` | 保留 `BaseModel`，设 `arbitrary_types_allowed=True`，`logger`/`cache` 类型为 `Any`（鸭子类型协作者） |
| D4 | §9 算法 2 未定义轮廓棱柱的高度离散、匹配粒度与失败分支 | 高度自 `table_z` 扫描至 `table_z+prism_max_height_m`（默认 0.5m），步长 `prism_step_m`（默认 0.01m）；取**达标（mean IoU ≥ `prism_iou_threshold`，默认 0.5）的最小高度**；若无达标则取全局 argmax 并加 `geometry_low_confidence` 警告；`uncertainty_m = prism_step_m*(1+1/views_used)` 加 IoU 亏缺项 |
| D5 | §3.4 称 `object_id` 为"稳定 id"，但 `percept_id` 是 uuid4 | 严格实现 `sha1(f"{percept_id}:{index}")`：仅在同一 Percept 及其序列化生命周期内稳定；跨请求跟踪属 v0.2（§1.2） |
| D6 | §5.3 说"每相机独立调用工具"，但 `ToolContext.images` 是全相机字典且 `multi_view_3d` 需要全部视角 | **每请求每工具调用一次**（带全部相机），工具内部遍历相机产出一条 Percept/相机；并发由 per-tool 锁串行化（§11.2） |
| D7 | §13 文件布局提到 `pydantic-settings`，但 §13 依赖清单没有 | 用 pydantic + pyyaml，不新增依赖；YAML 键 `logging.json` 暴露为属性 `json_output`（`json` 与 `BaseModel.json` 冲突），以别名 `json` 校验 |
| D8 | SDK 客户端用哪个 HTTP 库 | 用 stdlib `urllib`（`httpx` 在 §13 已标注"(测试)"，仅进 `dev` extra） |
| D9 | 代码默认 pipeline 与"CI 无 GPU"冲突 | 代码默认严格按 §5.3 取 `["sam3","multi_view_3d"]`；`configs/ci.yaml`、`configs/example.yaml` 用 `fake`；无权重时 `sam3` 在 `/healthz` 报 `error` 而非崩溃 |
| D10 | RLE 是否用 COCO 压缩字符串 | 附录 B 的交替 run-length 整数列表，不压缩；`sum(counts)==H*W` 由 `Mask` 校验；首像素为 1 时首段为 0 |
| D11 | 两个空 mask 的 IoU | 定义为 `0.0`（`0/0` 无意义），避免 `verification.stable` 把两次漏检当成一致 |
| D12 | §13 布局无编排模块，但 HTTP 与 `perceive_local()` 需同一语义 | 新增 `src/percepta/pipeline.py`（`PipelineRunner`），两处共用，避免编排逻辑重复 |
| D13 | §10 未定义 `VerificationResult.status` 的回退优先级 | `ok` → `not_found`（任一侧）→ `invalid_request`（action/options 畸形）→ 透传 `after.status`；实质性检查失败但 `after.status="ok"` 时 `status` 保持 `ok`，结论落在 `ok`/`checks` |
| D14 | §10 的 `stable` 在缺第三份观测时如何判定 | 仅在 `options["stable_with"]` 给出时计算；`required` 默认取"所有可评估项"，支持收窄；`required` 收窄也不能让 `not_found` 通过；`required=["stable"]` 但无 `stable_with` → `ok=False` + `invalid_request` |
| D15 | §5.3 的 geometry 工具与 §9 的 `fuse` 会重复计算几何 | geometry 类工具的 ok Percept 直接作为 `fused`，**不进入** `percepts`；仅当 pipeline 中无 geometry 类工具时才调用 `fusion.fuse`；该工具返回空 → `fused=null` + `geometry_unavailable` |
| D16 | §9 算法 2 的横截面取法未定义 | 高度按 D4 决定；质心/包围盒/朝向改用"各视角在**中高**处的锥截面之交"（visual hull），以抵消单视角"阴影"偏差；`bbox3d` 是**上界**而非紧贴框；仰角扫描在首次达标处提前结束 |
| D17 | 附录 A 的 `tool_unavailable → 503` 与"Percept 承载状态"冲突 | **保留 HTTP 200**，Percept 取 `status="unavailable"` + `tool_unavailable` 警告；`/healthz` 报 `error`。理由：503 的响应体无法是 `PerceiveResponse`，且多工具 pipeline 中单工具不可用不应吞掉其它工具的结果。**此项为对附录 A HTTP 列的显式偏离，留待复核** |
| D18 | `capabilities().prompts == ∅` 的语义 | 视为"**prompt 无关**"（geometry 类工具消费上游 Percept，忽略 prompts），不产生 `unsupported_prompt`；声明了非空集合的工具仍拒绝其余类型（S8 覆盖） |
| D19 | §11.3 的 `ok|loading|error` 词汇表没有"未启用" | `config.tools.<name>.enabled: false` 的工具**不出现在** `/healthz`、`/v1/tools`、`/v1/info` 中（缺席即"不属于本次部署"），绝不伪装成 `ok`；`lazy`（导入成功、按需加载）仍映射为 `ok` |
| D20 | §11.4 缓存 key 的"options 相关子集"具体指什么 | 仅纳入**影响结果**的 `masks`/`min_score`/`top_k`；`tools`/`fuse`/`timeout_ms`/`cache` 与 `context` 不纳入；缓存只作用于**首个无上游**的阶段（后续阶段依赖上游 Percept，key 无法覆盖） |
| D21 | 未知工具名 vs 已配置但加载失败 | 未知工具名 = 调用方错误 → HTTP 400 `invalid_request`；已配置却加载失败/加载后不可用 → HTTP 200 + Percept `status="unavailable"`（见 D17），永不降级为 `not_found` |
| D22 | 依赖管理方式 | 改用 **uv** + 提交 `uv.lock`（必须用**官方源**生成，镜像会污染可移植性）+ `.python-version` 钉 3.10；dev 工具进 PEP 735 `dependency-groups` 默认安装 |
| D23 | 原有版本约束有误 | `torch>=2.1`→`>=2.7,<3`；`sam3`→`==0.1.4`；`numpy` 补 sam3 声明的 `<2.4` 上限 |
| D24 | 不同模型的依赖互斥（如两个 torch 版本）如何共存 | 不用容器：把"模型跑在哪"做成配置项 `tools.<name>.backend`，抽象为 `inprocess`/`subprocess`/`http` 三种后端；容器仅留作 CUDA/系统库级冲突的最后手段。详见 `docs/model-isolation.md` |
| D25 | 跨进程/跨机传输格式 | 二进制分帧协议（magic + version + 长度前缀），仅 prompt/status 用 JSON；640×480 帧裸字节 921KB 对比 JSON 约 15MB。worker 只许写 stderr，stdout 归协议。worker 死亡 → `unavailable`，永不 `not_found` |
| D26 | 包 `__init__` 的导入开销 | 仅 `schema` 急加载，其余公共名走惰性 `__getattr__`；否则瘦环境里的 worker 会因缺 pyyaml/pillow 而启动失败。worker 环境只需 `numpy`+`pydantic` |
| D27 | 库与模型依赖/资产的边界 | **库不管理模型框架依赖**（不提供 requirements 清单，删除了 `requirements/`、`docker/`）；但**必须约束模型资产**：`checkpoint` 路径 + 可选 `checkpoint_sha256` + `verify_weights`（`if-declared`/`always`/`never`），并提供 `percepta models check` 启动前自检。理由：错误的权重会静默加载并输出错结果。YAML 纯数字哈希被解析为整数时**报错而非静默跳过** |

---

## 附录 A：错误码

| code | 场景 | HTTP |
|---|---|---|
| `invalid_request` | 字段缺失/类型错/尺寸不匹配 | 400 |
| `unsupported_prompt` | 工具不支持该 prompt 类型 | 400（Percept.status） |
| `unauthorized` | token 缺失/错误 | 401 |
| `payload_too_large` | 超过 `max_body_bytes` | 413 |
| `busy` | 队列满 | 503 |
| `timeout` | 超过 `timeout_ms` | 504 |
| `tool_unavailable` | 工具未加载/模型缺失 | 503（Percept.status=`unavailable`） |
| `internal` | 未预期异常 | 500 |

## 附录 B：RLE 规范

- 行优先展平 `(H,W)` → 一维 `uint8`（0/1）。
- 输出为**交替 run 长度**，从 0 段开始；若首像素为 1，则第一个计数为 0。
- `sum(counts) == H*W`；解码：从 0 开始交替填充，奇数段填 1。
- 参考实现（伪码）：
  ```
  change = flatnonzero(diff(flat)) + 1
  starts = [0] + change
  lengths = diff(starts + [flat.size])
  counts = ([0] if flat[0] else []) + lengths
  ```

## 附录 C：与上游模型的边界

- SAM3：官方 `sam3` 包；`build_sam3_image_model(device, checkpoint_path, load_from_HF=False, enable_inst_interactivity=True)`；`Sam3Processor(resolution=1008)`；`set_image` 一次、`set_text_prompt`/`add_geometric_prompt(box=[cx,cy,w,h] normalized)` 复用 state；输出 `masks (N,1,H,W) bool`（原图尺寸）、`boxes xyxy`、`scores`。
- 权重 gated（HuggingFace `facebook/sam3`）；路径由配置注入，仓库不含权重。
- 若上游 API 变更，只允许改 `tools/sam3.py`，不得泄漏到 schema。 

## 附录 D：最小使用示例（伪代码）

```python
from percepta import Prompt, PerceptaClient

client = PerceptaClient("http://127.0.0.1:8790")
resp = client.perceive(
    images={"cam_head": "head.png"},
    prompts=[Prompt(type="text", text="bottle", min_score=0.2)],
    calibration={"cam_head": calib},
)
for p in resp.percepts:
    for o in p.observations:
        print(p.camera, o.score, o.area_px, o.mask.encoding)
print(resp.fused.geometry.centroid if resp.fused else None)
```

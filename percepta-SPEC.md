# percepta — 功能规格与开发说明书（v0.3.0-doc.1，2026-11-12）

> 本文档是给**实现者（人或 Agent）**看的规格书，不是用户手册。目标是把"一套可插拔的机器人感知层"定义到**接口无歧义、验收可自动判定**的程度。实现时不要自行发明字段、端点或语义；确需变更，先改本文档并升版本。

> **实现对齐时间：2026-11-12（UTC+08:00）；代码基线：`534066a`。** 包版本仍为
> `0.3.0`，schema 仍为 `percepta/v1`；本次只修订说明，不新增接口或算法。根目录本文是主本，
> `docs/spec.md` 是内容相同的镜像。修订前原文、修改原因及逐项对照见
> `docs/documentation-changes.md`；历史决策保留，已被替代的条目另加说明。
>
> 原始 M0–M5 已完成。SAM3 及四个双目／三目模型均已部署，后四者属于 `Unreleased`；
> 后来的扩展方案包含更多设计任务，不代表当前全部已实现。当前调用方式见 `README.md`，
> 部署证据与限制见 `docs/model-isolation.md`，决策沿革见 `docs/decisions.md`。

---

## 0. 怎么用这份文档（给 Agent）

1. 先读 §1–§5（目标、数据模型、坐标约定），再读你要负责的模块与 §14 验收、§16 协作规约。
2. **§4/§5/§6/§7 是冻结接口**：字段名、枚举值、JSON 形状、HTTP 路径不得改动；新增只能"加可选字段"，且需在 §18 记录。
3. 任何"模型相关"的实现都必须能被 **fake 工具**替换，保证 CI 无 GPU 可跑。
4. 完成定义（DoD）：代码 + 单测 + 验收场景通过 + 文档更新 + 不新增未声明的第三方依赖。
5. 有歧义就在 §19 记录问题并选一个保守实现，不要阻塞。

---

## 1. 目标与非目标

### 1.1 目标

`percepta` 是**感知层（observation layer）**：把多个感知模型（开放词表分割、检测、深度、VLM 定位、几何）统一成**同一种可序列化的观测对象**，让 Agent Policy（代码策略/VLA/工具链）用一致的方式消费，并支持：

- 一张或多张 RGB（多相机）+ 可选深度；
- schema 表达文本 / 框 / 点三种提示；具体支持范围由工具声明，当前 SAM3 只支持文本 / 框；
- 输出实例 mask、框、分数、2D 统计；
- 用标定做**多视角几何 → 3D 位置/尺寸/朝向**；
- 动作前后观测对比（验证/证据）；
- 工具插件化：SAM3 只是默认实现，可替换/叠加（检测器、深度、VLM grounding）。
- 既可进程内调用（Python SDK），也可作为 loopback HTTP 服务（多机部署）。

### 1.2 非目标（当前范围）

- 不做规划/控制/VLA 推理；不输出动作。
- 不做训练/微调。
- 不做视频跟踪与跨帧实例 ID（保留为未来扩展）。
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
    object_id: str                   # sha1(f"{percept_id}:{index}")；仅本 Percept 生命周期内稳定
    score: float
    box_xyxy: list[float] | None = None  # 像素 [x0, y0, x1, y1]
    mask: Mask | None = None
    area_px: int | None = None
    centroid_uv: list[float] | None = None  # 像素 [u, v]
    label: str | None = None             # text prompt 或 detector label
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
    lateral_uncertainty_m: float | None = None  # D35：横向边缘误差估计
    height_uncertainty_m: float | None = None   # D35：顶部高度误差估计
    view_disagreement_m: float | None = None    # D35：视角拟合分歧
    method: str | None = None                  # prism / hang / single_view / stereo_* / triview_* 等

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
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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
    fuse: bool = True                # 无显式 geometry 阶段时是否补充调用 fusion.fuse，见 §5.3
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
    fused: Percept | None = None     # 显式 geometry 输出，或 options.fuse 启用的补充融合结果
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
4. **内参**：`K` 3x3，像素单位；输入不含镜头畸变参数，假定已去畸变。双目工具仍须按 K/T 做极线校正。
5. **深度**：`DepthRef` 为 float32 米，与 RGB 同分辨率；可选。`multi_view_3d` 无深度时使用轮廓/平面假设；`stereo_3d` / `triview_3d` 可从标定 RGB 估计深度。
6. **尺寸/分辨率校验**：图像 `(H,W)` 必须与 `Calibration.height/width` 一致，否则 `invalid_request`。
7. **mask 尺寸**：与原图同 `(H,W)`；`area_px = sum(mask)`；`centroid_uv` 为 mask 像素均值。

RLE 规范见附录 B。

---

## 5. 工具插件协议（冻结）

### 5.1 能力声明 + 生命周期

```python
class ToolCapabilities(BaseModel):
    kind: ToolKind
    prompts: set[PromptType] = set()  # 空集合表示 prompt 无关（D18）
    masks: bool = False
    boxes: bool = False
    points: bool = False
    depth: bool = False
    multi_view: bool = False
    geometry: bool = False

class ToolContext(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, arbitrary_types_allowed=True)
    request_id: str
    images: dict[str, np.ndarray]        # 已解码 RGB uint8 (H,W,3)
    calibration: dict[str, Calibration] = {}
    context: dict[str, Any] = {}
    options: PerceiveOptions = PerceiveOptions()
    logger: Any = None
    cache: Any = None
    extras: dict[str, Any] = {}          # D2：pipeline 状态，upstream 为上游 Percept 列表

class PerceptionTool(abc.ABC):
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
- 每请求每工具调用一次，带全部相机；工具内部遍历或联合处理视角，推理由 per-tool 锁串行化。
- `kind="geometry"` 阶段中的最后一条 `status="ok"` Percept 直接成为 `response.fused`，该条不再进入 `percepts`，也不二次调用 `fusion.fuse()`。
- 配置解析会移除禁用工具，并把 `multi_view_3d` 移到末尾；其它工具保留请求顺序。多个 geometry 阶段按此解析顺序执行，**后一个成功结果覆盖前一个**；后一个失败时保留已有成功结果，失败 Percept 仍可见。这不是自动质量选择或模型回退策略。
- `options.fuse=False` 关闭补充融合，但 pipeline 不会跳过显式 geometry 工具：`multi_view_3d` 内部遵守此选项返回空，`stereo_3d` / `triview_3d` 仍可填写 `fused`。要统一关闭某工具须从工具列表移除。

### 5.4 当前工具与模型

| 工具 | 类型 | 要求 |
|---|---|---|
| `fake` | fake | 纯 numpy 合成：按"颜色阈值 + 连通域"输出 mask/box，用于 CI；无外部依赖 |
| `sam3` | segmenter | 官方 `sam3` 包；支持 text / box；输出 mask/box/score；加载一次、模型 sha256 入 provenance |
| `multi_view_3d` | geometry | 多视角 mask + 标定 → Geometry（见 §9） |
| `stereo_3d` | geometry | 恰好两路标定 RGB；CPU SGBM、FoundationStereo 或 Fast-FoundationStereo；可选原左图 mask |
| `triview_3d` | geometry | 恰好三路标定 RGB；多组 SGBM 融合，或原生 MVSAnywhere HERO / DA3-BASE |

双目只支持当前实现允许的水平左右顺序；反向/竖直对显式拒绝。神经三目要求同尺寸三图和不同相机中心，
DA3 还要求中心非共线。FFS 官方剪枝权重的 `max_disp` 上限为 416 像素。参数与运行命令见部署文档。
这些神经几何工具描述场景可见表面，文本提示不会分割目标；没有自动 SAM3 mask 接线、对象补全或自动模型回退。

未来可选：`yoloe`/`omdet`（detector）、独立单目深度工具、`vlm_grounding`。
DA3 已作为三目后端部署，不表示独立单目工具也已实现；`adaptive_3d` 仍为设计。

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
- 错误映射（HTTP）：400 invalid_request / 401 unauthorized / 413 payload_too_large / 503 busy / 504 timeout / 500 internal；`not_found`、工具不可用以及工具拒绝某个 prompt 均通过 HTTP 200 的 Percept 状态表达（附录 A）。
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
- 客户端提供 `perceive_local()`：同进程调用共享 `PipelineRunner`，不经过 HTTP；工具自身仍可使用独立模型子进程。
- `wait_until_ready(timeout_s=120.0, interval_s=0.25, *, warm=True)` 等待服务可访问和 warmup 结束；预热失败也算结束，须另查 `tools[].status` 与 `warmup_state()`。双目/三目配置关闭通用单图预热，需使用对应 smoke 核实真实推理就绪。

CLI：

```
percepta serve --config config.yaml
percepta perceive --image cam_head=img.png --prompt text:bottle [--calib calib.json] [--json]
percepta tools list
percepta probe [--config config.yaml]
percepta models check --config config.yaml [--no-hash] [--json]
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

算法（当前实现，D35；各路径失败后尝试下一条）：

1. **有深度**：取 mask 内深度像素 → 反投影到相机 → 世界点云 → 中位数中心、分位数包围盒、顶部 z、PCA/最小外接矩形得 yaw。
2. **至少两路可用 mask**：采样轮廓几何（silhouette/prism）：
   - 每相机选最高分的可用 mask；`context["table_plane"]: [a,b,c,d]` 优先于 `table_z`。多视角缺省桌高为 0 m，并标注 `table_z_assumed`。
   - 世界 XY 网格中，一个底面单元仅在从桌面到候选顶部的采样柱体都投进所有 mask 时保留；不是取中高平面的投影阴影。
   - 扫描候选高度，对保留柱体重新投影并计算跨视角 IoU，取完整曲线的**最高分高度**；并列取较低者。默认最大高度 0.5 m、目标步长 0.01 m，实际采样数有上下限。
   - `prism_iou_threshold` 默认 0.5，仅决定是否标注 `geometry_low_confidence`，不用于首次达标即停止。
   - 当前还尝试 `hang` 候选，并仅在重投影分数超过 prism 一定幅度时选用；该候选有已确认的坐标限制，见 `docs/documentation-changes.md` 的待修代码问题，不能视作完整对象补全。
   - 输出中心、包围盒、顶部 z、yaw 及分轴不确定度估计；`uncertainty_m` 取横向和高度估计的较大值，视角分歧独立报告。
3. **最高分单视角兜底**：需要调用方明确提供可用 `table_plane` / `table_z`；否则不输出 Geometry。高度为粗略估计，`views_used=1`，当前高度不确定度至少 0.05 m；不是多视角测量结果。

约束：

- 无 `calibration` 时**禁止**输出 `Geometry`（不得用像素猜米）。
- 所有几何结果必须带 `frame="world"` 与 `uncertainty_m`。
- 至少 2 视角才允许 `views_used>=2`；单视角必须显式标注。
- 轮廓法沿世界 +Z 挤出，假定桌面近水平；倾斜 `table_plane` 只用于估计对象处桌高，不构成任意斜面重建。
- 不确定度是启发式估计，不是经过标定的误差界；本轮审计另记录了横向估计的世界原点依赖问题。

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
warmup:
  enabled: true              # 通用单图预热；双目/三目配置关闭并改用对应 smoke
  height: 480
  width: 640
  prompt: object
```

### 11.2 并发

- 每工具一把进程内锁（模型推理串行）；服务层 `BoundedSemaphore` 控制排队；超时/满 → 503/504。
- 多 GPU：一个进程绑一张卡（`CUDA_VISIBLE_DEVICES`）；多实例多端口。
- 子进程模型可各自设置 `cuda_visible_devices`；空字符串继承进程环境，不会自动选择空闲显卡。当前没有自动多 worker 调度或崩溃后重启。

### 11.3 可观测

- 结构化日志（JSON）：`request_id, tool, camera, status, latency_ms, cache_hit`。
- 指标（可选）：`percepta_requests_total{tool,status}`、`percepta_latency_ms{tool}`、`percepta_inflight`、`percepta_cache_hits_total`。
- `/healthz` 必须反映工具加载状态（`ok|loading|error`）。
- 未启用工具不列出；未加载的 `lazy` 在健康接口映射为 `ok`，所以健康可访问不等于真实模型已热态。常驻 worker 退出或失败会更新为 `error`（D38）。

### 11.4 缓存

- key：`sha256(image) + prompt + tool + options(相关子集)`；LRU + TTL；命中时 `provenance.cache_hit=true`（不改变结果内容）。
- options 只包含 `masks` / `min_score` / `top_k`；不包含 context、标定、模型配置。缓存仅用于无上游结果的阶段。依赖这些未入 key 数据的几何工具应关闭缓存，当前神经双目/三目配置均如此。

---

## 12. 安全与许可

- 默认只绑 loopback；跨机用 SSH 隧道；文档给出 `ssh -N -L` 与 systemd 常驻写法。
- 可选 bearer token；不实现 TLS。
- 请求体大小、超时、队列上限必须生效。
- 请求不接收源码、模型文件路径或 pickle，只收内联图像、标定和上下文。模型 worker 可在校验固定官方资产 SHA 后加载可信权重；FFS 的官方完整模型包含 pickle，不允许请求指定或替换其来源。
- 仓库许可建议 **Apache-2.0**；**不得 vendor** 任何模型权重或违反上游许可的代码。
  - SAM3 权重：SAM License（gated），运行时由用户提供路径。
  - Ultralytics 系（YOLOE/YOLO-World）：AGPL-3.0，作为**可选工具**，默认不启用、不随仓库分发。
  - GroundingDINO/OWLv2/OmDet-Turbo/SAM2：Apache-2.0（以各自仓库为准）。
  - 当前 FoundationStereo、Fast-FoundationStereo、MVSAnywhere HERO 权重限非商业研究；DA3-BASE 源码与权重为 Apache-2.0。

---

## 13. 仓库结构与依赖

```
percepta/
├── pyproject.toml            # hatchling; extras: service / stereo；默认 dev 依赖组
├── uv.lock / .python-version # 主库锁与 Python 3.10
├── README.md
├── AGENTS.md                 # 从本文档 §0/§18 派生的协作规约
├── configs/example.yaml
├── docs/spec.md              # 根 percepta-SPEC.md 的同步镜像
├── docs/documentation-changes.md # 带时间、原文与原因的文档修订记录
├── uv_envs/<model>/          # SAM3 与四个神经深度模型各自的项目、锁和资产清单
├── scripts/                 # 资产 bootstrap、模型 smoke、e2e 与锁检查
├── src/percepta/
│   ├── __init__.py           # 公共 API 导出
│   ├── schema.py             # §3 全部模型
│   ├── masks.py              # RLE 编解码、IoU、面积/质心
│   ├── geometry.py           # project/unproject、平面、包围盒、PCA/yaw
│   ├── fusion.py             # §9 multi_view_3d
│   ├── hull.py               # 世界 XY 网格与采样柱体轮廓约束
│   ├── verification.py       # §10
│   ├── prompts.py            # §8
│   ├── registry.py           # entry points 加载、能力校验
│   ├── config.py             # pydantic + PyYAML（无 pydantic-settings）
│   ├── pipeline.py           # HTTP / SDK 共用编排
│   ├── warmup.py / weights.py # 预热与模型资产检查
│   ├── cache.py
│   ├── service/{app.py,routes.py,metrics.py,logging.py}
│   ├── client.py             # PerceptaClient + perceive_local
│   ├── cli.py
│   └── tools/{base,fake,sam3,multi_view_3d,stereo_3d,triview_3d}.py
├── src/percepta/backends/     # 模型跑在哪：inprocess / subprocess / http（v0.1.3 新增）
│   ├── {base,protocol,inprocess,subprocess_backend,http_backend,worker,reference}.py
│   └── stereo / multiview 协议、常驻 worker 与各模型适配器
├── tests/
│   ├── unit/                 # schema/masks/geometry/prompts/verification/backends/weights
│   ├── contract/             # HTTP 契约（fake 工具，无 GPU）
│   └── gpu/                  # pytest -m gpu，默认跳过
└── systemd/perceptad.service
```

默认依赖仅为 `pydantic>=2`、`numpy>=1.26,<2.4`、`pillow>=10`、`pyyaml>=6`。
`service` extra 提供 FastAPI/uvicorn，`stereo` extra 提供 OpenCV；pytest/httpx/ruff 以及服务测试依赖位于默认安装的 `dev` 依赖组。**不得**把 torch 或模型框架设为默认依赖，也没有根 `sam3` extra。

依赖由 **uv** 管理（D22）。D33 已替代 D27 的“不维护模型依赖清单”：本仓库在
`uv_envs/sam3`、`foundation-stereo`、`fast-foundation-stereo`、`mvsanywhere`、`da3`
维护彼此独立的 `pyproject.toml` / `uv.lock` / `.python-version`；后四者另有 `assets.json`。
主库用 `tools.<name>.python` 指向对应环境，框架依赖不进入根锁的默认安装。
锁由官方源生成，CI 先检查可移植性，再检查根和全部模型锁是否与项目一致（D34）。
权重、模型源码和 `.venv/` 不提交；路径、哈希及部署命令见 `docs/model-isolation.md`。

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
| **M0 骨架** | pyproject、schema.py、masks.py、registry、fake 工具、pytest 骨架 | S2 绿；`uv sync` 可用 |
| **M1 契约** | service（FastAPI）+ client + CLI + 错误映射 | S1/S4 绿；`percepta serve` 可起 |
| **M2 SAM3** | `tools/sam3.py`（text/box）、provenance、加载/释放、warmup | S9 手动通过；无 GPU 环境不导入 |
| **M3 几何** | geometry.py + fusion.py（深度 & 轮廓） | S3/S5 绿 |
| **M4 验证/缓存** | verification.py、cache.py、指标、日志 | S6/S7 绿 |
| **M5 加固** | 队列/超时/大小限制、独立模型进程、systemd、文档、性能基线 | 原始验收完成；README 有部署示例；Docker 已由 D27/D33 调整出交付范围 |

以上是原始里程碑，已完成；扩展方案复用了 M0–M5 编号，不能混为一组。四个神经双目/三目模型的
部署与回归证据见 `docs/model-isolation.md` §11–§12。CPU 测试数量随代码变化，不在本规格中冻结旧计数。

建议并行拆分（文件所有权互斥）：
- A：`schema.py` + `masks.py` + 单元测试（M0/M1 依赖）
- B：`service/*` + `client.py` + `cli.py` + HTTP 契约测试
- C：`tools/sam3.py` + `tools/fake.py` + `registry.py`
- D：`geometry.py` + `fusion.py` + `verification.py` + 几何测试
- E：`config.py` + `cache.py` + systemd + 文档（独立模型环境按 D33）

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

## 17. 原始性能目标（v0.1，历史目标，非当前实测承诺）

下面保留最初目标值，不能解释为已达标。当前模型实测按输入尺寸、GPU 和日期分别记录在
`docs/model-isolation.md`，不据不同输入的单次 smoke 进行模型排名。

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
| v0.2.1 | 2026-11-05 | 工程加固：模型运行时改为 `uv_envs/<model>/` 独立锁定项目（移除根 `sam3` extra，D33）、锁文件可移植性由 CI 强制检查（D34）、空 `cuda_visible_devices` 语义修正（D32）。包版本 0.2.1，新增首个 CI 工作流 |
| v0.2.2 | 2026-11-05 | 附录 A 的 `tool_unavailable` 由 503 修正为 **200**，与 D17 实现对齐（消除 spec 自身矛盾）；D17 从「待复核」转为「已决」。补充该行的语义说明与调用方契约 |
| v0.3.0 | 2026-11-09 | D35 已落地：采样柱体轮廓约束、完整 IoU 曲线峰值搜索、可选分轴不确定度与 method；本行补记对应已发布版本，未重写此前历史 |
| v0.3.0-doc.1 | 2026-11-12 | 按 `534066a` 对齐当前字段、pipeline、模型隔离、双目/三目能力与 SAM3 API；D36–D38 部署及修复仍为 Unreleased。原文、原因和代码审计限制见 `docs/documentation-changes.md`；本次不升级包或 schema |

---

## 19. 原始待决问题与当前结论

1. point：schema 保留，当前 SAM3 官方处理器不支持，返回 `unsupported_prompt`；`fake` 可处理点提示。
2. 多相机调用：每请求每工具一次，带全部相机，推理由 per-tool 锁串行化；不提供自动多 worker 调度。
3. 独立单目深度工具：仍为可选未来项；当前 DA3 是三目后端。
4. 视频/跟踪：尚未实现，仍属当前非目标。
5. 与具体 Agent Policy（如 PeerAgent）的适配层是否进本仓库？建议**不进**，在策略侧写 adapter。
6. 缓存 key 是否包含 `context`？v0.1 不包含（避免调试困惑），在 §11.4 注明。

### 19.1 实现取向（v0.1 已决）

下列条目保留早期裁定；**2026-11-12 对齐说明**：D4/D16 的首过阈值与中高截面已由 D35 替代，
D23/D27 的模型依赖边界已由 D33 替代。现行描述以 §3/§5/§9/§13 及下表后续决策为准，
完整理由见 `docs/decisions.md`（同编号），修订前原文仍可在基线 `534066a` 查阅。

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
| D17 | 附录 A 的 `tool_unavailable → 503` 与"Percept 承载状态"冲突 | **HTTP 200** + Percept `status="unavailable"` + 顶层 `warnings` 含 `tool_unavailable`；`/healthz` 报 `error`。附录 A 该行已于 v0.2.2 同步修正（不再记为偏离）。理由：本接口是批量语义，503 会丢失部分结果与"哪个工具坏了"的信息。调用方应看 `percepts[].status` 而非仅看状态码 |
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

### 19.2 后续决策与当前实现

| # | 当前结论 |
|---|---|
| D28 | SAM3 实测接入：全关键字构造、传入 model 的 processor、PIL 图像、BPE 资产及 BF16 输出转换 |
| D29–D30 | 后台 pipeline warmup；客户端等待预热结束，另查工具及预热结果 |
| D31–D32 | introspection 统一模型元数据；空 CUDA 可见设备配置继承环境，不自动选卡 |
| D33–D34 | 每模型独立锁定 uv 项目，主库不含模型 extra；CI 检查官方源及所有锁的新鲜度 |
| D35 | 采样柱体约束、IoU 峰值、分轴不确定度；替代 D4/D16 的旧算法；本轮确认的两个实现限制另记修订日志 |
| D36 | FoundationStereo 独立双目运行时，实际 RGB/mask 校正与 OpenCV +Z 几何 |
| D37 | FFS、MVSAnywhere、DA3 独立运行时；严格资产加载、显式尺度与无静默模型替换 |
| D38 | DA3 尺寸校验及 float64 位姿重基准、wheel 资产清单、worker 故障健康状态 |

---

## 附录 A：错误码

| code | 场景 | HTTP |
|---|---|---|
| `invalid_request` | 字段缺失/类型错/尺寸不匹配 | 400 |
| `unsupported_prompt` | 工具不支持该 prompt 类型 | **200**（Percept.status=`invalid_request`，message/code=`unsupported_prompt`；D1） |
| `unauthorized` | token 缺失/错误 | 401 |
| `payload_too_large` | 超过 `max_body_bytes` | 413 |
| `busy` | 队列满 | 503 |
| `timeout` | 超过 `timeout_ms` | 504 |
| `tool_unavailable` | 工具未加载/模型缺失 | **200**（Percept.status=`unavailable`）— 见 D17 |
| `internal` | 未预期异常 | 500 |

**关于 `tool_unavailable`（D17）**：这一行原先写「503（Percept.status=`unavailable`）」，
自相矛盾——503 表示服务整体不可用，响应体不会是 `PerceiveResponse`，而 `Percept` 恰恰是
`PerceiveResponse` 里的对象。v0.2.2 起明确为：

- **HTTP 200**，响应体是完整的 `PerceiveResponse`；
- 受影响工具产出的 Percept 取 `status="unavailable"`，`message` 说明原因，
  `extras.code="tool_unavailable"`；
- 顶层 `warnings` 含 `"tool_unavailable"`；
- `/healthz` 中该工具为 `error`。

理由：本接口天然是**批量语义**（一次请求可含多相机 × 多 prompt × 多工具）。若因其中一个工具
不可用而返回 503，会同时丢失两样东西：其它工具已经算出的结果，以及「具体是哪个工具坏了」这一
信息。调用方应检查 `percepts[].status` 与 `warnings`，而不是只看 HTTP 状态码。

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

- SAM3：官方 `sam3==0.1.4`，当前适配器使用如下已核实的调用；构造器首位置参数是 BPE 路径，不能位置传入 device：

  ```python
  model = sam3.build_sam3_image_model(
      bpe_path=bpe_path, device=device, eval_mode=True, checkpoint_path=checkpoint,
      load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=True,
  )
  processor = Sam3Processor(model, resolution=1008, device=device, confidence_threshold=0.5)
  state = processor.set_image(PIL.Image.fromarray(rgb_uint8))
  state = processor.set_text_prompt(prompt="bottle", state=state)
  # 框提示：processor.add_geometric_prompt(box=[cx, cy, w, h], label=True, state=state)
  ```

  输出 `masks (N,1,H,W) bool`（原图尺寸）、`boxes xyxy`、`scores`；BF16 结果先转 float32 再转 NumPy。
  同图状态可复用；官方处理器不提供 point prompt。本段是适配器调用摘录，模型运行在其独立环境中。
- 权重 gated（HuggingFace `facebook/sam3`）；路径由配置注入，仓库不含权重。
- 若 SAM3 上游 API 变更，在 `tools/sam3.py` 的适配边界处理，不得泄漏到 schema。
- 四个神经双目/三目模型的适配器、协议及资产边界见 `docs/model-isolation.md` §11–§12；不复用 SAM3 单图 mask 协议。

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

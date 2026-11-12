# 模型隔离与依赖管理

> 状态：运行时设计与部署记录，现行说明于 **2026-11-12** 对照代码 `534066a` 核对。
> 本文档回答一个问题：**当两个感知模型要求互斥的依赖时，
> 怎么在不引入容器的情况下让它们共存。**
>
> 相关：`percepta-SPEC.md` §5（工具协议）、`AGENTS.md` §1（uv 工作流）、
> `docs/decisions.md` D22–D38。
>
> **修订记录（2026-11-12）**：原状态为“v0.1（设计阶段定稿）”。现已存在 SAM3、
> FoundationStereo、Fast-FoundationStereo、MVSAnywhere HERO、DA3-BASE 五套运行时；
> §6 的帧布局及 §10 的早期手工安装命令保留为历史，当前部署用 §5/§11/§12 的独立锁定项目。
> 带日期的说明保留被纠正的原话；本次没有新增模型、调度、RoboDojo 接入或评测功能。
> 本轮各处原文、修订内容和原因汇总于[文档修订记录](documentation-changes.md)。

---

## 1. 问题

一个 Python 进程里只能存在**一份** `numpy`、**一份** `torch`。所以：

- `sam3` 要求 `torch>=2.7`
- 假想的 `depth_anything_v2` 要求 `torch==2.6`
- 假想的 `yoloe` 要求 `numpy<2`

这三者在同一个环境里**无解**。不是配置问题，是 Python 导入机制决定的。

**任何解法都必须让冲突的模型跑在独立解释器进程里。** 差别只在于这个进程怎么起、怎么通信。
容器是其中一种最重的做法。

## 2. 不采用容器作为默认方案

容器能解决 **CUDA 版本 / 系统库** 级冲突，但代价是：

- 每个模型 GB 级镜像；
- 需要 docker 权限（本机无 sudo，虽然 docker 组可用）；
- 调试要进容器；
- 代码改动需要重新 build。

绝大多数感知模型的冲突是 **Python 包版本**级别，进程级隔离已经足够。
因此：**容器是最后手段，不是默认路径。**

## 3. 实测结论（本机验证）

| 能力 | 结果 |
|---|---|
| 同一台机器并存 `numpy 1.26.4` 和 `2.2.6` 的两个 venv | ✅ 互不干扰 |
| 一个 venv 的解释器通过 `subprocess` 调用另一个 venv | ✅ 返回正确结果 |
| Python 冷启动延迟 | **0.01 s** |
| `uv run --with <pkg>` 临时隔离环境 | ✅ 可用（仅适合一次性实验） |

结论：**多 venv + 子进程方案在本机完全可行，启动开销可忽略。**

> **2026-11-12 说明**：原“Python 冷启动延迟 0.01 s”“启动开销可忽略”仅指解释器启动，
> 不包含框架导入、权重读取或 CUDA 初始化。模型加载仍需数秒至数十秒，不能将该数字当作
> 服务就绪时间；真实 cold/warm 测量见 §10–§12。

## 4. 三种后端形态

关键设计原则：**工具接口不感知模型跑在哪里。**

`PerceptionTool.perceive(ctx, prompts) -> list[Percept]` 是冻结的（spec §5.1）。
因此"进程内 / 子进程 / HTTP"可以做成可替换的后端适配器：

```
tools/<model>.py
    └── <Model>Tool  ──►  backend: ModelBackend
                            ├── InProcessBackend    同环境装 torch（最快，零拷贝）
                            ├── SubprocessBackend   独立 venv（推荐默认）
                            └── HttpBackend         独立进程/另一台机器
```

配置选择，不改代码：

```yaml
tools:
  sam3:
    backend: subprocess          # inprocess | subprocess | http
    python: uv_envs/sam3/.venv/bin/python   # subprocess 后端
    endpoint: http://127.0.0.1:8801  # http 后端
```

### 4.1 选型建议

| 场景 | 后端 | 理由 |
|---|---|---|
| 只有一个模型，无冲突 | `inprocess` | 最简单，无通信开销 |
| 两个模型依赖冲突 | `subprocess` | 已验证，0.01 s 启动，进程级隔离 |
| 需要多机 / 独立重启 | `http` | 可分布，可独立看日志 |
| CUDA 版本冲突 | 容器 | 进程隔离解决不了系统库冲突 |

**默认路径：先 `inprocess`；`uv lock` 报冲突时切 `subprocess`。**

> **2026-11-12 现行说明**：上文“先 `inprocess`”保留为通用后端的历史选型建议，代码的
> 通用默认值仍是 `inprocess`；仓库提供的五套本地真实模型配置均使用 `subprocess`，
> 直接复用各自的独立环境，不需先把模型框架装入根项目试冲突。示例解释器路径由
> `.venvs/sam3/bin/python` 更新为 `uv_envs/sam3/.venv/bin/python`（D33）。
> `ModelBackend` 的 `set_image/segment` 形状用于 SAM3；双目和三目使用各自的视差／深度
> RPC，保持共同的“独立解释器＋常驻 worker”边界，不把所有模型强塞进分割接口。

## 5. 目录与依赖约定

### 5.1 每个模型一个隔离环境

```
uv_envs/                          # 提交：环境定义（可复现）
├── sam3/
├── foundation-stereo/
├── fast-foundation-stereo/
├── mvsanywhere/
└── da3/
```

五个项目均提交 `pyproject.toml`、`uv.lock` 和 `.python-version`；每个项目生成的 `.venv/`
保持 Git 忽略。后四个源码模型项目还提交 `assets.json`，记录固定源码版本、权重来源与 hash；
SAM3 的权重／词表配置见 `configs/local-sam3.yaml`。

> **修订记录（2026-11-12）**：原目录树只列 `sam3/` 和示意 `depth/`，未展示已部署的
> 四个新模型；现改为当前五项目及实际元数据，未创建新的模型目录。

**为什么目录叫 `uv_envs` 而不是 `envs`**：在机器人语境下 "envs" 很容易被理解为仿真环境
（simulator environments），与 Python 虚拟环境混淆。`uv_envs` 消除了这个歧义。

**为什么环境定义要进版本库**：模型运行时的可复现性和库本身同等重要。各模型锁与根项目锁
分开维护，但一起进入版本库、接受评审和回滚。`.venv/` 忽略。

创建方式：

```bash
# 一条命令从锁文件重建整个模型环境（约 7 GB，首次下载 CUDA 库较慢）
uv sync --project uv_envs/sam3 --python 3.10 --locked
# 产物：uv_envs/sam3/.venv/，配置里指向 uv_envs/sam3/.venv/bin/python
```

> **修订记录（2026-11-12）**：原“锁文件在库外”指与根项目分离，不表示不纳入此仓库；
> 原重建命令没有 `--locked`，现显式要求使用既有锁，避免重建时隐式更新依赖解析。

### 5.2 库的边界：根项目与模型运行时分别锁定

根 `pyproject.toml` 的默认依赖**不包含任何模型框架**，也**没有** `sam3` extra。
早期 extra 未能复现本机验证过的 CUDA 运行时，且缺少 sam3 未声明却实际 import 的 `psutil`。
模型运行时需要自己的版本、索引和锁文件，因此是 `uv_envs/<model>/` 下的独立项目（决策 D33）。

注意库**仍然**约束模型**权重**的路径与完整性 —— 这是两回事，见 §5.4。

仓库维护的是每个已部署模型单独的、经过验证的运行时清单，不是一个适用于所有模型和
CUDA / 驱动组合的根依赖集合。重建这些环境由部署方执行，库通过配置选择解释器。

> **修订记录（2026-11-12）**：原题“库的边界：不管模型依赖”、原文“本库不管理任何模型
> 框架的依赖”“一旦库开始维护模型依赖清单……必然腐化”表述过宽；D33 已明确在仓库内
> 提交各模型的独立 `pyproject.toml` / `uv.lock`。这里澄清维护位置和执行边界，资产契约不变。
> 原“那样构造出来必然是坏的：从 PyPI 解析 `torch` 得到的是 CPU 版”记录早期环境的问题，
> 不应作为所有平台的 PyPI 轮子都只有 CPU 构建的断言；现行要求是使用已验证的独立 CUDA 锁。

库只保证三件事：

1. **发现**：`percepta.tools` entry point 组能找到第三方工具实现；
2. **调用**：工具通过 backend 协议被调用，不关心模型装在哪；
3. **约束**：模型权重的**位置与完整性**由库校验（见 §5.4）。

### 5.3 主环境保持干净

默认 `uv sync` 在主环境（`.venv`）安装根默认依赖和 `dev` 组；可按用途显式添加
`service` / `stereo` extra，模型框架仍留在独立运行时。默认测试可以在无 GPU 的机器运行，
OpenCV 几何测试需另装 `stereo` extra。

> **修订记录（2026-11-12）**：原“永远只装……默认依赖 + `dev` 组”漏掉合法的可选
> `stereo` / `service` 安装，原“全量测试”也未区分需要 OpenCV 或真实模型的测试。

模型依赖由部署方装进 `uv_envs/<model>/.venv`（由该目录下的 `uv.lock` 生成）或任意虚拟环境，
然后通过 `tools.<name>.python` 指过去。

### 5.4 模型资产契约（库负责的部分）

根项目与各模型依赖分别锁定；库还校验**资产的路径与完整性**，因为截断或版本错误的权重
可以"成功加载"，然后安静地输出错的结果。

配置形态：

```yaml
tools:
  sam3:
    checkpoint: /models/sam3.pt        # 可位于仓库外或 ignored checkpoints/；永不 Git 跟踪
    checkpoint_sha256: "abc123…"       # 可选：64 位十六进制，YAML 里必须加引号
    verify_weights: if-declared        # if-declared | always | never
```

三种严格度：

| 模式 | 行为 | 用途 |
|---|---|---|
| `if-declared`（默认） | 声明了哈希就校验，否则只记录 | 探索期可用；仅声明期望哈希后才比对 |
| `always` | 本地文件没有声明哈希就拒绝加载 | CI / 生产 |
| `never` | 完全不校验 | 超大权重 + 慢磁盘的逃生门 |

> **修订记录（2026-11-12）**：原“依赖不管”已改为分别锁定的边界；checkpoint 注释原
> “绝不在仓库里”应是永不纳入 Git 跟踪，允许置于仓库内被忽略的 `checkpoints/`。
> 原 `if-declared` 用途“生产配置天然安全”保证过强：未声明期望 hash 时只记录、不比对，
> 不能据此声称完整性已验证。

启动前自检：

```bash
percepta models check --config configs/example.yaml
```

它会在服务启动前报告每个 checkpoint 的**存在性、实际 sha256、是否真正比对过**。
`verified: false` 表示"只记录未校验"，绝不谎报。

注意：YAML 会把纯数字的哈希解析成整数（`int(0)` 为假值）。
库对此**报错而非静默跳过**，并在消息里提示加引号。

## 6. 子进程通信协议（历史草案，非当前逐字节规范）

> **修订记录（2026-11-12）**：原标题为“子进程通信协议（草案）”。下列生命周期图和帧布局
> 保留原文，仅解释设计意图，不应照抄实现。现行编码以
> `src/percepta/backends/{protocol,stereo_protocol,multiview_protocol}.py` 为准：分别传输
> SAM3 的图像／提示／分割结果、双目 RGB／视差、三目 RGB＋标定／米制深度，均带外层长度
> 分帧。双目与三目 RPC 另限制像素／metadata 大小，实施启动及读写 deadline；stdout 仅用于协议。

### 6.1 为什么不用 JSON

`480×640×3` 的 uint8 图像转 JSON 文本约 **10–20 MB**，而原始字节只有 **921 KB**。
协议必须走二进制。

### 6.2 worker 生命周期

```
主进程                               worker（独立 venv）
  │  spawn(python -m percepta_worker)
  │────────────────────────────────────► 启动，加载模型（一次性，数秒）
  │◄──────────────────────────────────── {"ready": true, "model_sha256": "..."}
  │
  │  ── 每请求 ──
  │  [header][image bytes][prompt]  ──►
  │◄── [header][mask rle][boxes][scores]
  │
  │  shutdown
  │────────────────────────────────────► 释放显存，退出
```

**模型只加载一次，进程常驻**，所以每请求只付通信成本（毫秒级）。

### 6.3 帧格式

```
请求帧：
  uint32 magic      0x50524350 ('PRCP')
  uint32 version    1
  uint32 h, w, c
  uint32 prompt_len
  bytes  prompt_json       (Prompt.model_dump_json())
  bytes  image_raw         长度 = h*w*c，uint8 RGB
  bytes  depth_raw         (可选，长度 = h*w*4，float32)

响应帧：
  uint32 magic      0x50524352 ('PRCR')
  uint32 status_len
  bytes  status_json   {"status": "ok", "error": null}
  uint32 n_obs
  重复 n_obs 次：
    float32 score
    uint32  box_len     (0 表示无 box)
    float32[4] box_xyxy (可选)
    uint32  counts_len
    uint32[counts_len] rle_counts
    uint32  h, w
```

**所有整数小端序。** magics 用于检测协议错位（worker 崩溃时不会静默读到垃圾）。

### 6.4 错误语义

| worker 侧 | 主进程侧映射 |
|---|---|
| 模型加载失败（OOM / 缺权重） | `ToolUnavailable` → Percept `status="unavailable"` |
| 请求格式错误 | `invalid_request` |
| 推理抛异常 | `unavailable` + 错误信息进 `message`（**不吞成 `not_found`**） |
| worker 进程崩溃 | 主进程检测 EOF → `unavailable`，并记录 worker 已死 |

## 7. 显存管理

这是所有方案的共同难点，与隔离方式无关。

**约定：一个模型 worker 绑定一张卡。**

```yaml
tools:
  sam3:
    cuda_visible_devices: "0"
  depth:
    cuda_visible_devices: "1"
```

worker 启动时由主进程注入 `CUDA_VISIBLE_DEVICES`。好处：

- 避免显存争抢导致 OOM；
- 不同卡的模型可以真正并行；
- 显存规划写在配置里，可审计。

单卡场景：所有 worker 共享 `"0"`，由服务层的 per-tool 锁（spec §11.2）串行化。

> **2026-11-12 更正**：原“所有 worker 共享 `0`，由……per-tool 锁……串行化”不能保证
> 不同工具的 worker 互斥；锁只串行同一个工具，没有跨工具的全局 GPU 调度器。部署方需安排
> 卡与显存预算。`cuda_visible_devices: ""` 仅继承环境，不自动选择空闲 GPU（D32）。

## 8. 加一个新模型的检查清单

1. **为模型建立独立项目**：在 `uv_envs/<model>/` 提交 `pyproject.toml`、官方源生成的
   `uv.lock` 和 Python 版本，保持根默认依赖不变。
2. **依赖由部署方重建**：运行 `uv sync --project uv_envs/<model> --locked`；外部自管环境
   仍可通过解释器路径接入。
3. **在 `configs/*.yaml` 里声明 backend、python 路径与 `cuda_visible_devices`**
4. **工具实现只依赖 backend 抽象**，不直接 `import torch`
5. **写一个 fake backend**，让逻辑能在无 GPU 环境测试（见 `backends/reference.py`）
6. **权重走配置注入**：`checkpoint` + `checkpoint_sha256`，绝不进 git
7. **跑 `percepta models check`**，并在 `/healthz` 里如实报告未装模型为 `error`

> **修订记录（2026-11-12）**：原第 1 项“相容 → 走 extra，`inprocess` 后端”、第 2 项
> “不写进本仓库的 pyproject”已被 D33 的独立项目约定替代；禁止增加的是根默认模型依赖，
> 不是禁止提交 `uv_envs/<model>/pyproject.toml`。`inprocess` 仍是外部部署可选的通用后端。

## 9. 未决问题

1. **worker 是否复用**：当前设计是"每模型一个常驻 worker"。如果同时有 5 个模型，
   就是 5 个常驻进程。是否需要按需启动/空闲回收？建议 v0.2 再定。
2. **图像零拷贝**：目前设计走管道复制。`torch` 支持共享内存（`/dev/shm`），
   但增加复杂度。当前 `480×640` 复制约 1 ms，可接受，暂不优化。
3. **多个 prompt 的批量**：协议草案是一帧一 prompt。批量能减少往返，
   但需要重新设计帧格式，v0.2 再考虑。
4. **worker 的健康监控**：主进程如何发现 worker 卡死（而非崩溃）？需要心跳或超时。
   当前依赖请求级超时（spec §6），可能不够。

> **2026-11-12 状态更新**：上述条目保留早期问题原文。常驻复用已实现；按需空闲回收、共享
> 内存、prompt 批处理没有在本轮新增。第 4 项原“当前依赖请求级超时”不再概括全部模型：
> 神经双目／三目的 RPC 有启动和管道 IO deadline、故障回收，以及不等待推理锁的健康快照；
> 无需等待下一次推理即可识别 worker 退出。没有实现独立心跳或统一的多模型调度器。

---

## 10. 真实 SAM3 实测记录（v0.1，本机 H20）

这一节记录**真正把 SAM3 跑起来**之后才知道的事。前面几节是设计，这一节是事实。

### 10.1 环境实测

> **2026-11-12 历史标记**：下列 `.venvs/sam3`、`uv pip install` 命令是首次实测日志，
> 保留其原文供追溯；当前重建入口是 `uv sync --project uv_envs/sam3 --locked`，
> 不应通过这些历史命令重新生成锁文件。

```bash
uv venv .venvs/sam3 -p 3.10
uv pip install --python .venvs/sam3/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-url https://mirrors.aliyun.com/pypi/simple \
  --index-strategy unsafe-best-match \
  "torch==2.10.0+cu128" "sam3==0.1.4" pydantic
```

**必须放在一条命令里**，且 `--index-strategy unsafe-best-match` 不可省。原因：`sam3` 依赖 `timm`，
而 `timm` 只写了裸 `torch`（无版本约束）。分两条命令装时，第二条会把 cu128 版覆盖成
PyPI 的 CPU 版，并拖入一整套 **cu13** 库（与 cu128 不兼容）。

实测结果：py3.10 可用 `torch 2.10.0+cu128` + `torchvision 0.25.0+cu128`，8 张 H20 可见。

### 10.2 sam3 包的依赖声明不完整

`sam3==0.1.4` 的 METADATA **缺少 `psutil`**，但 `sam3/model/sam3_video_predictor.py`
顶层就 `import psutil`。所以 `import sam3` 会直接失败。

补一个包即可：

```bash
uv pip install --python .venvs/sam3/bin/python psutil
```

（其余 import 如 `cv2`/`scipy`/`numba`/`hydra` 都在 eval/train 路径，图片推理不需要。）

### 10.3 三个 API 陷阱（静态阅读源码 + 实测确认）

| 陷阱 | 正确用法 | 错误后果 |
|---|---|---|
| `build_sam3_image_model` 首参 | 是 `bpe_path`，**不是** `device` | 位置传参会把设备名当词表路径 |
| `Sam3Processor(...)` | `model` 是**必填**首参 | 只传 `resolution=` 直接 TypeError |
| `add_geometric_prompt` | 签名 `(box, label, state)`，`label` 必填 | 缺参数报错 |

实测签名（`inspect.signature`）：

```
build_sam3_image_model(bpe_path=None, device='cuda', eval_mode=True,
                       checkpoint_path=None, load_from_HF=True,
                       enable_segmentation=True, enable_inst_interactivity=False,
                       compile=False)
Sam3Processor(model, resolution=1008, device='cuda', confidence_threshold=0.5)
```

### 10.4 BPE 词表：wheel 里没有

`build_sam3_image_model()` 默认去 `../assets/bpe_simple_vocab_16e6.txt.gz` 找词表，
但 **PyPI wheel 不含 `assets/` 目录**（145 个文件里没有）。

官方文件在 GitHub 的 **`sam3/assets/`** 子目录（注意不是仓库根的 `assets/`）：

```bash
curl -L -o checkpoints/bpe_simple_vocab_16e6.txt.gz \
  https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz
```

1.3 MB，262,145 行，标准 CLIP BPE 词表。通过 `tools.sam3.bpe_path` 注入。

### 10.5 最隐蔽的陷阱：HWC vs CHW

**这是最难发现的一个。** `Sam3Processor.set_image()` 内部：

```python
elif isinstance(image, (torch.Tensor, np.ndarray)):
    height, width = image.shape[-2:]          # 记下 H, W
image = v2.functional.to_image(image)         # 但 to_image 要的是 CHW！
```

传 HWC numpy `(480, 640, 3)` 时，`to_image` 把它读成 `(C=480, H=640, W=3)`。
**不报错**，但输出 mask 形状变成 `(0, 1, 640, 3)`，**所有 prompt 都返回 0 个实例**。

症状是"模型跑得很正常，就是什么都找不到" —— 极易被误判成"模型不支持这个物体"。

**解法：传 PIL 图像**（无歧义）。`_Sam3Backend.set_image` 已做此转换。

### 10.6 BFloat16 无法直接转 numpy

SAM3 在 bf16 autocast 下推理，`boxes`/`scores` 是 bfloat16：

```python
TypeError: Got unsupported ScalarType BFloat16
```

numpy 没有 bf16 dtype，必须先 `.float()` 再 `.numpy()`。已封装为 `_to_numpy()`。

### 10.7 实测性能（H20，resolution=1008）

| 阶段 | 耗时 |
|---|---|
| worker 启动 + 模型加载 + 首帧 | **20.3 s**（一次性） |
| 单 prompt 推理 | **253–520 ms** |
| 同一帧第二个 prompt | **127 ms**（复用 backbone） |
| 跨进程传输开销 | 毫秒级，可忽略 |

真实照片（sklearn `flower.jpg`, 640×427）结果：

```
'flower'  -> 4 实例，top score 0.973，area 58717px，box [168.9, 84.9, 445.8, 359.3]
'blossom' -> 3 实例，top score 0.961
```

两个近义 prompt 识别出同一批物体、分数接近，说明推理稳定。

### 10.8 一条重要的测试教训

**合成图会骗人。** 最初用手绘的"瓶子"色块图测试，所有 prompt 返回 0 实例，
我一度怀疑是解析 bug。换真实照片立刻正常。

SAM3 是在真实照片上训练的开放词表模型，**平面矢量图里它找不到东西是正确行为**。
因此 `tests/gpu/test_sam3.py` 强制要求 `PERCEPTA_SAM3_PHOTO` 指向真实照片，
否则 skip —— 用合成的图"通过"测试会给出虚假的信心。

### 10.9 重复推理前必须 set_image

worker 崩溃后若被静默重启，新进程没有图像状态，会报
`set_image() must be called before segment()` —— 把矛头指向调用方代码。
已改为：`segment()` 检测到 worker 已死时直接抛 `BackendUnavailable`（spec §16.4）。

## 11. FoundationStereo 独立运行时

### 11.1 模型与资产

本节记录首个部署的 **FoundationStereo 23-51-11 / ViT-large**，不是 Fast-FoundationStereo。
选择器为 `stereo_backend: foundation_stereo`；后续 `ffs` 和原生三目部署见 §12。
官方研究权重限 **non-commercial research**，不随 percepta 分发。

- 源码：`NVlabs/FoundationStereo`，固定在 `6e8806816b533e4d13ddbb95ffa907b797060a62`。
- checkpoint SHA256：`60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`。
- 配套 cfg SHA256：`a9d9dd2137c30edc2236194f62df14d222dad5fd3287a33c7540b543bb93853f`。
- cfg 中 `max_disp=416`；使用 32 次更新，不套用 SGBM 的视差范围。
- 完整下载来源、辅助资产版本与校验值见 `uv_envs/foundation-stereo/assets.json`。

### 11.2 部署与离线加载

```bash
uv sync --extra stereo --locked
uv sync --project uv_envs/foundation-stereo --locked
uv_envs/foundation-stereo/.venv/bin/python scripts/bootstrap_foundation_stereo.py
uv run --extra stereo percepta models check --config configs/local-foundation-stereo.yaml
uv run --extra stereo python scripts/foundation_stereo_smoke.py --http
uv run --extra stereo percepta serve --config configs/local-foundation-stereo.yaml
```

启动前用 `nvidia-smi` 确认可用卡，并在 shell 设置 `CUDA_VISIBLE_DEVICES`。
配置中的空字符串表示继承环境，**不表示自动选择空闲 GPU**。
服务使用 127.0.0.1:8791；根默认 pipeline、SAM3 环境和系统 CUDA 均不改变。

上游初始化有两个隐式联网入口：timm 的 `edgenext_small(pretrained=True)`，以及
DINO 的远程 `torch.hub.load`。部署脚本预取并校验 EdgeNeXt 到模型专属 HF cache；
DINO 改用仓库自带源码，唯一修改的文件及补丁后 hash 也写入资产清单。
模型参数随后由完整 checkpoint 严格覆盖。worker 设置 HF offline 环境变量，
缺少缓存应明确失败，而不是请求时再下载。部署脚本可以重复执行；既有资产 hash
不一致或源码版本不符时拒绝覆盖。

### 11.3 双目协议和结果边界

SAM3 协议的单 RGB + mask 输出不适合视差，因此另设双目协议：两张等尺寸 RGB
uint8 输入，单张同尺寸 float32 像素视差输出。heavy imports 仅在独立 worker
执行，模型常驻，启动与通信均有超时，异常会回收自己的 worker，不自动换算法。

`context.stereo_left/stereo_right` 指定调用方选好的两路相机；工具不选择相机。
两幅图像先校正，原左图坐标的 `stereo_mask` 用最近邻同步重映射。`+Z` 为相机
前方，视差重建后经校正逆旋转和 `T_world_cam` 输出世界坐标。

没有 mask 时输出**整幅有效场景的可见点统计**，不是 prompt 指定物体的定位；
bbox/centroid 也不是对不可见表面的补全。不确定度公式中的视差噪声是配置假设，
不能解释成已测得的毫米级精度。缺权重/worker 失败通过 HTTP 200 返回
`status="unavailable"`；没有有效点时不伪造 Geometry。

通用 warmup 只有一张图，因此本配置关闭它。使用成对 smoke 观测真正的首帧、
热态调用和 HTTP `fused`，不能仅凭端口就绪判断双目推理已就绪。

### 11.4 实测记录（2026-11-11，H20）

官方 RGB 图像对为 960×540，原尺寸、32 次更新，`max_disp=416`、FP16 autocast；
Python 3.11.13、torch 2.4.1+cu124、timm 1.0.11、xformers 0.0.28.post1。
该固定源码实际使用 PyTorch SDPA / xformers，真实推理不需要额外安装或编译
`flash-attn`，因此锁文件未添加它。shell 设置 OMP 4 线程、OpenBLAS 1 线程。

| 阶段/指标 | 一次完整 smoke 的测量 |
|---|---|
| 校验资产、启动 worker、加载模型 | 13.39 s |
| 首次工具调用（不含上述加载） | 1.905 s |
| 第 2–3 次工具调用（含校正、传输、几何） | 0.828–0.831 s |
| 热态 GPU forward（CUDA event，不含传输/几何） | 643–644 ms |
| PyTorch 峰值 allocated 显存 | 3.70 GiB |
| 通过支持域/对应点过滤的像素 | 469,911 / 518,400（90.65%） |
| 可见点中位深度 | 0.507 m |

连续三次调用复用同一 worker，Geometry 一致。独立 GPU 测试通过上游
`run_demo.py` 原始 forward 路径对照像素视差（`rtol=1e-3, atol=1e-3`），并禁止
该参考进程的 socket connect，验证准备后的加载与推理不依赖临时联网。
这些数值证明运行与坐标一致性，不是物体定位精度报告。

研究 checkpoint 不只有 tensor，还带 NumPy 标量与 OmegaConf 训练元数据。
直接 `weights_only=True` 会拒绝它；worker 仅允许清单中实际需要的这些标准类型，
仍使用 `weights_only=True`，加载后清空 allowlist，不回退到无限制 pickle。
CPU 测试不加载权重；真实 GPU 测试须显式运行：

```bash
uv run --extra stereo pytest -m gpu tests/gpu/test_foundation_stereo.py -q
```

回归结果：默认环境（无 torch/OpenCV）483 passed、6 skipped；带 OpenCV 的 CPU
全套 595 passed、3 skipped；本模型真实 GPU 测试 2 passed。Ruff、既有 e2e smoke、
三个环境的 `uv lock --check` 和 lock portability 检查均通过。
真实 loopback HTTP 覆盖重复调用、空 mask 无 Geometry、缺 checkpoint 和主动终止
本测试 worker 后的 `unavailable`，并核对 HTTP 与直接工具路径的 Geometry 一致。

## 12. 其余三个神经模型的部署（2026-11-11）

四个调研模型现在均已部署。下表三个新运行时与 SAM3、FoundationStereo 分离，
所有 PyTorch/CUDA 依赖均留在各自环境，根默认依赖不变。系统 CUDA toolkit 没有替换。

| 模型 | 环境目录 | Python / torch / bundled CUDA | 本地配置 / 服务端口 |
|---|---|---|---|
| Fast-FoundationStereo 23-36-37 | `uv_envs/fast-foundation-stereo` | 3.12.13 / 2.6.0+cu124 / 12.4 | `configs/local-fast-foundation-stereo.yaml` / 8792 |
| MVSAnywhere HERO | `uv_envs/mvsanywhere` | 3.10.20 / 2.1.2+cu118 / 11.8 | `configs/local-mvsanywhere.yaml` / 8794 |
| DA3-BASE | `uv_envs/da3` | 3.12.13 / 2.11.0+cu128 / 12.8 | `configs/local-da3.yaml` / 8793 |

Fast-FoundationStereo、MVSAnywhere 保留上游兼容栈。DA3 使用官方 PyTorch previous-versions
页中最新的 cu128 组合（torch 2.11.0 / torchvision 0.26.0 / torchaudio 2.11.0）。
DA3 上游要求 NumPy<2，而 NumPy 1.26 wheels 不支持 Python 3.13/3.14，因此选择 3.12。

### 12.1 重建与启动

运行目录为本仓库根目录。每个运行时包含 `pyproject.toml`、`uv.lock`、`.python-version`
和固定源码/权重来源及 SHA256 的 `assets.json`。权重与上游源码放在 ignored 的
`checkpoints/`、`models/` 中；bootstrap 重跑时验证既有内容，损坏或版本不符时拒绝覆盖。

```bash
uv sync --project uv_envs/fast-foundation-stereo --locked
uv_envs/fast-foundation-stereo/.venv/bin/python scripts/bootstrap_fast_foundation_stereo.py
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run --extra stereo python scripts/fast_foundation_stereo_smoke.py --http --upstream

uv sync --project uv_envs/mvsanywhere --locked
uv_envs/mvsanywhere/.venv/bin/python scripts/bootstrap_mvsanywhere.py
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv_envs/mvsanywhere/.venv/bin/python scripts/mvsanywhere_smoke.py --repeats 3
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run python scripts/multiview_smoke.py --config configs/local-mvsanywhere.yaml --http

uv sync --project uv_envs/da3 --locked
uv_envs/da3/.venv/bin/python scripts/bootstrap_da3.py
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv_envs/da3/.venv/bin/python scripts/da3_smoke.py
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run python scripts/multiview_smoke.py --config configs/local-da3.yaml --http

# 选择一个服务常驻；根据 nvidia-smi 选实际空闲卡，端口默认仅绑定 loopback。
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run percepta serve --config configs/local-da3.yaml
```

`percepta models check --config <上述配置>` 可先检查主 checkpoint 哈希。
工具加载还校验模型配置；bootstrap 校验源码及离线补丁。模型启动时禁止临时 HF 下载。
通用单图 warmup 在这些配置中关闭，成对/三视图 smoke 才能证明推理已就绪。

### 12.2 接口与尺度

Fast-FoundationStereo 由 `stereo_backend: ffs` 选择，复用已有校正和双目传输。
官方模型是结构化剪枝后的完整 pickle；仅在独立 worker 校验清单中的固定官方
checkpoint SHA 后加载。实际测试 `max_disp=640` 会因 26-token 位置编码维度不足失败，
因此默认并强制限制为 **416**；未插值位置编码或修改模型。它不覆盖原计划中约
600 px 的近距 LR 情况，需调整输入分辨率/基线/工作距离后另行验证。

MVSAnywhere 由 `triview_method: mvs`、`triview_backend: mvsanywhere` 选择。
官方内存 API 支持数组，无需逐请求写目录。每个参考视图联合使用另两路图像，
输出三张深度图。OpenCV camera-to-world 与该入口一致；内部只消费相对相机变换。
先在 float64 中计算 `inv(T_world_cam[0]) @ T_world_cam[i]`，再转 float32，
消除任意世界原点带来的相消误差。输出仍为各原相机 +Z 米制深度，最终世界点
使用调用方的原始 `T_world_cam`。天空概率只用于排除天空，不能冒充深度置信度。
本地 DINO/ResNet 初始化不再临时下载，完整 HERO 参数随后严格覆盖全部网络权重。

DA3-BASE 由 `triview_method: mvs`、`triview_backend: da3` 选择。
先在 float64 中计算 `T_local_cam[i] = inv(T_world_cam[0]) @ T_world_cam[i]`，
上游接收 `inv(T_local_cam)`。这样不会因大世界坐标转 float32 丢失小基线。
使用 `align_to_input_ext_scale=True` 将深度绑定到输入标定的米制基线；
三相机中心必须非共线。严格 safetensors `load_model` 处理
官方文件中的共享 LayerNorm alias，没有使用 `strict=False`。输出深度/置信度
重采样回原图坐标，置信度保持原始非负分值，不解释成概率或定位误差。
在发送请求前检查缩放后是否至少保留一个 14 像素 patch；不支持的长宽比返回
`invalid_request`，不进入 worker，不影响后续正常推理。

FFS/MVSAnywhere 所需的资产清单也随 wheel 发布，已脱离源码树测试模型加载；
editable checkout 继续使用 `uv_envs/` 中的同一份清单。各运行时仍单独提供源码和权重。
后端提供不等待推理锁的故障快照，注册表 introspection 将已失败或已退出的 worker
标记为 `error`，`/healthz`、`/v1/tools` 和 `/v1/info` 因而反映实际故障。

三视图协议只传 RGB、K、`T_world_cam` 和有限 float32 深度/置信度；不传 pickle，
不接受请求指定 Python 路径、源码、checkpoint 或 factory。重复/退化相机标定
在进入模型前拒绝，worker 死亡/超时为 `unavailable`，不会静默替换算法。
没有有效深度时不输出 Geometry。所有几何均描述可见表面；文本 prompt 本身不选物体。
三目神经几何的 0.03 m uncertainty 是未校准的占位假设，结果显式标注警告。

### 12.3 本机真实验证

H20 上已验证 CUDA tensor、权重加载、重复推理、常驻进程复用、工具世界几何和
真实 loopback HTTP；主动终止各自 worker 后返回 `unavailable`。以下为单次记录，
输入/分辨率不同，不能据此声称模型精度或速度排名。

| 模型与输入 | 热态记录 | 峰值 allocated 显存 |
|---|---|---|
| Fast-FoundationStereo，官方 960×540 双目，8 iters / 416 disparity | 工具 286–289 ms；GPU forward 105–106 ms | 2.02 GiB |
| MVSAnywhere，官方三张 640×480，逐参考联合另两路，1 refinement | 三路深度 adapter 483–486 ms；三次 GPU forward 合计 476–479 ms | 1.47 GiB |
| DA3-BASE，252×336 三路标定平面，处理为 378×504 | 工具 290 ms；adapter 含处理/传输 230 ms | 1.10 GiB |

Fast-FoundationStereo 视差与禁网下的官方 demo **逐像素相同，最大差 0.0 px**。
MVSAnywhere 与官方内存入口在相同首相机归一化输入上数值一致；共同旋转/平移
世界坐标后，相机归一化最大差 8.88e-16，实际深度最大差 **0.0 m**。
DA3 重复结果一致，输入基线翻倍后深度比为 **2.0**。其合成 1 m 平面预测中位
深度约 1.102 m，因此尺度绑定验证不能解释成定位精度达标。

详细 JSON 证据在本机 `checkpoints/{fast-foundation-stereo,mvsanywhere,da3}/`，
不随权重提交。可复现 GPU 检查（每次为对应模型选择空闲 GPU）：

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run --extra stereo pytest -m gpu tests/gpu/test_fast_foundation_stereo.py -q
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run pytest -m gpu tests/gpu/test_mvsanywhere.py tests/gpu/test_multiview_tools.py -k mvsanywhere -q
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 uv run pytest -m gpu tests/gpu/test_da3.py tests/gpu/test_multiview_tools.py -k da3 -q
```

六个锁文件的 staleness/official-host 检查通过；CI 现在遍历所有模型锁，并另外
在含 OpenCV 的 CPU 环境运行回归。基础环境依然不含 torch/OpenCV。
本轮修复后带 OpenCV 的 CPU 回归为 **661 passed、9 skipped**，Ruff 与既有 e2e 通过；
部署初验中三个新模型的 **6 项真实 GPU 测试通过**，原 FoundationStereo 的 2 项 GPU 回归也通过。
本次修复另跑 DA3 的 2 项真实 GPU/HTTP 回归，覆盖大世界原点、拒绝尺寸后复用 worker
及故障健康状态；独立 wheel 环境完成 FFS 真实推理和 MVSAnywhere 严格权重加载。
另建不含 torch/OpenCV 的默认环境跑过完整 CPU 回归，并补验最后的可选依赖错误分支。
Fast-FoundationStereo 与 MVSAnywhere 的研究权重仅供非商业研究，DA3-BASE 为 Apache-2.0。
本轮交付是模型部署与工具接入，不包含 RoboDojo 的目标定位精度、Policy SR/Score 实测。

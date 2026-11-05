# 模型隔离与依赖管理

> 状态：v0.1（设计阶段定稿）。本文档回答一个问题：**当两个感知模型要求互斥的依赖时，
> 怎么在不引入容器的情况下让它们共存。**
>
> 相关：`percepta-SPEC.md` §5（工具协议）、`AGENTS.md` §1（uv 工作流）、
> `docs/decisions.md` D22–D26。

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
    python: .venvs/sam3/bin/python   # subprocess 后端
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

## 5. 目录与依赖约定

### 5.1 每个模型一个隔离环境

```
.venvs/                      # gitignored
├── sam3/                    # torch>=2.7 + sam3==0.1.4
└── depth/                   # torch==2.6 + depth-anything-v2
```

创建方式：

```bash
uv venv .venvs/sam3 -p 3.10
uv pip install --python .venvs/sam3/bin/python -r requirements/sam3.txt
```

### 5.2 主环境保持干净

主环境（`.venv`）**永远只装 `pyproject.toml` 的默认依赖 + `dev` 组**，
不含任何模型框架。模型依赖写在 `requirements/<model>.txt`，供隔离环境安装。

这条规则保证：主环境可以在无 GPU 机器上安装、跑全量测试。

### 5.3 为什么不用 extra 装进主环境

`optional-dependencies` 仍然保留（用于只有一个模型、无冲突的简单场景），
但**一旦出现冲突，extra 就无法表达"这两个不能同时装"**。
`requirements/<model>.txt` 是冲突场景的唯一出路。

## 6. 子进程通信协议（草案）

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

## 8. 加一个新模型的检查清单

1. **判断冲突风险**：新模型要求的 `torch`/`numpy` 主版本与现有模型是否相容？
   - 相容 → 走 extra，`inprocess` 后端
   - 冲突 → 建独立 venv，`subprocess` 后端
2. **写 `requirements/<model>.txt`**，版本钉死（不用 `>=`）
3. **在 `configs/*.yaml` 里声明 backend 与 `cuda_visible_devices`**
4. **工具实现只依赖 backend 抽象**，不直接 `import torch`
5. **写一个 fake backend**，让逻辑能在无 GPU 环境测试（沿袭 `Sam3Tool(backend_factory=...)`）
6. **权重走配置注入**，绝不进 git / 镜像
7. **验证 `/healthz` 如实报告**：未装的模型应显示 `error`，不能显示 `ok`

## 9. 未决问题

1. **worker 是否复用**：当前设计是"每模型一个常驻 worker"。如果同时有 5 个模型，
   就是 5 个常驻进程。是否需要按需启动/空闲回收？建议 v0.2 再定。
2. **图像零拷贝**：目前设计走管道复制。`torch` 支持共享内存（`/dev/shm`），
   但增加复杂度。当前 `480×640` 复制约 1 ms，可接受，暂不优化。
3. **多个 prompt 的批量**：协议草案是一帧一 prompt。批量能减少往返，
   但需要重新设计帧格式，v0.2 再考虑。
4. **worker 的健康监控**：主进程如何发现 worker 卡死（而非崩溃）？需要心跳或超时。
   当前依赖请求级超时（spec §6），可能不够。

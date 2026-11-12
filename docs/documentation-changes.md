# 文档修订记录

> **当前状态（2026-11-12 对 `9779a96` 再复查）**：D39坐标修复及D40–D43的20组改动已提交，
> 本轮另确认4项P2与1项P3待修，含回归及既有遗漏。“20组全部关闭”是上一批次的结论，现已收窄。
> 详见[复查与详细修复报告](reviews/2026-11-12-runtime-fix-review.md)及本页末尾；各批次原文保留。

## 2026-11-12：对齐已部署实现，保留历史原文

- **修改日期**：2026-11-12，Asia/Shanghai（UTC+08:00）。
- **记录整理时间**：2026-11-12 10:33（UTC+08:00）。
- **修改前基线**：Git 提交 `534066a`，`feat: deploy isolated stereo and three-view model runtimes`。
- **目的**：使当前使用说明与已验证实现一致，区分历史设计、现行行为及尚未实现的计划。
- **版本**：规格书文档修订号 `v0.3.0-doc.1`；包仍为 `0.3.0`，schema 仍为 `percepta/v1`。
- **记录方式**：下表逐项摘录旧文关键内容、说明替换或补充内容及原因。完整旧文由上述 Git 基线保留；历史决策和被撤回的 A1 说明还在原文旁保留，并附日期和替代关系。
- **镜像**：根 `percepta-SPEC.md` 为主本，`docs/spec.md` 同步相同内容。

查看任何文件的完整旧文或本次差异：

```bash
git show 534066a:percepta-SPEC.md
git show 534066a:docs/decisions.md
git diff 534066a -- percepta-SPEC.md docs/spec.md README.md configs src/percepta docs AGENTS.md CHANGELOG.md
```

本批次没有升级模型、修改几何计算、加入新接口或实施 RoboDojo 接入／评测。
可执行变化限于 SAM3 缺依赖时的安装提示，以及示例配置改用已经部署的独立运行时；具体记录如下。

### 规格书与协作约定

下列规格书修改同时作用于 `percepta-SPEC.md` 和 `docs/spec.md`。

| 位置 | 修改前原文／缺项 | 修改后与原因 |
|---|---|---|
| 标题及阅读说明 | `v0.1 draft`；“再读……§16 验收” | 文档修订号改为 `v0.3.0-doc.1`，注明日期、代码基线及镜像；验收实际在 §14，§16 是协作规约 |
| §1 提示与非目标 | “文本 / 框 / 点 三种提示”；跟踪“可在 v0.2 加” | 区分 schema 可表达的类型和 SAM3 实际 text/box 能力；跟踪仍为未来项，不把旧版本号当成交付承诺 |
| §3 Observation／Provenance | “稳定 id：sha1(percept_id + index)”；可空字段未写默认值；latency/created_at 无默认值 | 写清仅单个 Percept 生命周期内稳定及真实哈希格式，补实际默认值 |
| §3 Geometry | 止于 `views_used`，缺后续可选字段 | 补 D35 已有的 lateral/height uncertainty、view disagreement、method；未更改 schema 本身 |
| §3／§5 fuse | `fuse: bool = True # 多视角 -> 3D`；`fused` 注释称必须 options.fuse | 精确说明 pipeline 不跳过显式 geometry；multi_view_3d 自身遵守 False，stereo/triview 仍可返回几何 |
| §4 输入几何 | “假定已校正”；“没有深度时几何只能靠轮廓/平面假设” | 区分镜头去畸变与双目极线校正，说明神经双／三目可从标定 RGB 估计深度 |
| §5 ToolContext／工具协议 | 无 extras、logger/cache 默认值；`PerceptionTool(Protocol)` | 补已经实现的 extras 和默认值，标为实际 ABC，未增加新协议字段 |
| §5.3 编排 | “每相机独立调用工具” | 按 D6 每请求每工具一次，传全部相机；补 multi_view_3d 后置、后成功覆盖、后失败保留及无自动质量回退 |
| §5.4 工具清单 | 只有 fake、sam3、multi_view_3d，后续工具写“可选（v0.2）” | 加已部署 stereo/triview 和真实后端、视图约束、FFS 416 上限；未实现的 adaptive/目标 mask 接线仍明确为未实现 |
| §6／附录 A | `unsupported_prompt` 的 HTTP 列为 `400（Percept.status）` | 改为 HTTP 200 中的 invalid_request Percept；请求级坏字段仍 HTTP 400，遵守现有 D1 |
| §7 SDK／CLI | 本地调用“直接调 registry”；无 readiness、models check | 写清共用 PipelineRunner；补现有 readiness 签名与“失败也算结束”的边界，补已有 CLI |
| §9 轮廓算法 | “匹配最好且高度最小”；未解释采样柱体、分轴不确定度 | 按 D35 描述整柱采样约束、完整 IoU 曲线峰值、阈值仅控制警告，补真实 fallback 和桌面假设 |
| §9 单视角 | “只返回 2D + 粗略高度”；uncertainty ≥0.03 | 说明实际桌面接触射线与假定高度形成粗略 3D，当前高度估计下限 0.05 m；无明确桌面信息返回空 |
| §9 hang／uncertainty | 未记录本轮新发现的代码限制 | 明确不是完整对象补全或严格误差界；下文单列两项未修问题，不把文档修订写成算法修复 |
| §11 配置与缓存 | 无 warmup；GPU 配置／缓存 key 只作概述 | 补已有 warmup、空 GPU 值继承语义和无自动选卡；写明 key 未含 context/标定/模型配置，依赖这些输入的工具应关闭缓存 |
| §12 安全 | “不做 pickle/任意反序列化” | 区分不可信请求协议与独立 worker 内的哈希固定官方权重加载；FFS 官方 pickle 不可由请求替换 |
| §13 目录与依赖 | `extras: [service], [sam3], [dev]`；`pydantic-settings + YAML`；“不维护任何模型框架的依赖清单” | 写实际默认依赖、service/stereo extras、dev 依赖组、pydantic+PyYAML、五套独立锁定环境及当前文件结构 |
| §15 里程碑 | M5 包含 Docker；M0 用 `pip install -e .` | 说明 Docker 已按 D27/D33 移出交付范围，使用 uv；区分两套重名 M0–M5 |
| §17 性能表 | 标为“性能目标……仅参考” | 明确为历史目标而非现行实测承诺；保留原数字，引用分模型的实测记录 |
| §18／§19 | 最新只到 v0.2.2；D4/D16/D23/D27 易被当成当前规则 | 补 D35–D38 对应变更和替代关系；历史决策行保留，现行结论另表说明 |
| 附录 C SAM3 | `build_sam3_image_model(device, checkpoint_path, ...)`；`Sam3Processor(resolution=1008)` | 改为全关键字模型构造、必须传 model 的 processor、PIL 输入、必须传 label 的框提示与 BF16 输出处理 |
| AGENTS.md | 无本次修订入口；fuse seam 注释返回 Percept 而未列 None；DoD 直接用系统 python3 | 加日期、镜像及修订记录要求，返回注释与现有函数一致，澄清编排／HTTP 语义，检查命令统一 uv |

### 使用说明、示例配置与源码说明

| 文件／位置 | 修改前原文／缺项 | 修改后与原因 |
|---|---|---|
| README 状态与环境 | `uv.lock ... pins Python 3.10`；`full contract suite` | Python 由 .python-version 选择；默认 CPU 测试与可选 OpenCV／GPU 测试分别说明，增加当前状态入口 |
| README 总体定位 | `never by guessing` | 粗略估计必须标注假设与警告，无有效证据不给几何；不作所有几何都无假设的绝对承诺 |
| README 安装 | `dev` 被列为 extra；`unknown keys are rejected` | dev 是默认依赖组；section 严格校验，tools 字典参数由具体工具解释；支持的安装流程统一 uv |
| README 预热 | SAM3 16–70 s／137–530 ms 没有测量日期；泛称启动会预热 | 加 2026-11-05 H20 的历史范围及 enabled 条件；双／三目必须运行对应 smoke，端口可访问不保证已热态 |
| README 示例与编排 | SAM3 示例无 BPE/独立解释器；无多 geometry 与 fuse 例外说明 | 补独立环境、资产路径、能力范围与实际编排语义 |
| README 几何与新增模型清单 | “height search per D4”“single-view 2D + coarse height”；checksum“never from repository”；未安装模型必须报 error | 改为 D35 峰值与粗略单视角 3D，列当前 0.05 m 高度估计下限及两项未修限制；资产清单可跟踪、权重不可跟踪；区分 lazy 与已尝试加载失败 |
| README 隔离 | 虚构 `uv_envs/depth`；subprocess 固定 `0.01 s`；“workers never fight over VRAM” | 改已部署 FoundationStereo 配置节选；启动含模型加载；不承诺自动选卡或显存预留 |
| README 验证 API | `verify()`；示例 `assert result.checks["lifted"] and result.ok` | 使用真实 compare API 并检查返回证据，不从像素位移单独推断 lift |
| README 结构／部署／测试 | 工具目录遗漏 stereo/triview；权重“never copied into repository”；GPU 测试仅注释 SAM3 | 补当前目录与五模型测试命令；权重可在 ignored checkpoints 但不能 Git 跟踪／打包；注明 systemd 需配置用户和路径 |
| README 许可 | 主要列 SAM3 及未实现的 Ultralytics | 对齐当前五套已部署模型的实际许可范围 |
| configs/example.yaml 头部 | `Requires the sam3 extra` | 加 2026-11-12 日期、独立环境重建命令和权重/BPE 前置要求；根 extra 已删除 |
| configs/example.yaml SAM3 | `backend: inprocess`，无 bpe_path；“workers never contend for VRAM” | 示例改为 subprocess，沿用已存在的独立 python 路径，增加 `/models/bpe_simple_vocab_16e6.txt.gz`；删除无法保证的资源承诺 |
| configs/local-sam3.yaml | “Leaving it empty lets PyTorch pick the card with the most free memory” | 明确空值继承环境，PyTorch 不自动挑选最空闲卡；保留此前冷启动记录作为历史测量 |
| tools/sam3.py 模块说明 | 与旧附录 C 相同的错误位置参数签名和未传 model 的 processor | 对齐实际关键字调用、processor model 参数与 PIL 输入 |
| tools/sam3.py 异常提示 | `install the 'sam3' extra and provide a checkpoint` | 指向 `uv sync --project uv_envs/sam3 --locked`、subprocess 解释器与部署文档；异常类型和失败语义不变 |
| schema.py method 注释 | `None for the depth/single-view paths` | 列出已存在的 single_view/stereo_*/triview_*，仅部分 depth 路径为空；未改任何字段或默认值 |
| fusion.py 模块／helper | “smallest h whose mean mask IoU reaches threshold wins”；uncertainty floor 似为全局保证 | 生产路径为 sampled-column 峰值；旧 helper 仅标为历史；区分 depth 噪声下限与各几何估计 |
| fusion.py hang／lateral 说明 | 声称 hang 满足全部视角约束；lateral 是投影误差界 | 如实标注下述已确认问题与日期，未更改算法；height uncertainty 实际仅按低 IoU 放大 |
| tools/multi_view_3d.py capabilities | `needs depth and many views` | 深度可选，有桌面时单视角可给粗略几何；补可执行示例 |

### 设计、调研与部署历史

| 文件／位置 | 修改前原文／旧判断 | 修改后与原因 |
|---|---|---|
| decisions D4／D16 | first threshold、mid-height cone slice | 原文保留，日期说明已由 D35 的完整峰值和整柱约束替代 |
| decisions D8／D35 | dev 被称作 extra；D35 后没有本轮发现的计算问题提示 | 保留旧句并补当前 dependency group 术语；D35 增加本轮两项待修坐标问题入口 |
| decisions D15 | `a geometry stage that yields nothing gives fused=None`，未解释多个阶段 | 旧句仅适用于此前没有成功结果；补后成功覆盖、后失败保留和各工具 fuse=False 差异 |
| decisions D27 | 仓库不维护模型 dependency manifests | 原文保留，标记该部分由 D33 的独立 uv 项目替代；资产完整性约定继续有效 |
| decisions D36 | `ffs, still unavailable`；`gains no neural model in this deployment` | 原文作为当时阶段保留，注明现已由 D37/D38 部署和修复补全 |
| model-isolation 状态／后端 | “设计阶段定稿”；0.01 s 启动；先 inprocess 试冲突 | 区分历史建议与五套实际 subprocess 配置；0.01 s 仅解释器启动，不包含模型加载 |
| model-isolation 依赖／目录 | “不管模型依赖”；不存在的 depth 目录；主环境“永远”仅默认＋dev | 列五套实际独立锁；根可装 service/stereo；运行时定义提交，大资产忽略 |
| model-isolation 资产 | checkpoint“绝不在仓库里”；if-declared“生产配置天然安全” | 可以使用 ignored checkpoints；仅声明哈希后才真正比对，未声明只记录 |
| model-isolation 协议／并发 | 历史帧布局像当前规范；per-tool lock 被描述为全 GPU 互斥 | 标明草案，现行三种协议以源码为准；锁仅按工具，不提供全局显卡调度 |
| model-isolation 安装记录 | §10 旧手工安装命令容易照抄；健康监控仍列为待做 | 标为历史日志，当前重建用各项目锁；D38 已提供 worker 故障快照，未实现项仍保留 |
| model-survey 总体 | “只做信息核实，没有安装……”；“需要你决定的” | 作为部署前快照保留，增加当前四模型已部署说明，不再把历史问题当成本轮待决定事项 |
| survey FoundationStereo／FFS | 必须 flash-attn；max_disp 可直接设约640 | 用实测纠正；固定源码不需要 flash-attn，官方 FFS 剪枝 checkpoint 上限416 |
| survey MVSAnywhere／DA3 | 磁盘目录/conda 路线；尺度待处理；confidence 直接作为 uncertainty | 当前是官方内存 API、独立 uv、已知位姿尺度对齐；confidence 未校准，固定0.03 m仅为已声明假设 |
| survey 误差数字 | “实测的 0.5–1.7 mm” | 说明是视差噪声假设下的几何估计，不是模型定位误差测量 |
| implementation plan A1 | “问题已在 D15 修复”；“第一个产出 ok 的生效，后续的被忽略并记录警告” | A1 原块完整保留并明确撤回；前置现行说明为后成功覆盖，真实严格双／三视图契约及未实现的自动化分别说明 |
| implementation plan 范围 | 两组 M0–M5 易混淆；640视差的错误解释 | 区分历史扩大规划与当前交付，注明本轮不实施外部接入／评测；保留原640建议并解释416实测限制 |
| localisation-explained 来源 | 图中数字“算出来的”，未区分生成时间、合成条件与通用结论 | 注明图文随 `267ad92` 于2026-11-10提交（不冒称生成时刻），2026-11-12对齐代码；保留所有旧图和数字，未重生成或测量 |
| localisation-explained 误差说法 | “精度是毫米级”“高度1–3 mm” | 原句保留并注明是假设／特定合成示例，不能作为当前模型通用定位误差或 uncertainty 保证；新增两项待修问题入口 |
| localisation-explained 高度搜索 | “只有一个高度……”；投影到“其他”相机 | 实际对全部参与视角取平均 IoU，柱体沿高度有限采样；峰值可能不唯一，取离散最高分，并列取较低者 |
| localisation-explained 输出名 | “PyRUA-Lean 的 views 字段” | 旧句保留作为错误引用，当前为 `percepta.Geometry.views_used`；单视角标注粗略假设 |
| localisation-explained 对向视角 | “两个相机看的是物体的同一面……等价于一个相机” | 明确撤回该几何断言；相反方向并非同一面，特定脚本误差不能推广为一般单目等价 |
| localisation-explained 轮廓限制 | “凹体（碗、杯子、L形零件）……无论多少视角都会膨胀” | 原句保留并更正：限制的是所有允许视角始终不可观测的结构，并非所有凹体或L形都不可恢复 |
| CHANGELOG.md | 无本轮记录；decisions 相对链接为 `decisions.md` | 在 Unreleased 加日期、修订入口和未修问题说明，修正链接；不重写旧版本发布记录 |

### 核对中新确认的代码问题：尚未修复

以下为提交 `5156672` 时的原始审计记录；后续修复状态见本页末尾，原文与复现数字保留。

这两项属于既有 `multi_view_3d` 轮廓路径，本次只修正说明并登记，不改计算，也不声称测试通过就代表问题已解决。

| 编号 | 位置与证据 | 当前影响与状态 |
|---|---|---|
| DOC-20261112-01 | `fusion._hang_cells()` 将 ray/plane 求得的世界 XY polygon 直接传给要求像素坐标的 `geometry.polygon_mask()`，未按 grid.lo/grid.cell 转换；视图间做 OR，prism_cells 未使用。最小坐标检查中，同一方形直接进入 helper 为0 cells，先转网格坐标后为110 cells | hang 候选可能空或错位；原说明中的全部视角约束也未实现。生产 _fuse_prism 可调用此路径，待专门修复和回归 |
| DOC-20261112-02 | `fusion._lateral_uncertainty()` 的 `norm(hit[0] - origin[0])` 中 origin 为 shape(3,) 光心，origin[0] 只是 x 标量。固定相对几何、统一平移世界原点后，输出从0.0055 m变为2.241515202 m | 横向估计及其参与的总 uncertainty 会受任意世界原点影响；待修正距离公式并补坐标平移回归 |

以上最小检查不运行模型、不评估任务精度，也不改变现有模型部署结果。

### 验证与范围

2026-11-12 在当前工作区执行的结果：

- 两份规格书逐字一致；规格书与 README 的 13 个 Python、8 个 Bash、3 个 YAML 摘录语法检查通过；15 个文档模型字段集合与实际 schema 完全一致。
- 最终检查的 34 个本地文档／图片链接目标存在；example/local-sam3 配置通过 AppConfig 加载。README 中保留的 `/models/` 路径是部署方需准备的资产，不在本次下载。
- fusion/schema/multi_view_3d 的 AST 去掉 docstring 后与基线一致；sam3 也仅安装诊断字符串不同。local-sam3 YAML 的解析值与基线完全一致；example 模板 backend/BPE 的变更已在上表记录。
- 新 capabilities 文档示例的 1 项 doctest 通过。
- `uv run --locked pytest -o addopts= -q`：**661 passed，9 skipped**，42.68 s；有 1 条既有 Starlette/httpx 弃用警告。
- `uv run --locked ruff check .`、`uv run --locked python scripts/e2e_smoke.py`、`git diff --check` 通过。

这些是本次执行结果，不用历史模型测试计数代替本次检查；CPU 回归通过不覆盖上面两个新发现的坐标问题。

未重跑 GPU、未下载或更换模型资产、未执行长期负载验证；上述两个计算问题仍待修复。

## 2026-11-12：修复两处轮廓几何坐标计算（D39）

- **修改日期**：2026-11-12，Asia/Shanghai（UTC+08:00）。
- **记录整理时间**：2026-11-12 11:00（UTC+08:00）。
- **修改前基线**：`5156672`，`docs: align specifications and deployment guides with implementation`。
- **授权与范围**：用户要求重新确认后授权修复，范围是以下两个已复现的坐标计算错误。
- **状态**：`DOC-20261112-01`、`DOC-20261112-02` 的坐标错误已修复；前一批次的审计原文保留。

| 文件／位置 | 修改前原文或计算 | 修改后与原因 |
|---|---|---|
| fusion._hang_cells | `geo.polygon_mask(ring, (grid.ny, grid.nx))`；ring 来自射线与世界平面求交，是以米计的世界 XY | 先按 `(ring - grid.lo) / grid.cell` 转网格像素，再栅格化；grid.lo 已是第0格中心，无需额外半格偏移 |
| fusion._hang_cells 的取样 | 栅格化后又用 grid.xy 推回整数行列索引取值 | 输出已是同一网格的行优先排列，直接使用 raster.ravel；删除重复取样但保留视图间 OR 合并 |
| fusion._lateral_uncertainty | `np.linalg.norm(hit[0] - origin[0])`；origin 为 shape(3,) 光心，origin[0] 只是 x 标量 | 改为交点减完整光心向量，消除不确定度对任意世界原点的依赖 |
| 回归测试 | 原有测试未覆盖以上坐标换算及共同平移场景 | 增加实际相机射线／网格坐标测试和公开 fuse 路径的平移不变性回归；先确认原实现失败，再验证修复通过 |
| README／spec 两份镜像／图解 | “two confirmed coordinate issues awaiting repair”；“尚未修复”“坐标限制”“世界原点依赖问题” | 当前说明改为 D39 已修复并给正确坐标公式；规格文档修订号变为 v0.3.0-doc.2，包版本与 schema 不变 |
| decisions／CHANGELOG | D35 的审计补记及前一批 Documentation 中记录为 pending | 原记录保留并追加同日修复状态；新增 D39 和 Unreleased Fixed 条目，不把历史批次改写为当时已修 |

### 修复前的独立复核

- hang：11×11 网格、间隔0.01 m、实际针孔相机射线；相对场景不变，世界 XY 平移100 m。
  原 helper 的候选格数从1变0；转换坐标后的对照与直接投影网格点检查均为81格。
- lateral：已有10 cm立方体三视角 fixture，使用公开 `fuse()`。世界 XY 平移 `(10, 20)` m 后，
  原 lateral／总 uncertainty 从0.020901 m变为0.229853 m；使用完整光心的同公式对照均为0.013827 m。

### 验证结果

先只增加测试，保留原始计算，执行：

```bash
uv run --locked pytest -o addopts= tests/unit/test_footprint.py \
  -k 'hang_top_shadow_uses_plane_grid_coordinates or prism_fusion_is_invariant_to_world_translation' -q
```

结果为 **3 failed、18 deselected**：两个原点下的 hang 格子与直接投影对照一致率仅约0.230；
公开 fuse 的 lateral 在共同 XYZ 平移前后分别为0.00776237 m与1.82305398 m。
这确认测试能捕捉原错误，不只是跟随修复后的实现。

修复后同一命令为 **3 passed、18 deselected**。新增测试位于 `tests/unit/test_footprint.py`：

- `test_hang_top_shadow_uses_plane_grid_coordinates`：两个原点，使用真实轮廓射线与独立相机投影对照。
- `test_prism_fusion_is_invariant_to_world_translation`：公开 fuse 路径，共同 XYZ 平移并同步 table_z，检查质心相对位置、顶部、选中高度、方法和分轴／总 uncertainty。

既有几何回归：

```bash
uv run --locked pytest -o addopts= tests/unit/test_fusion.py \
  tests/unit/test_footprint.py tests/unit/test_geometry.py -q
```

结果为 **70 passed**，18.57 s，未放宽旧断言。独立复核确认非方形网格的 row-major 展平也与
`grid.xy()` 一致；三维共同平移后的直接距离估计保持不变。Ruff、真实 HTTP e2e、两份规格镜像、
35 个本地文档／图片链接及差异格式检查通过。

完整 CPU 回归 `uv run --locked pytest -o addopts= -q`：**664 passed、9 skipped**，46.88 s；
有1条既有 Starlette/httpx 弃用警告。三项新增回归计入通过数，未运行 GPU 模型或更换资产。

### 保留的方法边界

hang 仍按各视角候选做并集，并使用原有 IoU 改善门限筛选；未实现旧注释中所说的
“与 prism_cells 联合并满足全部视角约束”。本次不把这些设计意图扩成新功能，也不承诺完整对象补全。
uncertainty 仍为启发式估计，修复距离公式不意味着完成误差标定。未运行 GPU 模型或任务精度评测。

## 2026-11-12：关闭全盘审计的20组问题（D40–D43）

- **修改日期**：2026-11-12，Asia/Shanghai（UTC+08:00）。
- **记录整理时间**：2026-11-12 12:10（UTC+08:00）。
- **最终验收整理时间**：2026-11-12 12:20（UTC+08:00）。
- **修改前基线**：`a761656`；本表编号对应在该基线上完成的全盘审计。
- **授权**：用户要求按照审计建议修复；第19项单独确认“Options 作默认，Prompt 显式值优先”。
- **版本**：规格修订号 `v0.3.0-doc.3`；包 `0.3.0`、schema `percepta/v1`、模型线协议 v1 不变。
- **范围**：修复审计中的20组运行时／接口问题，补回归与现行文档；不新增默认依赖，不改变模型资产。

完整旧源码、测试及文档可从 Git 复查，例如：

```bash
git show a761656:src/percepta/tools/sam3.py
git show a761656:src/percepta/service/routes.py
git diff a761656 -- src/percepta tests README.md percepta-SPEC.md docs CHANGELOG.md
```

### 逐项修改前后对照

下表全部已修复。“原行为”描述修复前基线，不能作为当前代码的待办清单。

| 审计项 | 原代码／行为及影响 | 修复后行为与原因 |
|---|---|---|
| 1 · SAM3 旧图复用 | 大图 `_image_key` 使用 `id(image)`，地址复用或原地更新缓冲区后仍用旧特征；真实连续帧会误报 not_found | 删除跨调用图像身份缓存；每次 perceive 为每相机准备一次图像，同次多提示仍共享特征。完整请求命中仍由父进程 PerceptCache 跳过推理 |
| 2 · SAM3 提示污染 | 官方 processor 的 box 提示追加、text 不清 box；box→box→text 会携带1框／2框／文本+2框 | 每次独立 segment 前执行原地 `reset_all_prompts(self._state)`，保留图像特征、清除语言／几何提示及预测；不把其 None 返回值赋给 state |
| 3 · 通用 worker 无期限 | `timeout_s` 只保存，启动／管道写入读取可永久阻塞 | 复用已有 `_DeadlinePipe`，启动与每次完整写入+读取 RPC 各有单调时钟预算；超时清理并回收 worker，不静默重启；stderr 持续有界排空，close 的 IO 也有界 |
| 4 · HTTP504 释放仍计算的容量 | handler finally 释放 semaphore／inflight，但 `to_thread` 线程仍执行；6个超时可留下6线程且报告0 inflight | 强引用 pending task，shield 防止 HTTP 超时／取消传播到托管任务；真正结束才由回调一次性释放容量、收集异常。shutdown 停止准入，等待请求和预热，再关闭 runner |
| 5 · 首次加载并发／失败泄漏 | 单例检查、构造、load 没有同一锁，首次并发可加载两份模型；capabilities 失败还可能遗失已加载资源 | 用已有 entry RLock 覆盖检查至发布；capabilities 成功后才发布实例。失败尽力 close，即使 close 再报错也保留原始加载错误 |
| 6 · 非物理标定生成伪几何 | schema 只检查形状；全零 T 能产生 ok、零坐标及0.001 m uncertainty，奇异 K 可能 HTTP500 | NumPy-only 共享验证供 schema／直接 geometry／fuse 使用：正焦距、零 skew pinhole K，有限 proper rigid T。HTTP坏矩阵在请求验证返回400；直接调用明确 ValueError。R 容差1e-6、底行／skew容差1e-8 |
| 7 · geometry 失败后擅自融合 | 未加载 entry.kind 为 None，失败工具被视为 segmenter，继而运行未选择的 multi_view_3d | 使用 `entry.describe()` 的类声明 kind；geometry 加载失败仍属于已选择阶段，保留 unavailable，不执行补充 mask 融合 |
| 8 · SAM3 prepare 逸出500 | prepare 位于 segment 异常映射之外，启动失败的 BackendUnavailable 直接 HTTP500 | 在每相机准备边界转换为 ToolUnavailable，进入既有 unavailable Percept 语义 |
| 9 · 坏协议伪装为 not_found | 非法状态可解码，backend 当作空结果；完整帧尾随垃圾也被接受 | v1 解码严格检查阶段允许状态、字段／重复 JSON key、错误组合、数组形状和有限性、完整帧消费；ready 仅限启动。坏帧／EOF 为终止故障；合法请求级错误仍可复用 worker |
| 10 · SDK 修改污染缓存 | pipeline 缓存写入和命中均浅复制，修改返回 Observation 可改变下一请求 | 在 pipeline 缓存读写边界深复制 Percept；保留通用 PerceptCache identity 契约 |
| 11 · 重复提示 N² 输出 | 同值提示反复匹配整组，输入1／2／3次产生1／4／9条结果 | 等效提示去重推理／缓存写入，再按输入顺序和次数复制回独立结果；覆盖冷缓存、热缓存、混合及禁用缓存 |
| 12 · 非 world Z 当抬升 | 只比较 frame 相等，camera／pixel Z变化被称作米制 lift | lifted 必须两侧 world；相同非 world 设置 `world_frame_required`，不提供 lift_m、不通过抬升；不同 frame 仍用原 frame_mismatch |
| 13 · 无效数值／溢出 | 负 sigma 产生负 uncertainty；字符串／NaN／小数整数选项可逸出500；极小 prism step 在封顶前 int(inf)；大 sigma 权重或 median 溢出 | 推理前共享类型／有限性／范围校验，坏选项返回 invalid_request。sigma>0、fraction在[0,1]、max_pairs整数1..3；高度候选先封顶512再round/int。相对 sigma 权重避免平方溢出；拒绝计算后非有限 sigma；median 用 `low+(high-low)*0.5`，保留最小正subnormal |
| 14 · SGBM 小图非法原生分配 | width≤16 强制16视差，OpenCV 尝试巨大分配或断言失败，HTTP500 | matcher 前确认合法16倍数搜索范围及适合的 window；不适合的小图返回 `sgbm_image_too_small`。合法大范围仍 clamp，偶数正整数 window 仍转奇数 |
| 15 · SDK readiness 超期 | 20 ms预算可等待150 ms后还返回成功；socket TimeoutError逸出。连续慢字节的body／status／headers及chunked CRLF也可突破期限，redirect重用旧连接预算 | 一个绝对deadline贯穿重试／backoff／轮询及受控HTTP状态行／响应头／body读取；readiness专用标准urllib opener按line／chunk重设剩余socket预算，redirect连接刷新剩余期限。成功返回前拒绝超期，传输超时统一 PerceptaError。普通请求仍用原urlopen，未加线程 |
| 16 · 替换 local runner 泄漏 | 覆盖旧 runner，close 只清理最后一个，旧模型／registry仍加载 | 先关闭旧 runner，再构造／加载替代者；覆盖共享 registry 顺序及替换失败路径 |
| 17 · CLI 错误退出0 | perceive 返回 unavailable 仍 exit0，自动化无法识别失败 | 输出仍可检查，任一 percept／fused 为 unavailable、invalid_request、timeout 则 exit1；普通 not_found 仍 exit0 |
| 18 · health／hash失真 | 通用 worker 退出后 registry仍ok；provenance有模型hash而 /v1/info 为null | 通用 subprocess／HTTP backend 提供不等待推理锁的 health_error；info使用 `_model_meta()` 或既有 model_sha256／_model_sha256，与来源信息一致 |
| 19 · Options过滤未执行 | Options.min_score／top_k仅进入cache key，实际工具只使用Prompt；优先级未定义 | 用户确认 Options 只填未显式字段；以 `model_fields_set` 判断，显式0.2／1也覆盖。请求验证复制并合并，SDK序列化前完成，HTTP往返一致；Options验证score在[0,1]、top_k≥1 |
| 20 · 三目误标为双目 | 使用两组pair／三台相机时 `degraded_to_two_view=true` | 按实际参与相机数判断双视角；未用满pair仍保留subset警告 |

### 回归证据

这些检查均使用现有环境；新增反例先在旧实现失败，再验证修复后的行为。

- SAM3／registry首批：**5 failed → 27 passed**；非world抬升：**2 failed → green**；capabilities失败清理：**2 failed → green**。最终 registry／verification相关组 **38 passed**。
- HTTP准入／失败geometry：7项新回归先红，相关组 **87 passed**；缓存／重复提示6项先红，相关组 **51 passed**。
- 标定边界：**22 failed、3兼容项通过 → 25 passed**；相关9文件 **206 passed**。10项旧测试改为先创建合法 ToolContext 再突变标定，以保留原工具边界拒绝断言。
- 数值／SGBM／相机参与：**45 failed → 45 passed**，另有兼容及median边界3项；最终相关6文件 **191 passed**。median的2项新增反例曾分别触发Geometry.inf和per_pair.inf。
- 协议／worker／相关模型工具：**231 passed、1 deselected**；暂缓的独立临时venv安装测试已纳入最终全套。
- Options合并：9项中7项旧实现失败、2项兼容通过，修复后全绿，相关组 **72 passed**；CLI／info：14项中10项先红、4项兼容，相关组 **42 passed**。
- SDK生命周期／readiness：真实200及503单字节慢body，40 ms预算在旧实现耗时约278 ms；2项先红再绿。最终复核另补status／headers的200／503四项（80 ms预算旧实现耗约229／290 ms）、chunked CRLF（120 ms预算耗183 ms）、redirect剩余连接预算，共6项先红再绿；兼容读取及关闭行为一并覆盖，最终client相关组 **43 passed**。
- 联合CPU worker／真实HTTP：请求先504、推理占位时下一请求503、worker期限回收后释放容量；随后请求HTTP200/unavailable，health显示error，shutdown后pending=0。
- 官方SAM3 processor的CPU状态探针：forward替换为记录输入的stub，无权重推理。修复后box→box→text分别为1框、1框、文本+0框，text→box不携带旧文本；确认reset保留图像特征且原地返回None。

最终整体验收结果见本节末尾；分组计数有重叠，不能相加成全套通过数。

### 文档对齐与保留边界

README、规格主本／镜像、部署说明、D40–D43与CHANGELOG同步现行语义；本记录保留旧代码位置、
公式及错误后果。既有D1–D39和上一批审计原文保留，根规格与docs镜像必须逐字相同。

HTTP504仍不能强制终止原生推理线程；容量保留到实际结束，shutdown也会等待。独立worker的期限
分别用于启动和每次完整RPC，首次准备可能使用两段预算。SDK可以限制受控HTTP读取，但同步API
不能强制中断OS DNS、底层连接／TLS建立或调用方自定义的阻塞health()；不承诺任意传输阶段均可硬取消。
非法／不支持的标定与请求参数现在会被明确拒绝。

本批未运行GPU forward、任务精度评估、长时间负载或跨机部署；没有更换权重、下载模型、推送或发布。
辅助／插件API的3个较低优先边界（超大直接fuse mask分配、PlaneGrid整数索引helper、极端长宽比grid上限）
仍未纳入本批20项修复；模型部署的既有证据与方法限制仍以model-isolation及D35–D39为准。

### 最终验收

在median边界及SDK状态行／headers／chunk framing／redirect修复冻结后执行：

- `uv run --locked pytest -o addopts= -q`：**903 passed、9 skipped**，72.33 s；包含已安装 OpenCV 5.0.0 的几何回归。有1条既有 Starlette/httpx 弃用警告；跳过项未计入通过。
- `uv run --locked ruff check .`：通过；`git diff --check`：通过。
- `uv run --locked python scripts/e2e_smoke.py`：**E2E OK**；真实 uvicorn 下的 HTTP／进程内结果一致，not_found 仍 HTTP200。
- 共享标定、registry和通用协议的 **11项doctest通过**。
- `uv run --locked python scripts/check_lock_portability.py`：6份锁文件均使用官方域名；根项目和5个模型项目的 `uv lock --check` 均通过，依赖和锁文件未修改。
- 规格主本与镜像逐字一致，15个文档模型字段集合与实际schema一致；本次涉及的7份文档中11个唯一Markdown本地链接目标存在；README／规格的13个Python、1个JSON代码块语法通过。

中间两轮全套分别为895／897 passed、9 skipped；后者已含2项median回归。之后又补6项SDK读取回归，
最终结果以本节最终验收数为准。
分组结果只用于展示具体覆盖，不替代本轮完整回归。所有修改保存在本地Git提交，未推送／发布；
原有 `.qoder/` 本机配置不纳入提交。

## 2026-11-12：再次review并形成详细修复报告

- **记录时间**：2026-11-12 12:58，Asia/Shanghai（UTC+08:00）。
- **报告最终核对时间**：同日13:00（UTC+08:00）。
- **审查基线**：`9779a96`，对照其父提交 `a761656`。
- **用户要求**：再review一次，并给出详细修复报告。
- **范围**：运行时代码和测试保持原样；新增报告、更新本页当前状态；未修复或提交下述运行时问题。

本页顶部修改前原文保留如下：

> **当前状态（2026-11-12 后续修复，D39–D43）**：`DOC-20261112-01/02` 的坐标错误，
> 以及之后全盘审计的20组问题均已修复。本页各批次中的“尚未修复”保留当时的历史状态。
> 坐标修复前基线为 `5156672`，20组修复前基线为 `a761656`；旧行为、新行为及验证见各批次记录。

**修改原因与新结论**：主体修复有效，但全套通过未覆盖所有异常组合，因此“全部关闭”表述过满。
独立复查确认4项P2、1项P3，其中3项为该提交引入或放大的回归，2项是修复前已存在的遗漏。
上一批次的20项前后对照及其验收结果不改写为本轮结论；本轮新增反例与后续验收要求见
[详细报告](reviews/2026-11-12-runtime-fix-review.md)。

| 新编号 | 分类／状态 | 已复现的行为 | 建议修复 |
|---|---|---|---|
| R1 | P2，回归，待修 | worker将BackendRequestError误编码为unavailable，新增终止策略使下一合法提示也永久失败；父版下一提示成功 | worker统一识别请求错误，保留健康worker |
| R2 | P2，回归，待修 | 打开1100个文件后，通用worker因select高fd限制启动失败并被回收；父版成功 | 共享期限IO与stderr等待使用poll／selectors |
| R3 | P2，既有遗漏，待修 | 部分提示命中缓存，未命中推理失败时直接返回失败组，已命中结果缺失 | 统一按输入顺序合并命中与失败结果 |
| R4 | P2，旧#13残余，待修 | table_z／table_plane／prism_step_m的超大JSON整数漏捕获OverflowError，导致HTTP500 | 完整覆盖融合数值解析边界，进入既有坏输入处理 |
| R5 | P3，回归，待修 | 切换config／registry时关闭了仍传给新runner的可关闭cache，随后缓存被禁用；默认cache无close，不受影响 | 转移本次仍复用的cache生命周期 |

本轮实际验证：完整CPU **903 passed、9 skipped**，68.10s；Ruff和真实uvicorn e2e通过。
定向组backend／SAM3为142 passed、1 deselected，metadata／registry为18 passed，几何组186 passed，
service／SDK／CLI组63 passed，计数重叠不相加。另有真实CPU worker、HTTP及OpenCV反例，
R1–R5均经root独立核对；R1／R2做父版本行为对照，R3／R4确认父源码已存在缺口。

未运行GPU、模型精度、长期压力或跨机部署；原有 `.qoder/` 保持未跟踪；未推送／发布。

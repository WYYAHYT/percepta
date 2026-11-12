# 文档修订记录

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

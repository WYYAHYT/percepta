# Percepta × RoboDojo：RGB 双目 / 三目感知扩展实施方案

> **文档类型**：面向 Coding Agent（Codex / Cursor / Claude Code 等）的技术设计与实施任务书
> **版本**：v1.1（由 2026-11-10 技术方案整理为 Agent 可执行版）
> **项目**：[WYYAHYT/percepta](https://github.com/WYYAHYT/percepta)
> **评测对象**：RoboDojo Agent-Only Policy
> **状态**：设计与任务规划，**不是已实现功能或实测结果**

---

> ## ⚠️ 编者补充说明（Editor's amendments）
>
> **本节及全文所有标记 `【编者补充 v1.2】` 的内容，均由 AI Agent 于 2026-11-10 在阅读当前
> 仓库代码（commit `267ad92`，v0.3.0）后追加，不属于原作者 v1.1 原文。**
>
> 原文未被删除或改写；补充内容一律以引用块或独立小节附录形式插入，并标注来源。
>
> ### 为什么补充
>
> 原文 §0.3 要求实施者“先审计，再编码”，并声明“本文的接口、文件路径与 CLI 仅是设计建议，
> 需按仓库实际代码对齐”。本次审计发现 **三处原文与当前代码不符，两处关键数字缺失**，
> 若不修正会让实施者重复已解决的问题、或缺少判断方案是否值得做的依据。
>
> ### 补充清单
>
> | 编号 | 位置 | 性质 | 摘要 |
> |---|---|---|---|
> | A1 | §2 表格 / §8.3 | **原文已过时** | “后运行 geometry tool 覆盖 `fused`”的问题已在 D15 修复并有测试锁定 |
> | A2 | §9.2 | **原文要求正确，代码有缺口** | `Calibration` 缺 `frame_id`，无法表达“同一相机不同时刻的位姿” |
> | A3 | §3.1 D0 行 | **原文缺少量级** | 现有轮廓法的实测精度（厘米级）未给出，无法判断双目要提升多少才有意义 |
> | A4 | 新增 §12.0 | **原文缺少前置测量** | 相机共视率与基线长度未先行测量，而这决定整个方案可行性 |
> | A5 | §1.2 | **原文缺量化门槛** | 假设可证伪，但没有“改善多少算改善”的阈值 |
>
> ### 补充者立场
>
> 这些补充**不改变原方案的技术路线**。原文的 M0→M5 顺序、特权信息隔离要求、
> 失败码设计、“不编造”原则，经核实均正确且必要。补充内容只涉及：
> 修正已过时的描述、指出代码层缺口、提供可量化的判断依据。

---

## 0. Coding Agent 首先阅读这里

### 0.1 项目目标

**在现有 Percepta 仓库中扩展无 VLA 训练的 3D 感知工具**，使用 RoboDojo 合法提供的三路 RGB 图像、相机标定和机器人状态，增加：

1. `stereo_3d`：任意有效双相机组合的 RGB 双目估计；
2. `triview_3d`：三相机原生 MVS 或多组 Stereo 结果融合；
3. `adaptive_3d`：根据数据质量在 MVS、Stereo 与既有 Mask Geometry 之间选择或回退；
4. 与当前 Percepta `Geometry` / `Percept` 兼容的统一输出；
5. RoboDojo Agent-Only 的**受控基线实验**，最终测量 `Success Rate (SR)` 与 `Score`。

**明确非目标**：训练或微调 VLA、改动 Agent LLM 权重、开发端到端动作神经网络、重写 Percepta 核心框架、修改 RoboDojo 评分器、使用模拟器真值作弊。

### 0.2 最高优先级的不可违反约束

- **不得在在线 Policy 输入或控制逻辑中读取模拟器 Ground-truth Depth、Object Pose、隐藏 Mesh 或其他未获准特权信息。** 离线数据评测允许存储 GT，但必须与在线 Policy 代码和数据结构隔离。
- 必须沿用 Percepta 已存在的 `Tool` / `Pipeline` / `Geometry` 契约；**优先新增模块，避免破坏已有 API**。
- Percepta 基础安装应保持轻量；Stereo/MVS/DA3 模型及其 GPU 依赖应设计成可选后端。
- 每条 Wrist Camera 的标定外参必须与该 RGB 图像属于**同一个观测时刻**；禁止使用过时的外参。
- 必须明确坐标系方向、变换方向及单位。**无可靠几何证据时返回不可用，不能编造米制 3D 结果**。
- 不可将 `visible_surface_centroid`（可见点云中心）直接当成 `object_center`（估计实体中心）。
- 不可将网络输出的 confidence 直接解释成毫米级标准差；必须另行做验证集标定。
- 没有实际运行评测，不得填写 Stereo/MVS/DA3 的 RoboDojo SR/Score。
- 正式评测协议、允许的相机元数据、外部模型、任务清单及许可约束，以**提交时官方规则**为准；上线前核实，不要从本设计推断已获许可。

### 0.3 Agent 工作方式

1. **先审计，再编码**：阅读 `schema.py`、`geometry.py`、`fusion.py`、`pipeline.py`、`registry.py`、`tools/multi_view_3d.py`；记录实际代码路径、类签名、数据结构和当前 Git commit。本文的接口、文件路径与 CLI 仅是设计建议，需按仓库实际代码对齐。
2. **先建纯几何单元测试**：在不需要 GPU、不启动 RoboDojo 的条件下验证坐标变换、校正、三角测量、深度尺度、相机同步。
3. **先完成双目 MVP 再扩展三目**：交付顺序 M0 → M1 → M2 → M3 → M4 → M5，不应先集成所有大模型。
4. **每阶段先运行验收测试，再继续下一阶段**。测试不通过时记录问题，不得用硬编码或 GT 泄漏掩盖错误。
5. **每次交付附变更说明**：已改文件、接口兼容性、测试结果、已知限制、配置/权重 hash、下一步。
6. 不对仓库内容作未经请求的破坏性重构；不在未核实官方 URL/许可证时自动下载大权重或提交权重进 Git。

### 0.4 先交付的最小闭环（MVP）

`合法 RGB + 同步标定 → SAM3 Mask → Fast-FoundationStereo → 深度/点云 → 现有几何先验融合 → Agent 结构化结果 → RoboDojo P0/P1 成对评分`

如果相机对没有足够共同视野，**先测量该事实**，不要为了“让模型出结果”而强行使用无效匹配。

---

## 1. 科学问题和成功判据

### 1.1 待回答问题

- **Q1**：在 RoboDojo 的 Head + Left Wrist + Right Wrist 非标准三相机系统下，RGB Stereo/MVS 是否比现有 Mask Geometry 提供更准确的米制 3D 信息？
- **Q2**：三组双目结果融合（pairwise stereo fusion）与一次原生多视角推理（native MVS），在深度质量、有效率与推理成本方面有何区别？
- **Q3**：上述 3D 感知改善是否能在**Agent、Prompt、控制器和原生动作预算相同**时提高 RoboDojo 的 SR/Score？
- **Q4**：自动回退能否降低深度模型在大视角、弱纹理、遮挡场景下的负收益？

### 1.2 可证伪假设

| ID | 假设 | 必须观察的证据 | 失败/否证解释 |
|---|---|---|---|
| H1 | 双目改善关键几何定位 | `center_xyz/top_z/grasp_point` 的成对误差下降 | 无共视、视差不可靠、标定误差 |
| H2 | 三目主要改善鲁棒性 | valid rate、覆盖率、失败率改善 | 第三视角不共视或引入冲突 |
| H3 | 质量门控优于强制使用网络 | 错误深度不会显著拉低整体 SR | quality 未校准、错误 fallback |
| H4 | 感知进步转换为任务收益 | 在 Precision 子集及全任务集合中 SR/Score 有可复现的成对改善 | 控制器、接触控制、任务难度成为瓶颈 |

**注意**：感知的深度 MAE 改善，不应被自动解读为 Policy Score 改善。

> **【编者补充 v1.2 — A5：假设可证伪，但缺少“改善多少算改善”的阈值】**
>
> 原文的 H1–H4 指定了“必须观察什么证据”和“什么情况算否证”，但没有给出**判定门槛**。
> 缺少门槛会导致两种失败：(a) 结果出来后无法判断是否算成功；
> (b) 事后按结果倒推标准。
>
> 建议在 **M0 结束前**（即看到任何深度模型结果之前）冻结下列阈值。
> 下表阈值是基于 v0.3.0 实测基线（见 §3.1 编者补充 A3）提出的**建议起点**，
> 实施者应结合 §12.0 的测量结果调整，但**必须在开测前定稿**：
>
> | 假设 | 指标 | 现有基线(D0) | 建议判为“有改善”的门槛 | 理由 |
> |---|---|---|---|---|
> | H1 | 3D 中心 P50 误差 | 0.3–1.0 cm | **≤ 5 mm 且 P90 ≤ 15 mm** | 需明显优于轮廓法才有意义；5 mm 已可用于精确抓取 |
> | H1 | `top_z` P50 误差 | 0–2 cm | **≤ 5 mm** | 高度是目前最弱的量，改善空间最大 |
> | H1 | 脚印面积误差比 | 0.65–2.18× | **≤ 1.3×** | 双目有真实表面，应能显著收紧 |
> | H2 | 有效深度覆盖率（目标区域） | 单目不可用 | **三目 ≥ 双目 + 10 个百分点** | 否则“第三个视角”只是成本 |
> | H3 | 低质量帧的误用率 | — | **< 5%** | 门控的意义就是挡住坏结果 |
> | H4 | RoboDojo SR | 待引官方基线 | **配对比较，95% CI 下界 > 0** | 不能用单 seed 波动宣称提升 |
>
> **三条纪律**（补充，原文未明确）：
>
> 1. **阈值必须在看到测试结果之前写入实验记录**，事后修改须标注并说明原因。
> 2. **H1/H2 达标但 H4 不达标**是合法结论，且是有价值的负结果——说明瓶颈不在感知。
>    此时不应通过调感知继续刷 L1 指标。
> 3. **S1 类指标（深度 MAE）达标不能替代 H1**：MAE 只在“模型成功返回深度”的像素上统计，
>    而 §10.3 已正确要求“同时报告 valid fraction”。**覆盖率低的模型可以有很低的 MAE。**

---

## 2. 现有 Percepta 状态与设计边界

> 以下是原方案对公开代码的审计摘要。**Coding Agent 仍须重新阅读其正在操作的具体 Git revision**；不得把摘要视为当前仓库永远成立的事实。

| 现有模块（需按当前仓库核实） | 功能 | 扩展原则 |
|---|---|---|
| `schema.Calibration` | 相机 `K`、`T_world_cam`、可选 Depth、尺寸等 | 兼容保留；新增时序和来源信息使用内部类型 |
| `schema.Geometry` | 世界坐标的 3D 实体几何及 uncertainty 字段 | 保留 Agent 高层输出语义；明确实体中心 vs 表面中心 |
| `tools/multi_view_3d.py` | 多路 Mask + 桌面/Prism/Hang 几何 | 基线、fallback；不删除 |
| `fusion.py` | 现有 Mask/深度反投影与融合流程 | **网络预测深度不应冒充模拟器提供的 `Calibration.depth`** |
| `pipeline.py` | 工具调度、聚合 fused 结果 | 解决多个 geometry tool 的覆盖顺序问题 |
| `registry.py` | 插件注册 | 注册新增后端，GPU 依赖惰性加载 |

### 2.1 推荐模块结构（建议，不是强制重排）

```text
src/percepta/
├── depth/
│   ├── schema.py             # DepthEstimate / PointCloudEstimate / Backend Protocol
│   ├── calibration.py        # 外参与坐标约定校验
│   ├── stereo_rectify.py     # 极线校正；位姿感知缓存
│   ├── disparity.py          # 视差 -> 米制 depth / point cloud
│   ├── stereo_backend.py     # Fast-FoundationStereo backend
│   ├── mvs_backend.py        # MVSAnywhere backend
│   ├── da3_backend.py        # DA3 backend（可选）
│   └── quality.py            # valid / consistency / failure reason
├── geometry_fusion/
│   ├── association.py        # 跨相机同一实例匹配
│   ├── pointcloud.py         # 点云筛选、变换、合并
│   ├── shape_completion.py   # 与 Prism/Hang 先验融合
│   └── selector.py           # adaptive / fallback
├── tools/
│   ├── multi_view_3d.py     # 已有，保留
│   ├── stereo_3d.py         # 新增
│   ├── triview_3d.py        # 新增
│   └── adaptive_3d.py       # 新增
└── adapters/
    └── robodojo.py         # 可选，不污染通用 Percepta 核心
```

**模块边界**：Depth Backend 输出表面深度，不直接决定抓取点；Geometry Fusion 决定实体和动作相关几何；Agent Tool Adapter 负责表达为 Agent 可调用的结构化结果。

---

## 3. 深度模型选型与集成顺序

### 3.1 候选比较

| 优先级 | 模型/方案 | 类别 | 输入与先验 | 用途 | 主要限制 |
|---|---|---|---|---|---|
| D0 | 现有 Mask Geometry | 轮廓拟合 | RGB Mask + 相机标定 + 桌面先验 | 永久保留基线 | 物体复杂形状、部分遮挡 || D1 | OpenCV StereoSGBM | 传统 Stereo | 校正后的 RGB Pair + K/T | 不依赖训练的几何对照 | 弱纹理/大夹角不稳定 |
| D2 | **Fast-FoundationStereo** | 神经 Stereo | 校正双目 RGB + 几何尺度 | **第一阶段 MVP** | 需要校正、共同视野和视差覆盖 |
| D3 | FoundationStereo | 神经 Stereo | 校正双目 RGB | 精度/延迟对照 | 计算资源开销 |
| D4 | 3-pair Stereo Fusion | 多双目后融合 | 最多 3 路 RGB + K/T | 先实现的“三目”路线 | 不等于原生三视角联合推断 |
| D5 | **MVSAnywhere** | 原生 MVS | 多路 RGB + 标定/位姿 | **三目主线** | 坐标约定、依赖、模型适用域 |
| D6 | **Depth Anything 3**（BASE 优先） | 通用多视角 3D | 多路 RGB + 可选已知 Pose | 基础模型对照 | 绝对尺度与微小结构需测量 |
| D7 | Depth + Percepta Shape Prior | 几何融合 | D2/D5/D6 + Mask/先验 | Agent 最终 Geometry | 融合规则与置信度校准 |

以上排序是**工程优先级而非 RoboDojo 实测精度排名**。模型 license、checkpoint 和版本均应由 Agent 在集成时核对并记录；不要把全仓库 LICENSE 误认为所有权重的授权。

> **【编者补充 v1.2 — A3：D0 基线的实测精度，原文未给出量级】**
>
> 原文只说 D0 的局限是“物体复杂形状、部分遮挡”，但没有数字。**缺少这个数字，
> 就无法判断 D1–D6 要提升到多少才有意义。**
>
> 以下是 v0.3.0 在**合成立方体、已知真值**下的实测（`scripts/footprint_accuracy.py`，
> 10 cm 立方体，20° 仰角，无深度、无噪声）：
>
> | 相机布局 | 质心误差 | 高度误差 | 脚印面积（真值 100 cm²） |
> |---|---|---|---|
> | 3 视角 0°/120°/240° | **0.33–1.04 cm** | 0–2 cm | 132–155 cm² |
> | 2 视角 0°/180°（最差） | 0.50–1.00 cm | 1–2 cm | 165–219 cm² |
> | 4 视角 0°/90°/180°/270° | **0.01–1.50 cm** | 0–3 cm | 108–120 cm² |
>
> **三条可直接使用的判断**：
>
> 1. **质心已到厘米级**，且横向精度明显高于高度精度（见决策 D35 的拆分不确定度）。
>    所以双目要证明价值，**主要看它能否把高度误差从厘米级压到毫米级**。
> 2. **脚印面积系统性偏大**（0.65–2.18×）。这是视觉外壳的**信息上限**，不是实现缺陷——
>    增相机、加视角都无法消除。**双目若能给出真实表面，这一项才可能真正改善。**
> 3. **0°/180° 正对是最差配置**：两相机看同一面，几何上等价于单相机。
>    这提示 M1 的相机对选择（原文 §4.4）应优先看**视角差异**而非基线长度本身。
>
> **注意这些数字的前提**：合成 mask、无遮挡、无标定误差、物体立在桌面。
> 真实 RoboDojo 场景会更差，但**量级可作为 M1 的对照起点**。

### 3.2 Fast-FoundationStereo（首选双目）

项目：<https://github.com/NVlabs/Fast-FoundationStereo>

- 任务：从校正后的左右 RGB 推断视差，再通过已知基线恢复米制深度。
- 与 RoboDojo 的关联：RGB-only、无需训练 VLA；适合作为按需调用的 Percepta 工具。
- 集成要求：**先 rectification，再 inference**；正确处理左右图顺序、有效 ROI、视差方向与最小/最大视差。
- 建议不将模型深度直接注入原有 `Calibration.depth`；通过新的 `DepthEstimate(source="rgb_inferred")` 返回。
- 记录 `checkpoint_sha256`、分辨率、迭代次数、model-only latency、端到端 latency、最大显存。

### 3.3 MVSAnywhere（首选原生三目）

项目：<https://github.com/nianticlabs/mvsanywhere>

- 任务：用一个参考图像、多个源图像与相机几何，在候选深度上做多视角特征匹配，预测参考视角深度。
- 与逐对 Stereo 不同：第三视角在**深度推断阶段**参与约束，而不是最终点云才融合。
- 必须核实权重期望的相机坐标约定、可接受源视角数量、深度范围设置和可用模型输入。
- 如 reference 与两路 source 不同时共视目标，须允许退化成双目或 Mask Geometry，而不是强制声称三目有效。

### 3.4 Depth Anything 3（基础模型对照）

项目：<https://github.com/ByteDance-Seed/Depth-Anything-3>

- 优先测试 pose-conditioned、多视角、已知相机参数的模式；不要额外估计已知相机位姿而引入误差。
- 优先考虑可复现且许可兼容的 SMALL/BASE checkpoint；更大模型作为可选离线对照。
- 必须验证**没有 GT scale alignment 的绝对米制误差**，不能只展示深度图主观质量。
- 对薄结构、插孔、杯沿、边界等几何细节进行单独误差分析。

### 3.5 依赖部署策略

- Percepta core 与 GPU 预测模型隔离；模型适配器可进程内运行或经可选 depth worker 调用。
- SAM3、Fast-FoundationStereo、MVSAnywhere、DA3 的依赖分别锁版本；不要盲目将多个不兼容 PyTorch/CUDA 环境合并。
- 不在默认单元测试启动时下载模型权重；提供 mock backend 运行 CPU CI。

---

## 4. 双目几何原理与实现约束

### 4.1 坐标约定

推荐在 Percepta 内部统一采用：

- `K`：像素相机内参；
- `T_world_cam`：**相机坐标 → 世界坐标**的 4×4 齐次变换；
- `camera_cv`：OpenCV 约定，X 向右、Y 向下、Z 向前；
- 距离和深度：米；
- 像素坐标：`(u, v)`，原点和 resize/crop 后的 K 必须同步处理。

**绝对不要在不检查定义的情况下把一个名为 `extrinsic` 的矩阵当成 `T_world_cam`。** 它可能表示 world→cam。

若 RoboDojo 对应相机坐标约定是 X 右、Y 上、Z 后，可用轴转换示意：

```python
T_world_cam_cv = T_world_cam_robo @ np.diag([1, -1, -1, 1])
```

但**必须以当前适配器、投影测试和官方坐标定义核对**；上式不能仅靠名称直接套用。

### 4.2 任意双相机的几何

两相机之间的相对位姿：

\[
T_{c_2c_1}=T_{wc_2}^{-1}T_{wc_1}
\]

对已校正的标准平行双目，理想情况：

\[
Z=\frac{fB}{d}, \qquad
\sigma_Z\approx\frac{Z^2}{fB}\sigma_d
\]

注意这是理想化公式。RoboDojo 的实际计算应依据 `stereoRectify()` 给出的校正投影矩阵 `P1/P2` 和 `Q` 进行坐标恢复，并处理视差符号、有效 ROI 和原始相机坐标系变换。`Z=fB/d` **不能直接用于未校正的三路原始 RGB**。

### 4.3 双目流水线

```text
Same-step RGB + Intrinsics + Extrinsics
    ↓
Camera convention verification
    ↓
Select candidate camera pairs: HL, HR, LR
    ↓
Undistort / Stereo Rectification
    ↓
Overlap / disparity-range / ROI validity checks
    ↓
Fast-FoundationStereo or StereoSGBM
    ↓
Disparity + confidence + valid_mask
    ↓
Metric depth / reprojected points (reference camera)
    ↓
SAM3 target mask / correspondence filtering
    ↓
World point cloud + quality diagnostics
```

### 4.4 自动选择双目配对

每个观测步枚举 `HL / HR / LR`，按以下信息排序：

1. 目标 Mask 在两个视角中的**真实共视程度**；
2. rectification 后有效共同 ROI；
3. 目标尺度与预测视差是否在模型可用范围；
4. 相机基线、三角交角以及遮挡风险；
5. 模型输出的 valid ratio、左右一致性、重投影误差；
6. 额外推理延迟（仅在质量相近时使用）。

初始评分可以使用加权启发式，但权重须在**独立验证集**上选择，不能用最终 test episodes 不断调参。

### 4.5 必须支持的失败情况

| 失败码示例 | 场景 | 正确行为 |
|---|---|---|
| `no_overlap` | 相机无有效共同视野 | 跳过模型，回退 |
| `target_not_co_visible` | 目标只在一路可见 | 不输出假双目深度 |
| `invalid_calibration` | K/T 缺失、方向错误或数值非法 | fail closed |
| `frame_mismatch` | 图像与动态外参不同步 | 不进行三角测量 |
| `disparity_out_of_range` | 近距目标视差超出模型支持 | 换 pair/配置或回退 |
| `low_valid_fraction` | 绝大多数对应关系失败 | 返回低可信或不可用 |
| `model_timeout` | GPU 超时 | 保持服务可用，回退 |

---

## 5. 三目：两条必须分开的技术路线

### T1. Pairwise Stereo Fusion

对三路图像分别尝试 `Head–Left`、`Head–Right`、`Left–Right`：

```text
HL: Stereo → depth/points
HR: Stereo → depth/points
LR: Stereo → depth/points
            ↓
       world-frame transform
            ↓
     consistency filtering + deduplication
            ↓
         object point cloud
```

不可直接平均三个视角的深度图：它们的像素不是同一物理位置，也可能观测不同的物体表面。融合必须在相同世界坐标系或显式对应关系下完成，并考虑可见性与测量置信度。

### T2. Native Multi-View Stereo（MVSAnywhere）

选择 reference view `r`，其余有效相机作为 source views。设参考像素 \(\tilde u_r\)、候选深度 \(z\)：

\[
X_w(z)=T_{wr}\begin{bmatrix}zK_r^{-1}\tilde u_r\\1\end{bmatrix}
\]

把该 3D 假设重投影到其他 source，比较多视角特征一致性，预测参考深度。实际网络可能使用多尺度/学习到的代价聚合，不一定与该简化公式逐层一致。

### 实验纪律

- 在**相同图像、同一时刻、同一个相机位姿集合**下对比 T1 和 T2。
- 明确哪些三目帧退化成了双目，报告三目实际有效率。
- 对比模型延迟与显存，不仅比较点云可视化。
- 第一阶段三目不额外移动机械臂，以免把视角控制与深度模型的收益混在一起。

---

## 6. 多视角同一实例关联

Stereo/MVS 的前提不仅是多相机共视，还要确保 Mask 对应的是**同一个物体**。当场景里存在多个相似盒子时，“每相机选置信度最高 Mask”可能关联错误。

推荐：

1. 对每个相机独立生成候选 Mask / ID / embedding / 2D bbox；
2. 利用标定、极线约束、2D 区域关系、语义一致性和重投影进行实例匹配；
3. 保留跨视角关联置信度；
4. 关联模糊则输出多个假设或不可用，而不是选错物体后输出高置信 3D。

该模块应在离线测试中覆盖：两个同色方块、遮挡、只见一半的物体、某个相机目标缺失等情况。

---

## 7. Geometry Fusion 与输出语义

### 7.1 不可混淆的 3D 字段

| 字段 | 精确定义 |
|---|---|
| `visible_surface_centroid` | 由有效深度生成的**可见点云**统计中心 |
| `object_center` | 结合实体形状先验估计的三维几何中心 |
| `top_z_m` | 目标估计最高表面世界高度 |
| `footprint_dims_m` | 目标底面长宽（注明估计方式） |
| `long_axis_yaw_rad` | 底面长轴方向 |
| `grasp_pose_world` | 面向指定抓取策略的候选抓取位姿 |
| `uncertainty` | 已定义估计方法和校准范围的不确定性元数据 |

### 7.2 建议融合算法（先确定性，不训练）

1. 根据 SAM3 Mask 提取深度有效的目标表面点；
2. 通过左右一致性、多视角重投影及点云离群点剔除清理表面；
3. 使用现有桌面平面、Prism/Hang 模型生成一个或多个实体候选；
4. 用深度表面误差 + Mask 投影误差共同选择或约束候选；
5. 输出明确的 `method_used`、`quality`、`fallback_reason` 与 `geometry`；
6. 若测量不可靠，则明确回退 Mask Geometry；如果 Mask 也失败，返回不可用。

示例伪代码：

```python
if valid(mvs) and mvs.quality >= tau_mvs:
    return fuse(mvs, mask_geometry)
elif valid(stereo) and stereo.quality >= tau_stereo:
    return fuse(stereo, mask_geometry)
elif valid(mask_geometry):
    return mask_geometry.with_warning("depth_fallback")
else:
    return GeometryUnavailable(reason="insufficient_evidence")
```

### 7.3 不确定性与诊断字段

至少在内部保留：

- `valid_fraction`
- `reprojection_error_px`
- `left_right_consistency`（若适用）
- `cross_view_consistency_m`
- `position_covariance_3x3`（有校准的前提下）
- `shape_fit_residual`
- `calibration_quality`
- `chosen_camera_pair`
- `chosen_backend`
- `warnings` / `failure_code`

这里的 `quality` 是**方法选择的工程得分**，不能未经验证就解释成概率意义上的 `P(correct)`。

---

## 8. API 和数据契约（设计草案）

> **重要**：以下类、字段和工具名称是准备让 Agent 实现的**目标设计**，不代表当前 Percepta 已具备这些 API。开始编码前应与已有 schema 保持兼容。

### 8.1 内部深度估计契约

```python
from dataclasses import dataclass
from typing import Literal
import numpy as np

@dataclass(frozen=True)
class DepthEstimate:
    reference_camera: str
    depth_m: np.ndarray              # (H, W), OpenCV camera Z, meters
    valid_mask: np.ndarray           # (H, W), bool
    confidence: np.ndarray | None    # (H, W), raw/relative score
    source_cameras: tuple[str, ...]
    frame_id: str                    # binds image to dynamic calibration
    source: Literal["rgb_inferred"]
    backend: str
    geometry_convention: str
    diagnostics: dict[str, float | str]
```

对于一个真正的 3D 模型，也可额外定义 `PointCloudEstimate`；但**不应只给一组无参考系与来源说明的 `xyz` 数组**。

### 8.2 Percepta 侧高层调用（目标）

```python
from percepta import PerceptaClient, Prompt, PerceiveOptions

client = PerceptaClient("http://127.0.0.1:8790")

result = client.perceive(
    images={
        "cam_head": head_rgb,
        "cam_left_wrist": left_wrist_rgb,
        "cam_right_wrist": right_wrist_rgb,
    },
    calibration=calibrations,  # K/T for the same observation step
    prompts=[Prompt(type="text", text="red block")],
    options=PerceiveOptions(
        tools=["sam3", "adaptive_3d"],
        fuse=True,
    ),
)
```

### 8.3 工具边界

| 工具 | 输入 | 关键返回 | 退化逻辑 |
|---|---|---|---|
| `stereo_3d` | ≥2 RGB + K/T + Mask | `Geometry + pair_metrics` | 无效 pair → Mask Geometry |
| `triview_3d` | ≤3 RGB + K/T + Mask | `Geometry + view_metrics` | 三目 → 双目 → Mask Geometry |
| `adaptive_3d` | 1–3 RGB + K/T + Mask | `Geometry + chosen_method + provenance` | 根据质量门控选择 |

**特别注意**：若现有 `pipeline.py` 存在“后运行的 geometry tool 覆盖 `response.fused`”行为，要制定**单一拥有者**或显式融合/优先级规则，避免 `sam3, multi_view_3d, stereo_3d` 的顺序暗中决定最终结果。

> **【编者补充 v1.2 — A1：此问题已在 v0.3.0 解决，原文描述已过时】**
>
> 审计 commit `267ad92` 确认：**“后运行的 geometry tool 覆盖 `fused`” 不再成立**。
>
> 实际代码（`src/percepta/pipeline.py:299`）：
>
> ```python
> if self._tool_kind(name) == "geometry":
>     # D15: a geometry-kind stage *is* the response's `fused`; it is never kept in
>     # `percepts`, and fusion.fuse() is not called a second time.
> ```
>
> 现行语义（决策 D15，`docs/decisions.md`）：
>
> | 情况 | 行为 |
> |---|---|
> | pipeline 中有 geometry 类工具 | 它的 ok Percept **直接成为** `response.fused`，不进 `percepts` |
> | 有多个 geometry 工具 | **第一个**产出 ok 的生效，后续的被忽略并记录警告 |
> | pipeline 中无 geometry 工具 | 才调用 `fusion.fuse()` |
> | geometry 工具返回空 | `fused=None` + `geometry_unavailable` 警告，**不再二次 fuse** |
>
> **对 M2/M4 的实际影响**：不需要再设计“单一拥有者”机制，但**新增 `stereo_3d` /
> `triview_3d` / `adaptive_3d` 时必须意识到它们与 `multi_view_3d` 同属 `kind="geometry"`**。
> 若同时启用多个，**只有第一个会生效**——这与原文担心的“顺序暗中决定结果”是同一个风险的
> 另一种形态，只是现在行为是确定的而非偶然的。
>
> **建议**：新增的 `adaptive_3d` 应当**取代**而非并存于 pipeline 中的几何工具；
> 已有测试 `tests/unit/test_pipeline.py` 覆盖此行为，新增工具须补充同类测试。

### 8.4 建议配置（不是既有字段）

```yaml
pipeline:
  default: [sam3, multi_view_3d]  # 保持旧默认行为

tools:
  stereo_3d:
    backend: fast_foundation_stereo
    pair_selection: auto
    rectify: true
    max_disparity: 192  # 示例；须依据实际视差几何设置
    fallback_to: multi_view_3d

  triview_3d:
    backend: mvsanywhere
    reference_camera: auto
    fallback_to: stereo_3d

  adaptive_3d:
    candidate_methods: [triview_3d, stereo_3d, multi_view_3d]
    fail_closed: true

robodojo_adapter:
  require_same_observation_step: true
  forbid_privileged_depth: true
  forbid_object_gt_pose: true
```

### 8.5 示例结果（**数值虚构，仅展示 schema**）

```json
{
  "object_id": "obj_001",
  "method": "stereo_3d+shape_prior",
  "source_cameras": ["cam_head", "cam_left_wrist"],
  "coordinate_frame": "world",
  "center_xyz_m": [0.22, 0.36, 0.81],
  "top_z_m": 0.87,
  "grasp_yaw_rad": 0.41,
  "quality": {
    "valid_fraction": 0.63,
    "reprojection_error_px": 0.84
  },
  "fallback": false,
  "source": "rgb_inferred",
  "frame_id": "step_0082"
}
```

---

## 9. RoboDojo Adapter 设计

### 9.1 唯一职责

从 RoboDojo/XPolicyLab 允许提供的 observation 与机器人配置中，构造 Percepta 的 `RGBFrameBundle + Calibration`；包括：

- 三相机 `RGB`、相机 ID、图像分辨率；
- 有效的 `K` 和必要的畸变信息；
- 与 RGB 同步的 `T_world_cam`；
- 机器人关节状态、手眼标定/FK（仅在规则允许的范围内）；
- `frame_id`、`observation_step`、来源和格式元数据。

### 9.2 动态 Wrist Camera

Head 相机可固定，但 Left/Right Wrist Camera 随关节状态改变位姿。同一相机 ID 不代表外参不变。

任何 rectification cache 的 key 都必须包含：图像尺寸、对应 K/畸变参数、**参与校正的相机相对姿态**，而不仅是 `(left_camera_id, right_camera_id)`。

> **【编者补充 v1.2 — A2：当前 `Calibration` 无法表达时序，这是一个真实缺口】**
>
> 原文的正确要求在代码层**目前无法满足**。审计 `src/percepta/schema.py:133`：
>
> ```python
> class Calibration(_Forbid):
>     camera: str
>     width: int
>     height: int
>     K: list[list[float]]
>     T_world_cam: list[list[float]]
>     depth: DepthRef | None = None
>     id: str | None = None       # 语义模糊：是标定版本？还是采集时刻？
> ```
>
> `grep -c "frame_id\|observation_step" src/percepta/schema.py` 返回 **0**。
>
> **风险**：腕部相机每帧位姿变化。若调用方用 `id` 缓存 rectification map，用第 80 步的
> map 处理第 82 步的图像，会得到**系统性米制偏差**——而且这个偏差在图像上看不出来。
>
> **建议的最小改动**（符合 spec §3“只加可选字段”的规则）：
>
> ```python
> class Calibration(_Forbid):
>     ...
>     frame_id: str | None = None          # 拟新增：绑定图像与外参的同一观测时刻
>     captured_at: datetime | None = None  # 拟新增：可选，用于跨进程时序校验
> ```
>
> **实施要求**（供 M1 使用）：
>
> 1. rectification / depth 缓存的 key **必须**包含 `frame_id`；
> 2. `PerceiveRequest` 校验：若任一 `Calibration.frame_id` 非空且与图像来源的 frame 不符，
>    返回 `frame_mismatch` 失败码（原文 §4.5 已定义该码）；
> 3. 该字段为**可选**，因此不会破坏既有调用方——`frame_id=None` 时退化为旧行为。
>
> **注意**：以上字段名是**拟新增**，不代表已存在。实施前需与 `docs/spec.md` §3 的变更记录
> 同步更新（该 spec 规定新增可选字段需记录在 §18）。

建议优先以同一次 observation 获取图像和位姿；如 API 分次读取，需要进行严格时序检验。

### 9.3 特权信息隔离

| 信息 | 在线 RGB-only Agent | 离线评价 |
|---|---|---|
| 三路 RGB | 在当前协议允许下可用 | 可用 |
| 允许的 K / robot state / camera extrinsics | **必须核对正式规则** | 可用 |
| 模拟器 Depth Buffer | **不得作为模型输入** | 可用作 GT |
| 物体真值 Pose / Mesh | **不得用于规划或估计** | 可用作 GT |
| 测试集 GT 对齐缩放、后验标定 | **禁止** | 仅作分析、须标注 |

建议在线进程完全不暴露 GT 数据接口；将离线 GT 写在不同的数据命名空间/文件目录/进程中，便于代码审计。

---

## 10. 单元测试和离线几何验证

### 10.1 必须先做的 CPU 单元测试

- [ ] **坐标方向**：3D 点 world→cam→pixel 与解析预期一致。
- [ ] **回投影**：pixel + 已知 depth → 原始 3D 点（允许数值浮点容差）。
- [ ] **三角测量**：两相机已知刚体变换下恢复合成点，并评估投影残差。
- [ ] **Rectification**：对应点的校正后极线残差接近零。
- [ ] **视差方向**：左右顺序互换不能静默产生正的错误深度。
- [ ] **Scale**：米、毫米和厘米的单位错误能被验证或检测。
- [ ] **图像尺寸**：resize/crop 后的 K 正确缩放或平移。
- [ ] **多相机不同步**：frame/step mismatch 显式拒绝计算。
- [ ] **无重叠/目标缺失**：返回具体 failure code 而不是虚构结果。
- [ ] **动态外参缓存**：同 pair 不同 Wrist 位姿不会误用旧 rectification map。

### 10.2 GPU 模型集成测试

- [ ] 真实 checkpoint 能在目标 GPU 启动并完成一次推理。
- [ ] 运行 640×480 的默认输入，记录全部前后处理和推理耗时。
- [ ] 模型权重、CUDA/PyTorch 版本可复现。
- [ ] 对同一输入多次推理结果相同或在记录的容差内。
- [ ] 不使用 GT 深度做在线比例对齐。
- [ ] GPU 模型不可用时，Percepta 核心测试及 Mask Geometry 仍可运行。

### 10.3 离线数据集与指标

推荐按以下维度分层保存：

- 高/低纹理，反光、透明/半透明；
- 目标尺寸：小物体、薄结构、杯口/孔洞；
- 高/低遮挡；
- 小/大相机基线和观察夹角；
- 近距大视差；
- 两个相似物体导致的关联歧义；
- 动态相机位姿和时间偏移。

统计：

- Depth MAE / RMSE（仅在 GT 有效且目标可见的像素），**同时报告 valid fraction**；
- 3D object center、top_z、footprint、yaw、grasp point 的误差分布（P50/P90/P95）；
- 投影误差、跨相机一致性、匹配失败率；
- 置信度覆盖率和校准误差；
- 模型推理与 Percepta 端到端 p50/p95 延迟、显存峰值。

不能只在“模型成功返回深度”的样本上计算低 MAE 而忽略失败样本。

---

## 11. RoboDojo Agent-Only Policy 对照实验

### 11.1 Policy 矩阵

| Policy ID | 可用感知工具 | 可主动移动观察臂 | 研究目的 | SR / Score |
|---|---|---|---|---|
| P0 | 原始 `multi_view_3d` | 否 | 不变的 Agent baseline | **待实测/引用注明环境** |
| P1 | `stereo_3d` | 否 | 增加双目 | 待测 |
| P2 | 原生 `triview_3d`（MVSAnywhere） | 否 | 增加三目 MVS | 待测 |
| P3 | Pairwise 3-view Stereo Fusion | 否 | 三组双目后融合 | 待测 |
| P4 | `adaptive_3d` | 否 | 动态选择/回退 | 待测 |
| P5 | P4 + 可选主动观察动作 | 是 | 后续研究（单独计入移动成本） | 待测 |

**第一篇/第一轮报告建议仅做 P0–P4**，不要一开始混入主动相机运动。P5 可以作为明确独立的第二阶段。

### 11.2 公平性约束

1. 固定 Agent 基础模型版本、系统 Prompt、Tool descriptions、运动/抓取控制器及解码配置；
2. 任务版本、相同任务清单、seed、初始布局、episode 数量一致；
3. P0–P4 使用同一时刻的**相同三路 RGB**；禁止额外调用环境私有信息；
4. 对更高计算开销的模型同时记录 wall-clock/推理 budget；
5. P5 必须单列增加观察动作带来的 native action count；
6. 报告成对差值及 bootstrap 95% 置信区间；
7. 首先评测 Precision 子集，然后评测当前官方要求的完整任务集合。

### 11.3 实验分层

- **L0（工程正确性）**：相机位姿/尺度/时间戳/重投影；
- **L1（感知质量）**：Depth 和 Agent 所需 Geometry 精度；
- **L2（策略结果）**：SR、Score、原生动作数、调用次数、故障码和耗时。

必须能够串联单 Episode 中的：`percept_id → image_hash / frame_id → geometry → tool call → robot action → outcome`。

### 11.4 评测命令模板（非经验证的可执行命令）

> 在运行前先对照锁定版本的 `scripts/robodojo.sh --help` 或官方示例。下列 `<...>` 均为占位符。

```bash
# 单任务 smoke test（实际 flag 和 policy adapter 需先核实）
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --task stack_blocks \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --eval-num 1

# 多任务评测模板
bash scripts/robodojo.sh benchmark \
  --policy-dir XPolicyLab/policy/<POLICY_NAME> \
  --ckpt <CKPT_NAME> \
  --policy-env <POLICY_ENV> \
  --eval-num native --seed 0

bash scripts/robodojo.sh summarize
```

**任务数量不可写死**：原方案查阅时“42 simulation tasks”与 Quick Evaluation 的“54 runnable tasks”并存。需以锁定 benchmark commit 的实际 task inventory、seed 和评测要求为准。

---

## 12. 分阶段实施任务单

> **【编者补充 v1.2 — A4：建议在 M0 之前插入 §12.0 的“可行性前置测量”】**
>
> 原文从 M0 直接进入代码审计，但**整个双目方案依赖两个尚未验证的物理前提**。
> 若这两点不成立，M1–M3 的工作量会大幅贬值。建议把下列测量放在 M0 的最前面
> （它属于“审计”范畴，不增加阶段）。
>
> ### §12.0 可行性前置测量（属 M0，不需 GPU、不需模型权重）
>
> **测量 1：三相机两两共视率**
>
> RoboDojo 的相机布局（`env_cfg/camera/camera_config.yml` + `env_cfg/robot/dual_x5.yml`）：
>
> | 相机 | 位置 | 特性 |
> |---|---|---|
> | `cam_head` | 固定 `[0, -0.41, 1.308]`，俯视 60° | 看全局 |
> | `cam_left_wrist` | 挂在左臂末端，**随关节角变化** | 焦距更长（397 px），视野窄 |
> | `cam_right_wrist` | 挂在右臂末端，**随关节角变化** | 同上 |
>
> 已知约束（引自 PyRUA-Lean 对同一 benchmark 的实测描述）：
>
> > “At the start each wrist camera sees the table ahead of its own arm, from about
> > y = −0.15 outward and some 40 degrees to either side”
>
> **推论**：两臂归位时，`LR` 配对很可能**零共视**；双臂伸向同一物体时，两腕相机
> 可能靠得很近导致**基线趋零**。这两种情况都会让 `LR` 双目失效。
>
> **要测的量**（每个任务、每个典型关节配置下）：
>
> - `HL` / `HR` / `LR` 三对的全景共视率（像素级重叠）；
> - **目标物体**在两视角的共视率（有掩码后的真实重叠）；
> - 基线长度分布（`LR` 随双臂姿态的变化范围）；
> - 由 θ（两相机光轴夹角）与 B（基线）估算的理论深度标准差
>   `σ_Z ≈ Z²/(f·B)·σ_d`，其中 `σ_d` 取 0.5–1 px。
>
> **验收**：给出一张表，列出每种配对在典型任务下的“有效共视率 / 基线 / 理论 σ_Z”。
> **若 `LR` 共视率长期低于某个阈值（建议 5%），M2/M3 的“三目”应重新定义为
> “`HL` 与 `HR` 两个双目的并集”**，而非三视角联合推断——这与原文 §5 中
> “明确哪些三目帧退化成了双目”的要求一致，但需要在 M0 就定下预期，而不是等到 M3 才发现。
>
> **测量 2：视差范围是否落在模型可用区间**
>
> 近距物体（< 0.3 m）视差可能超出神经立体模型的预训练范围。用测得的 B 与 f，
> 对目标距离 0.2–1.0 m 估算视差 `d = f·B/Z`，与 Fast-FoundationStereo 支持的
> 最大视差（原文 §8.4 示例为 192 px，需核实）比较。
>
> **验收**：给出“距离 vs 视差”曲线，标出模型可用区间与 RoboDojo 实际操作距离的重叠部分。
>
> ### 为什么必须在写网络代码之前做
>
> 这两项测量**全部可用纯几何 + 现有合成工具完成**（`tests/synth_geom.py` 已有
> `camera_rig` / `cube_masks` / `mask_percepts`），不需要 GPU、不需要权重、
> 不需要启动 RoboDojo 仿真。
>
> 换句话说：**用 1 小时就能判断整个方案值不值得投入数周**。如果共视率或视差范围
> 不满足，正确的做法是先调整方案（例如只做 `HL`/`HR` 双目、或改用单目基础模型），
> 而不是先集成模型再说。

### M0：代码审计、合法数据采集、几何验证

**任务**

- [ ] 锁定 Percepta / RoboDojo / XPolicyLab commit SHA、Python/CUDA/GPU 信息。
- [ ] 检查真实 `Calibration`、`Geometry`、`Tool`、Registry、Pipeline API。
- [ ] 确认 RGB、相机 K/T 与机器人状态哪些在 Agent-Only 协议内允许。
- [ ] 开发只读取合法在线字段的 `RobodojoAdapter`。
- [ ] 为离线验证另建 GT exporter（隔离 Policy 进程）。
- [ ] 完成第 10.1 节纯几何测试。

**验收**：同帧标定正确、合成 3D 点投影/反投影一致，在线无 GT 访问。

### M1：Fast-FoundationStereo 双目 MVP

**任务**

- [ ] 定义 `DepthEstimate` 和 backend protocol。
- [ ] 实现相机对选择与 rectification，支持动态 Wrist 外参。
- [ ] 集成 Fast-FoundationStereo 权重为可选依赖。
- [ ] 实现 disparity → metric depth → world point cloud。
- [ ] 使用 SAM3 Mask 输出目标局部点云和 quality diagnostics。
- [ ] `stereo_3d` 注册并保持原 pipeline 可运行。

**验收**：有效视角能生成可复现米制点云；无共视、错误标定和模型失败时显式返回或回退。

### M2：Depth + Mask Geometry 融合

**任务**

- [ ] 定义物体中心与可见表面中心不同语义。
- [ ] 将深度表面约束与现有 Prism/Hang 候选融合。
- [ ] 加入质量门控、置信度校准、失败诊断。
- [ ] 解决多个 geometry tool 对 `response.fused` 的输出覆盖歧义。
- [ ] 建立稳定的 `adaptive_3d` 初版 fallback。

**验收**：结构化 Geometry 可靠；低质量 Stereo 不会覆盖更可信的 Mask-only 结果。

### M3：原生三目和基础模型对照

**任务**

- [ ] 实现 3-pair Stereo 后融合。
- [ ] 集成 MVSAnywhere，并核实相机位姿约定。
- [ ] 可选集成 DA3 BASE（pose-conditioned）。
- [ ] 对相同 RGB 帧进行 D0–D7 的 L1 指标对比。
- [ ] 记录模型 latency/显存/许可证/失败类型。

**验收**：得出“原生三目 vs 双目后融合”可复现对照，而非只有展示图。

### M4：Agent-Only Policy 集成

**任务**

- [ ] 将 Percepta 新工具接入现有 Agent Tool 调用机制。
- [ ] 固定相同 Agent/Prompt/Controller，构建 P0–P4 配置。
- [ ] 确保不读取任何模拟器 privileged depth/pose。
- [ ] 完成单任务 smoke test；每个 Episode 记录 geometry 与动作关联。

**验收**：P0/P1 能在相同 seed/layout 上成功完成至少一次规范 smoke evaluation；日志可追溯。

### M5：正式实验、消融和复现包

**任务**

- [ ] Precision 子集成对比较。
- [ ] 在当前官方任务集合/episode 配置下扩展评测。
- [ ] 汇总每项 SR、Score、动作数、额外 latency、失败原因。
- [ ] 计算配对置信区间，不用单次 seed 下的小波动宣称提升。
- [ ] 形成模型/许可证审计报告、复现 README 和可运行配置。

**验收**：提供完整结果表、可复现实验记录和方法局限说明；**不在无实测前写出 P1–P4 成绩**。

---

## 13. Agent 开发检查清单（DoD）

- [ ] 原 Percepta API、默认 `[sam3, multi_view_3d]` 和 CPU-only CI 未破坏。
- [ ] 所有动态相机外参与图像同帧，缓存不会跨姿态误用。
- [ ] Rectification、disparity、depth、3D world projection 通过数值测试。
- [ ] 相机 ID、帧 ID、模型名称、权重 SHA 与来源出现在日志/结果中。
- [ ] Stereo / MVS 不可用时可明确拒绝并安全回退。
- [ ] 不混淆 `visible_surface_centroid` 和 `object_center`。
- [ ] 置信度经过评估，不伪称毫米级误差。
- [ ] SAM3 跨视角实例关联不会默认“最高 Mask 分数 = 同一物体”。
- [ ] 不导入模拟器 GT 深度或物体真值到在线 Policy。
- [ ] 模型 latency 与全链路 latency 分开记录，含显存峰值。
- [ ] D0–D7 离线对照和 P0–P4 Policy 消融可复现。
- [ ] SR/Score 的任务、seed、episode 数和 benchmark 版本记录明确。
- [ ] 模型许可证与 checkpoint 许可逐项核对，不捆绑受限权重分发。

---

## 14. 风险登记

| 风险 | 可能后果 | 对策 |
|---|---|---|
| 三相机无共同视野 | 三目结果不可靠 | 先量化有效率，自动降级 |
| 动态 Wrist 标定错误 | 系统性米制点云偏差 | 同帧绑定 + 数值投影回归 |
| 弱纹理/透明/反光物体 | 高置信错误匹配 | 跨视角一致性 + 质量门控 |
| 近距离视差过大 | 超出预训练可用范围 | 视差范围检查、换 pair |
| 多实例关联错误 | 估计到错误对象 | 实例关联分数、冲突拒绝 |
| 将表面中心当实体中心 | 抓取偏移 | Shape prior 与字段语义分离 |
| 大模型超时/显存不足 | Agent tool 不稳定 | 独立 worker、timeout、fallback |
| 多个 geometry tool 相互覆盖 | 输出不是预期方法 | 单一 fused owner / 显式 selector |
| 在线 GT 泄漏 | 评分不合规 | 进程/Schema/数据源隔离审计 |
| 许可证不兼容 | 无法发布/使用权重 | 适配代码与权重解耦 |
| Eval 版本发生变化 | 数据不可横比 | commit/task inventory/seed 锁定 |

---

## 15. 建议的首个 Coding Agent 任务（可直接复制）

> 你将为 `WYYAHYT/percepta` 开发 RoboDojo RGB 双目/三目感知扩展。请先完整阅读当前仓库的 `schema.py`、`geometry.py`、`fusion.py`、`pipeline.py`、`registry.py`、`tools/multi_view_3d.py`、README 和测试。不要直接重构 API 或下载大型权重。首先完成 **M0：代码审计 + 纯 NumPy/OpenCV 双目几何基础**。输出：① 当前关键接口及兼容性报告；② 与现有工程风格一致的标定坐标约定文档；③ 双目相对位姿、投影、三角测量和校正的纯几何单元测试；④ 一个不依赖模型权重的 Stereo Backend Protocol/Mock 方案；⑤ 预期新增文件和后续 Fast-FoundationStereo 集成计划。必须明确在线不得读取 RoboDojo 模拟器 GT depth/object pose。所有接口提案要标记“拟新增”而非声称已经存在。只有 M0 验证通过后才推进 M1。

推荐首个 PR 命名：`feat(percepta): add calibrated stereo geometry foundation`。

---

## 16. 资料来源及需二次核对的链接

- [Percepta 仓库](https://github.com/WYYAHYT/percepta)
- [Percepta schema.py](https://github.com/WYYAHYT/percepta/blob/main/src/percepta/schema.py)
- [Percepta fusion.py](https://github.com/WYYAHYT/percepta/blob/main/src/percepta/fusion.py)
- [Percepta pipeline.py](https://github.com/WYYAHYT/percepta/blob/main/src/percepta/pipeline.py)
- [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo)
- [FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [MVSAnywhere](https://github.com/nianticlabs/mvsanywhere)
- [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
- [RoboDojo Benchmark](https://github.com/RoboDojo-Benchmark/RoboDojo)
- [RoboDojo Quick Evaluation](https://robodojo-benchmark.com/doc/usage/quick-evaluation/)
- [RoboDojo Submission Guide](https://robodojo-benchmark.com/doc/usage/robodojo-submission/)

**最终声明**：本文是一份可供 Agent 执行的研发计划，不是声称以上新增工具已经实现的 API 文档。任何命令、模型依赖、权重许可、排行榜约束和 benchmark 任务数量，都需在实际编码或提交前按锁定版本复核。

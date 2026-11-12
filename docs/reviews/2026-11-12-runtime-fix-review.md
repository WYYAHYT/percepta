# 9779a96 修复复查与详细报告

**复查日期：2026-11-12，Asia/Shanghai（UTC+08:00）；报告整理时间：12:58。**
报告最终核对时间为同日13:00。
修复前基线为 `a761656`，被审查提交为 `9779a96`，共43个文件、2909行增加、307行删除。
本轮核对该提交、相关调用者／worker／适配器／测试及规格，补充CPU与本机HTTP反例。
运行时代码和测试保持 `9779a96`；本轮仅新增报告和更新修改记录，没有修复下述新发现。

**复查结论：主体修复有效，仍有4项P2和1项P3待修。** 未确认新的P1。
上一轮“20项全部关闭”说得过满：正常路径与已登记的反例通过，不代表异常组合和资源边界全部覆盖。
本次发现包括3项本提交引入或放大的回归、2项修复前就存在的遗漏，下面分别标明。

假定使用方式为本机loopback服务、并发请求、连续图像帧、同图多提示、常驻worker，
以及可嵌入其他Python宿主的SDK。自定义可关闭cache作为明确的扩展场景单独标注。
不涉及RoboDojo接入、策略评估或模型精度判断。

**一、必须补修的发现**

**R1 · P2：合法的请求错误会使健康worker永久不可用（本提交放大的回归）**

- 位置：[subprocess_backend.py:303](../../src/percepta/backends/subprocess_backend.py#L303)、[worker.py:114](../../src/percepta/backends/worker.py#L114)；关联 [Sam3BackendError](../../src/percepta/tools/sam3.py#L55)。
- 原意：请求错误返回 `invalid_request`，worker继续服务；真正的传输／模型不可用才终止worker。
- 问题：worker只把 `ValueError/KeyError` 当请求错误。项目自己的 `BackendRequestError` 继承 `RuntimeError`，因此被编码为 `unavailable`；本次新增的终止策略收到该错误后立即回收worker。
- 实际调用链：`SubprocessBackend → worker → 项目现有 HttpBackend → 本机二进制模型服务`。prepare返回ok；服务对文本 `invalid` 返回合法 `error/invalid_request`，对随后文本 `red` 能正常生成64像素mask。
- 当前结果：`invalid` 得BackendUnavailable，随后 `red` 仍得BackendUnavailable，`worker_retained=False`。服务本身健康，后续请求无需再次触发原错误也一直失败。
- 父提交对照：`a761656` 首次已有误分类，但随后 `red` 成功、mask sum=64、worker仍在。因此新增后果是永久失效，旧误分类与新终止策略需要一起修。
- 最小修法：worker显式识别 `BackendRequestError`；让SAM3自己的请求错误纳入同一异常语义。保留真正不可用错误的终止处理。
- 必要回归：真实worker的adapter先抛请求错误，再接受合法请求；断言第一次为请求错误、第二次成功、PID不变，并核对进程内／子进程状态一致。
- 不修的影响：一次可恢复的输入／适配器请求错误会变成需要重建工具或重启服务的模型故障。

本轮CPU对照输出：

```text
9779a96  invalid → BackendUnavailable
9779a96  red     → BackendUnavailable; worker_retained=False
a761656  invalid → BackendUnavailable（旧误分类）
a761656  red     → OK; mask_sum=64; worker_retained=True
```

**R2 · P2：编号较高的正常文件描述符使worker启动失败（新增到通用后端的回归）**

- 位置：[subprocess_backend.py:213](../../src/percepta/backends/subprocess_backend.py#L213)、同文件 [stderr pump:180](../../src/percepta/backends/subprocess_backend.py#L180)；共享根因是 [foundation_stereo_backend.py:43](../../src/percepta/backends/foundation_stereo_backend.py#L43)。
- 原意：复用已有 `_DeadlinePipe` 给启动和完整RPC施加实际期限。
- 问题：该helper和stderr pump使用 `select.select()`，本机实现不能处理编号超过1023的fd。Linux允许更高fd；即使只启动一个worker，已有文件／socket较多的宿主也可能得到高编号管道。
- CPU反例：先打开1100个 `/dev/null`，然后使用现有ThresholdAdapter启动worker并准备8×8 RGB。
- 当前结果：`BackendUnavailable: ... filedescriptor out of range in select()`，worker被回收，health为终止错误；父提交同例可分割出64像素mask。root也独立复现了当前故障。
- 最小修法：共享期限IO和stderr等待统一使用 `poll()` 或 `selectors`，保留原单调期限、部分读写、EOF、长度限制和回收语义。
- 必要回归：在独立CPU测试进程中制造高fd，确认prepare、segment、stderr排空和close正常；同时保留已有启动／堵塞写入／部分响应期限测试。
- 不修的影响：SDK嵌入已有较多资源的宿主时，健康模型无法启动。小规模默认服务通常不会达到该条件。

**R3 · P2：缓存部分命中时，另一提示失败会丢掉命中结果（既有遗漏）**

- 位置：[pipeline.py:527](../../src/percepta/pipeline.py#L527)；同类提前返回还在UnsupportedPrompt和ToolError分支。
- 原意：命中部分使用缓存，未命中部分推理，最后按输入提示次序合并。
- 问题：异常分支直接返回 `_fail_percepts(misses)`，尚未执行后面的合并，因此 `cached` 中已有结果被整个丢弃。
- CPU反例：实际PerceptCache和已有RecordingTool先缓存 `red block`；令工具不可用，再请求 `[red block, blue block]`。
- 当前结果：只返回 `blue block / unavailable`；`red block` 没有任何Percept。预期应有缓存的red和失败的blue两条结果。
- 基线关系：`a761656` 已有同样提前返回，并非本次深复制或去重引入；本轮 #10/#11 的混合成功测试没有覆盖“部分命中＋未命中失败”。
- 最小修法：把未命中的失败结果送入统一排序／合并流程，保留命中结果；按提示出现位置关联，兼容重复提示及多相机。
- 必要回归：分别让未命中部分抛UnsupportedPrompt、ToolUnavailable、ToolError；检查结果数、顺序、cache_hit、状态和对象隔离。
- 不修的影响：批量请求静默缺结果，HTTP与SDK调用方无法按请求提示完整消费响应。

```text
warmup:    red block / ok
requested: red block, blue block
returned:  blue block / unavailable / cache_hit=False
```

**R4 · P2：超大JSON整数仍让融合返回HTTP500（旧审计 #13 的残余）**

- 位置：[fusion.py:181](../../src/percepta/fusion.py#L181)、[fusion.py:209](../../src/percepta/fusion.py#L209)、[fusion.py:230](../../src/percepta/fusion.py#L230)；plane转换还经过 [geometry.py:87](../../src/percepta/geometry.py#L87)。
- 已有修复：stereo/triview数值选项会验证类型、有限性和范围；极小prism step先封顶再转整数。
- 遗漏：轮廓融合仍有只捕获 `TypeError/ValueError` 的 `float()`／NumPy转换。`context: dict[str, Any]` 能接受401位JSON整数，转换成float时抛 `OverflowError`，未进入既有坏输入处理。
- 真实HTTP反例：CI配置、正常单相机红块及合法标定、fuse=True，传入 `table_z=10**400` 或 `table_plane=[0,0,1,10**400]`。两者均为HTTP500/internal；双视角 `prism_step_m=10**400` 同样可触发。
- 基线关系：父提交已有该转换缺口；本次 #13 未覆盖所有融合解析入口。该整数仅401字符，落在正常请求体大小限制内。
- 最小修法：在这些共享解析边界捕获OverflowError，进入现有的无效值警告／默认值处理；若要改为invalid_request，需要同步明确该路径的统一策略。
- 必要回归：真实HTTP覆盖上述三类输入；不能只测试单个float helper。断言不再逸出500，并检查既有无效值处理语义。
- 不修的影响：不合法的客户端数值会成为服务器内部错误，还可能触发SDK对5xx的重复重试。

```text
table_z     → 500 {"error":"internal","message":"unexpected error while perceiving"}
table_plane → 500 {"error":"internal","message":"unexpected error while perceiving"}
```

**二、应补修的扩展场景**

**R5 · P3：更换本地runner会关闭仍准备复用的cache（新增回归）**

- 位置：[client.py:499](../../src/percepta/client.py#L499)、同文件新runner的cache传入位置 [503](../../src/percepta/client.py#L503)；关闭动作在 [pipeline.py:254](../../src/percepta/pipeline.py#L254)。
- 原意：切换config／registry前关闭旧runner，防止遗留模型和worker。
- 问题：旧runner.close也会关闭cache；随后新runner仍接收调用方本次明确传入的同一个cache。
- CPU反例：PerceptCache子类实现正常close生命周期，关闭后get／put报错。同config两次调用的cache_hit为False、True；换成 `config.model_copy(deep=True)` 且传同一cache后，两次均False，cache.closed=True，日志为 `cache disabled after error: cache is closed`。
- 影响范围：仅显式注入实现close的cache。内置PerceptCache没有close，不触发该问题；推理结果仍能产生，所以列为较低优先级，而非默认部署故障。
- 最小修法：关闭旧runner前转移本次仍复用的cache所有权；让旧runner继续关闭其工具、让新runner最终负责关闭该cache。
- 必要回归：分别更换config／registry并复用可关闭cache，断言仍能命中；客户端最终close时cache只关闭一次。更换成另一个cache时，旧cache应正常释放。

**三、上一轮20项修复逐项核对**

下表中的“通过”仅表示该修复的主路径和所列反例已验证。R1–R5独立保留，不以全套绿测覆盖它们。
旧行为、当时红绿测试及D40–D43决策原文仍在 [修改记录](../documentation-changes.md) 和 [决策记录](../decisions.md)。

| 旧项 | 修改前为什么错 | 已落地的修法 | 本轮验证及残余 |
|---|---|---|---|
| 1 · SAM3旧图 | ndarray地址不是内容，原地换帧／地址复用会继续使用旧特征 | 删除跨请求身份缓存，每请求每相机prepare一次；同请求多提示共享图像特征 | mutable buffer回归通过；完整请求缓存仍能跳过推理 |
| 2 · 提示污染 | 官方processor会追加box，text也不自动清除几何提示 | 每次segment前原地reset_all_prompts，保留图像特征，不接收其None返回值 | text/box排列回归通过；已安装官方API的既有CPU状态探针证据保留 |
| 3 · 子进程无期限 | timeout只保存，启动及管道读写可能无限等待 | 启动预算与每次完整写读RPC预算分别执行，终止故障回收worker，stderr有界排空 | 普通启动／堵塞写入／部分响应／close回归通过；高fd仍有R2 |
| 4 · HTTP504超额积压 | handler超时就释放slot，但线程仍计算 | pending强引用及shield保留任务，done回调释放slot／inflight并取异常；shutdown先等待任务与预热 | timeout、handler取消、异常、shutdown次序回归通过；原生线程不可强停仍为方法边界 |
| 5 · 首次加载竞态 | 检查至发布无锁，首次并发可能加载两份模型；capabilities失败资源未发布也未清理 | entry RLock覆盖整个首次加载，capabilities成功才发布；失败best-effort close且保留原异常 | 并发首次加载及close再次报错回归通过 |
| 6 · 非物理标定 | 只校验矩阵形状，零T也能生成成功的世界几何 | NumPy-only共享K/T校验供schema、geometry、fuse使用 | HTTP400／直接ValueError及float32容差回归通过；未新增重型import |
| 7 · 加载失败擅自融合 | entry.kind未记录，失败geometry被误判为segmenter | 通过describe读取类声明kind，失败geometry也记为已选择阶段 | 不运行未选择的mask融合；失败结果可见，回归通过 |
| 8 · prepare变500 | prepare位于segment的不可用异常映射之外 | 在prepare边界将BackendUnavailable转ToolUnavailable | 对应unavailable回归通过；请求异常分类另见R1 |
| 9 · 坏协议当没找到 | 非法状态／尾随垃圾可能成为空mask | 严格校验阶段、status、JSON字段、重复key、数组、有限性和完整消费；ready仅启动 | 帧级反例通过；worker自己的请求异常分类仍有R1 |
| 10 · 响应污染缓存 | pipeline缓存读写都是浅复制 | pipeline两端deep copy，通用cache identity不变 | 嵌套Observation／mask／attributes／extras突变回归通过；混合失败输出另见R3 |
| 11 · 重复提示N²条输出 | 每次相同提示都追加整组结果 | 等效提示推理去重，按原输入次数／顺序复制回独立结果 | 禁用／冷／热／混合缓存的成功回归通过；异常组合仍有R3 |
| 12 · camera/pixel抬升 | frame相同就把Z差当米制竖直位移 | lifted要求两侧world；同一非world不给lift_m并标记world_frame_required | camera、pixel拒绝及原frame_mismatch行为通过 |
| 13 · 数值选项／溢出 | 无效sigma、分数、整数参数可产生负uncertainty或异常；极端中间计算可溢出 | stereo/triview共享验证；相对sigma权重；稳定median；高度候选先封顶 | 既有边界回归通过，真实fuse tiny step成功；轮廓融合超大整数仍有R4 |
| 14 · SGBM小图分配 | width≤16仍强制16视差，进入非法原生分配 | matcher前要求合法搜索范围和适合窗口，返回结构化invalid_request；保留合法clamp／偶数转奇数 | 8种真实OpenCV尺寸／窗口探针通过，包含17px最小宽度；不只依赖mock测试 |
| 15 · SDK readiness超期 | idle timeout不等于总期限，慢字节／状态行／headers／chunk framing／重试可突破预算 | readiness专用标准urllib opener及受控read／readline／read1；每跳更新remaining，拒绝迟到成功，统一TimeoutError | 新增相关回归通过；额外100 Continue、慢chunk trailer、慢chunk-size探针约81ms停止（80ms预算） |
| 16 · 替换runner泄漏 | 直接覆盖旧runner，客户端只会关闭最后一份资源 | 替换前关闭旧runner，按共享registry重新加载 | 模型／registry关闭顺序及替换失败回归通过；复用可关闭cache仍有R5 |
| 17 · CLI错误退出0 | unavailable也被shell当成功 | 任一percept／fused为unavailable、invalid_request、timeout则exit1；not_found为0，保留输出 | CLI状态回归通过 |
| 18 · health/hash失真 | worker退出仍显示ok，info未使用provenance的hash来源 | 后端非阻塞health_error快照；info读取_model_meta或现有hash字段 | metadata／registry回归通过；错误分类导致的假故障另见R1 |
| 19 · Options筛选无效 | Options只进入cache key，工具只读Prompt，优先级未定义 | 用户确认Options补默认、Prompt显式字段优先；请求验证复制合并，SDK序列化前解决 | 显式0.2／1、独立覆盖、HTTP／SDK往返及原Prompt不变的回归通过 |
| 20 · 三目退化标记 | 少用一组pair就标成两视角，即使覆盖三相机 | 用实际参与相机数决定degraded_to_two_view，subset警告保留 | 两组pair使用三相机时标记false，相关回归通过 |

**实现保持的契约与行为变化**

- 包仍为0.3.0，schema为percepta/v1，二进制布局／线协议版本不变；无新增默认依赖、权重或部署模型。
- 坐标仍为像素u=列/v=行、box右下角exclusive、OpenCV相机坐标、camera→world外参。
- K现在明确要求有限、正焦距、零skew和标准底行；T要求proper rigid、R正交且det=+1。R容差1e-6、skew／底行1e-8；此前可进入计算的非物理矩阵被拒绝是预期变化。
- 坏标定请求HTTP400；工具级不支持／不可用保留HTTP200中的Percept状态；not_found仍为正常200。R4的500是残余bug，不是该契约的例外。
- HTTP504后仍计算的任务继续占容量，因此随后503是预期行为；shutdown等待这些任务和预热。没有新增强杀原生推理线程的能力。
- 通用worker的timeout分别用于启动与每次完整RPC；首次prepare可能经历启动预算再经历prepare预算，不是两者共用一段总预算。
- SDK对受控响应读取施加deadline；同步DNS、底层连接／TLS建立、自定义阻塞health不能保证强制中断。没有新增后台取消线程；这项已声明的限制不重复计为新bug。
- 缓存未把context、标定和模型配置加入key；依赖它们的几何阶段仍应关闭缓存，现有神经双／三目部署配置如此。该已声明限制不等于本轮R3。

**四、本轮实际执行的验证**

| 验证 | 本轮结果 | 说明 |
|---|---|---|
| 完整CPU／已安装OpenCV测试 | 903 passed、9 skipped，68.10s | uv run --locked pytest -o addopts= -q；有1条既有Starlette/httpx弃用warning |
| Ruff | 通过 | uv run --locked ruff check . |
| 真实uvicorn端到端 | E2E OK | HTTP与进程内结果一致；not_found仍HTTP200 |
| backend／SAM3定向 | 142 passed、1 deselected | 单独跳过创建临时独立环境的用例；完整套已运行该用例 |
| metadata／registry定向 | 18 passed | 含并发首次加载、失败清理和hash来源 |
| 几何／选项／校验定向 | 186 passed | 加真实OpenCV 8种组合及tiny step探针 |
| service／SDK／CLI定向 | 63 passed | 加慢chunk trailer、100 Continue、chunk-size的真实loopback探针 |
| 本轮新问题反例 | R1–R5均已复现 | root独立核对每项；R1、R2有父提交对照；R3、R4核对父源码已有相同缺口 |
| 锁文件可移植性 | 6份均通过 | 依赖与锁文件未改动 |
| 报告与修改记录链接／规格镜像 | 通过 | 11个唯一相对链接目标存在；两份规格书逐字一致 |

分组计数有重叠，不能累加成新的总数。903通过表示现有测试通过，R1–R5的反例没有被现有测试覆盖。
本轮没有把反例加入产品测试或修复运行时代码，因此下述待办仍未关闭。

**五、后续修复顺序与验收要求**

1. R1：先补worker请求异常分类，保证请求错误不会永久关闭健康worker；保留真正不可用的终止测试。
2. R2：再修共享期限IO与stderr的高fd等待，覆盖通用、双目、三目共用该helper的调用者。
3. R3：合并命中结果与失败结果，覆盖重复提示、多相机和三种已声明工具异常。
4. R4：补全部轮廓融合数值解析的OverflowError处理，以真实HTTP断言状态和警告。
5. R5：处理runner替换时仍复用cache的生命周期，补最终close一次的回归。

每项应先增加能在9779a96失败的回归，再修实现；最后运行完整CPU、Ruff和真实HTTP e2e。
修改记录继续保留本报告的原行为和证据，追加修复时间／提交／结果，不能把本次待修状态改写成当时已修。

此外，前一轮已登记的3项辅助／插件API边界仍未处理：超大direct-fuse mask分配、PlaneGrid整数索引helper、
极端长宽比的grid上限。它们没有与R1–R5重复计数，也未在本轮扩成新的功能任务。

**未检查：** GPU forward／真实权重加载、任务精度、长期压力、跨机网络和systemd实机部署；不审计上游模型／虚拟环境依赖的全部内部实现。
**交付判断：** 原有修复有明确效果，但应先补R1–R4再将这批改动标记为可靠性问题已全部关闭；使用可关闭自定义cache的调用者还需R5。

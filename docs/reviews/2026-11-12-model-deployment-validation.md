# 五模型最终版本部署验收

- **日期**：2026-11-12，Asia/Shanghai（UTC+08:00）；实际运行约17:44–17:56。
- **验收代码**：`8255027af9d72bfa491d57f2b71d33eaead37b63`，`fix: keep transient perception failures out of cache`。
- **任务**：对最后可靠性修复后的五模型执行真实GPU、常驻进程、SDK／HTTP与故障处理验收。
- **结果**：五模型均通过，未发现新的部署阻塞；未修改运行源码、测试、配置、依赖、锁或权重。
- **机器可读记录**：[验收JSON](2026-11-12-model-deployment-validation.json)。包含完整checkpoint SHA256、环境版本、15份配置／锁／manifest快照及原始证据文件的SHA256。

## 1. 环境与模型身份

先查询GPU占用，分别使用物理GPU0／1／2，三目模型依次使用GPU3；均为NVIDIA H20，
驱动570.86.10。每个独立运行时都实际执行CUDA tensor运算，不只检查`is_available()`。
沿用已锁定的cu118／cu124／cu128环境，没有升级或迁移模型栈。

| 模型 | Python | PyTorch／bundled CUDA | 物理GPU | 资产校验 |
|---|---|---|---|---|
| SAM3 0.1.4 | 3.10.20 | 2.10.0+cu128／12.8 | 0 | checkpoint匹配配置，BPE和输入照片hash另记 |
| FoundationStereo 23-51-11 | 3.11.13 | 2.4.1+cu124／12.4 | 1 | checkpoint／cfg、固定源码与离线补丁、辅助backbone匹配清单 |
| Fast-FoundationStereo 23-36-37 | 3.12.13 | 2.6.0+cu124／12.4 | 2 | checkpoint／cfg及固定源码文件匹配清单 |
| MVSAnywhere HERO | 3.10.20 | 2.1.2+cu118／11.8 | 3 | checkpoint匹配配置，源码revision／manifest留档 |
| DA3-BASE | 3.12.13 | 2.11.0+cu128／12.8 | 3 | checkpoint／模型config匹配配置，源码revision／manifest留档 |

五模型实际使用的checkpoint均与原配置声明的SHA256相同。运行后再次核对预检记录中的
15份配置／锁／manifest，全部未变；运行源码及测试仍与上述验收代码相同。

## 2. 正常调用与结果一致性

| 模型 | 本轮真实输入与断言 | 结果 |
|---|---|---|
| SAM3 | 640×427真实花朵照片；text提示flower、top_k／排序、原图mask尺寸、权重来源hash；同图重复与常驻复用 | 通过；本地及真实HTTP均返回3实例，mask逐像素相同，box／score／面积／质心对照通过 |
| FoundationStereo | 官方960×540 RGB对，32次更新；3次常驻调用；禁网官方demo视差对照；真实HTTP重复、空mask | 通过；每次469,911个有效点，本地／HTTP全部Geometry字段精确相等 |
| Fast-FoundationStereo | 官方960×540 RGB对，8次更新、416视差；3次常驻调用；禁网官方demo对照；真实HTTP重复、空mask | 通过；官方视差最大绝对差0.0px，本地／HTTP全部Geometry字段精确相等 |
| MVSAnywhere HERO | 官方640×480三图；禁网内存API、重复／官方入口／共同世界变换对照；常驻工具与真实HTTP | 通过；三视图几何有效，本地／HTTP及重复调用的centroid一致；世界变换后深度最大差0.0m |
| DA3-BASE | 原图H×W=252×336的三图标定平面，处理为H×W=378×504；禁网加载、重复／基线尺度／大世界原点对照；常驻工具与真实HTTP | 通过；本地／HTTP及重复调用的centroid一致；基线翻倍时深度比2.0；大原点平移后深度最大差0.0m |

三目既有HTTP smoke显式比较Geometry.centroid，未在本轮逐字段比较全部Geometry；
SAM3对照使用同一个adapter的本地与HTTP路径，不是另一套独立官方实现。
SAM3另验证真实预热及SDK readiness：warmup=ok后真实照片正常，缓存命中仍保持相同像素结果。
DA3不支持的尺寸返回invalid_request，后续合法请求保持同一个已加载worker且成功。

记录的时间是本机单次验收数据，不是跨模型性能排名。例如SAM3本地首次调用18.29s，
真实服务就绪18.62s；FoundationStereo第2–3次工具调用819.61／815.74ms，FFS为277.28／277.63ms。
输入、初始化阶段和处理路径不同；完整时间、每阶段命令和显存数据保存在原始证据中。

## 3. 故障处理与资源清理

五模型分别验证：缺checkpoint、主动终止本次创建的worker，以及真实模型加载并成功推理后，
故意收紧下一次完整RPC期限。SAM3／FoundationStereo额外只暂停本次worker，模拟无响应。
这些是故障注入，用于验收错误处理，不能表述为模型正常运行时自行发生的错误。

| 模型 | 注入的完整RPC预算 | HTTP请求实际耗时（含编码／回收） | 结果 |
|---|---:|---:|---|
| SAM3 | 100ms，并暂停自有worker | 0.479s | unavailable；health=error；子进程回收、管道／stderr线程清理，后续不自动重启 |
| FoundationStereo | 100ms，并暂停自有worker | 0.959s | 同上 |
| Fast-FoundationStereo | 10ms | 0.742s | 同上 |
| MVSAnywhere HERO | 0.01ms | 0.470s | 同上 |
| DA3-BASE | 0.01ms | 0.287s | 同上 |

RPC预算约束完整管道写入／读取，不包含HTTP编解码和故障进程回收时间。
五模型缺权重时均为HTTP200中的unavailable、health=error，未发布已加载实例；
主动终止worker后均报告unavailable，不伪装为not_found或生成Geometry。
所有记录的本轮worker均已回收，检查时无percepta模型worker残留，真实loopback服务按context退出。

结束后有其他任务重新占用GPU，因此以自有PID、管道和线程的清理断言判断本轮资源释放，
不把整机GPU必须空闲作为验收条件。未干预其他任务。

## 4. 命令与证据

所有运行均通过既有uv环境，使用`--locked --no-sync`；原始证据位于本机Git忽略目录：

```text
checkpoints/validation-8255027-20261112-1745/
```

没有覆盖原有`checkpoints/<model>/smoke.json`。本轮主要入口如下，实际GPU按上表设置：

```bash
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0 PERCEPTA_SAM3_GPU=0 \
PERCEPTA_SAM3_PHOTO=checkpoints/validation-8255027-20261112-1745/sam3-flower.jpg \
uv run --locked --no-sync pytest -o addopts= -m gpu -q tests/gpu/test_sam3.py

CUDA_VISIBLE_DEVICES=1 uv run --locked --no-sync python scripts/foundation_stereo_smoke.py --http
CUDA_VISIBLE_DEVICES=1 uv run --locked --no-sync pytest -o addopts= -m gpu -q tests/gpu/test_foundation_stereo.py
CUDA_VISIBLE_DEVICES=2 uv run --locked --no-sync python scripts/fast_foundation_stereo_smoke.py --http --upstream
CUDA_VISIBLE_DEVICES=3 uv run --locked --no-sync python scripts/multiview_smoke.py --config configs/local-mvsanywhere.yaml --http
CUDA_VISIBLE_DEVICES=3 uv run --locked --no-sync python scripts/multiview_smoke.py --config configs/local-da3.yaml --http
```

MVSAnywhere另运行现有`scripts/mvsanywhere_smoke.py`；DA3另运行基于现有
`scripts.da3_smoke.run()`的禁网adapter探针。五模型故障探针、SAM3真实HTTP／cache对照、
计时器和环境探针源码副本与JSON／stdout／stderr均保存在上述目录，具体文件及hash见验收JSON。

- 本轮已有GPU pytest为 **SAM3 4 passed、FoundationStereo 2 passed**，无跳过。
- 其余使用既有真实smoke和补充assert，不将脚本阶段数量虚报成pytest测试数量，也未重复同等逻辑的GPU pytest。
- CPU验收仍引用此前同一运行源码的 **1030 passed、9 skipped**；这次未重复完整CPU suite。
- 配置／权重、源码／manifest身份、正常／故障断言和清理证据已交叉核对。

本批提交只保存文档和精简JSON证据，原始图片、视差数组、日志与权重继续留在忽略目录。
未修改模型依赖／资产，未发布、推送或长期启动服务；未进行任务精度、RoboDojo接入、
长期压力或跨机HTTP验证。

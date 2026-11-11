# 感知模型调研：能不能装、怎么装、值不值得装

> 调研时间 2026-11-11。对象是 `docs/Percepta_RoboDojo_Agent_Implementation_Plan.md` 里
> 列出的 D2/D3/D5/D6 四个神经模型。**只做信息核实，没有安装、没有下载权重、没有跑推理。**
>
> 结论先行：**四个里只有 FoundationStereo 一个值得优先接，Fast-FoundationStereo 紧随其后。
> MVSAnywhere 和 Depth Anything 3 都可以接但有前提条件**（见各自的"阻塞项"）。

---

## 总览

| 编号 | 模型 | 输入 | 权重获取 | 授权 | 判定 |
|---|---|---|---|---|---|
| D2 | **Fast-FoundationStereo** | 校正双目 | 需申请/Google Drive | NVIDIA Open Model Agreement | ✅ **优先** |
| D3 | **FoundationStereo** | 校正双目 | Google Drive 直链 | 研究用途（`LICENSE`） | ✅ **优先** |
| D5 | **MVSAnywhere** | N 视图 + 位姿 | 公开直链 | **非商业**（Niantic） | ⚠️ 可接，需坐标转换 |
| D6 | **Depth Anything 3** | N 视图，位姿可选 | HuggingFace | BASE/SMALL 是 **Apache 2.0** | ⚠️ 可接，尺度需处理 |

---

## D3 FoundationStereo（先接这个）

**仓库**：`NVlabs/FoundationStereo`（2990★，CVPR 2025 Oral，Best Paper Nomination）
**默认分支**：`master`（**不是 `main`**，README 是 `readme.md` 小写）

### 接口

```bash
python scripts/run_demo.py \
  --left_file ./assets/left.png \
  --right_file ./assets/right.png \
  --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth \
  --out_dir ./test_outputs/
```

**输出**：视差图 → 米制深度 / 点云。

### 关键约束（官方原文）

> The input left and right images should be **rectified and undistorted** ... the epipolar
> lines are horizontal between the left/right images.

> Do not swap left and right image. The left image should really be obtained from the
> left-side camera.

**这两条正是 `stereo_3d` 已经在做的事**（`_rectify` 用 `cv2.stereoRectify` 做校正）。
所以接线成本低：校正后的两张图直接喂进去。

### 点云需要额外参数

> To get point cloud for your own data, you need to specify the intrinsics. In the intrinsic
> file in args, 1st line is the flattened 1x9 intrinsic matrix, 2nd line is the baseline.

**第二行是基线**——和我们 `Z = f·B/d` 的用法一致，可以直接对上。

### 权重

| 模型 | 说明 | 链接 |
|---|---|---|
| `23-51-11` | ViT-large，精度最好 | Google Drive 文件夹 |
| `11-33-40` | ViT-small，更快 | 同上 |

放进 `./pretrained_models/<name>/model_best_bp2.pth`。

### 环境

```bash
conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn   # 必须单独装
```

**`flash-attn` 要单独安装**，官方明确说放一起会报错（issue #20）。

### 阻塞项

1. **权重在 Google Drive**，需要能访问。国内网络可能要走代理。
2. **需要 `flash-attn`**，编译耗时且对 CUDA 版本敏感。H20 + CUDA 12.8 需实测。
3. **非商业授权**。研究场景可用。

### 接入方式

```
stereo_3d 工具
  → backend="foundation_stereo"
  → subprocess backend（独立 uv_envs/envs/foundation-stereo/）
  → worker 收校正后的左右图 + K + B
  → 返回视差图
  → percepta 侧复用现有 Z=fB/d 与反投影
```

**复用面很大**：校正、反投影、不确定度、Geometry 构造全都不用改，只替换视差计算。

---

## D2 Fast-FoundationStereo

**仓库**：`NVlabs/Fast-FoundationStereo`（1589★，CVPR 2026）
**默认分支**：`master`，README 是 `readme.md` 小写

### 与 FoundationStereo 的关系

官方原文：

> This model is designed for real-time applications. For offline computation for the best
> accuracy, please checkout our earlier work FoundationStereo.

**同一套接口思路**，参数量更小、速度更快。

### 关键参数（之前分析过，这里确认）

> `--max_disp` Maximum disparity for volume encoding, **192 should be enough, unless you need to
> sense very near objects (e.g. <0.1m)**. Increasing it runs slower and uses more memory.

**`max_disp` 是可调参数，不是硬上限。** 我们实测 RoboDojo 需要 ~400 px（HL）到 ~600 px（LR）
才能在 0.4 m 工作，所以必须调大到 ~640。

### 权重

| 来源 | 用途 |
|---|---|
| `nvidia/c-fast-foundationstereo`（HF） | **商业**用，NVIDIA Open Model Agreement |
| Google Drive `23-36-37` 等 | 研究用 |

**注意**：商业版和研究版是两套。研究用途拿 Drive 的。

### 环境

```bash
pip install torch==2.6.0 torchvision==0.21.0 xformers --index-url .../cu124
pip install -r requirements.txt
```

与 FoundationStereo 的 conda 环境不同，**需要独立环境**。

### 阻塞项

与 FoundationStereo 类似：Google Drive 权重 + 非商业授权。**没有 flash-attn 的额外麻烦**。

---

## D5 MVSAnywhere

**仓库**：`nianticlabs/mvsanywhere`（CVPR 2025）

### 接口

```bash
CUDA_VISIBLE_DEVICES=0 python src/mvsanywhere/run_demo.py \
    --name mvsanywhere \
    --config_file configs/models/mvsanywhere_model.yaml \
    --load_weights_from_checkpoint weights/mvsanywhere_hero.ckpt \
    --data_config_file configs/data/nerfstudio/nerfstudio_empty.yaml \
    --scan_parent_directory /path/to/recordings/ \
    --scan_name recording_0 \
    --image_height 480 --image_width 640
```

### 输入格式：这是最大的接入成本

它**不吃内存里的数组**，而是吃**磁盘上的目录结构**：

```
recording_0/
│-- images/
│   │-- image_0.png
│   │-- image_1.png
│   └-- transforms.json
```

`transforms.json` 是 NeRF/nerfstudio 格式，包含逐帧的 `transform_matrix`（camera-to-world）
和 `fl_x/fl_y/cx/cy`。

**这意味着 worker 每次推理都要写临时目录**：3 张 PNG + 1 个 JSON。可行，但比 FoundationStereo
的"直接喂数组"重。

### 权重

**公开直链，无需申请**：

```
https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt
https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_dot.ckpt
```

这是四个模型里**权重最容易拿的**。但 `storage.googleapis.com` 在国内可能需代理。

### 坐标约定：必须转换

官方原文：

> This repo uses the notation "cam_T_world" to denote a transformation from world to camera...
>
> **the model weights we provide assume a ScanNet coordinate system.** This is important since we
> include ray information as part of metadata. Other datasets used with these weights should be
> transformed to the ScanNet system.

**这是最容易出错的地方**：percepta 用 OpenCV 相机系（`T_world_cam`，看向 -z），而权重
假设 ScanNet 系。**必须做一次明确的坐标变换**，否则深度方向会错——而且不会报错，只会给出
看起来合理的错误结果。

**这条必须写进 worker 的注释和测试里。**

### 环境

```bash
make create-mamba-env   # 用 mamba，不是 conda
pip install -e .
```

**需要 mamba**，我们的 uv 体系里得手工翻译 `environment.yml`。

### 阻塞项

1. **非商业授权**（Niantic，Patent Pending）。
2. **ScanNet 坐标系转换**——最大风险点。
3. **磁盘 IO 格式**——每次推理写临时目录。
4. 需要 `mamba`（可翻译成 uv）。

### 价值

**"原生三视角联合推断"**——这是 `triview_3d` 的 `mvs` 后端。与 `pairs`（三组双目融合）
是本质上不同的方法，值得对比。

---

## D6 Depth Anything 3

**仓库**：`ByteDance-Seed/Depth-Anything-3`

### 接口：四个模型里最干净的

```python
from depth_anything_3.api import DepthAnything3
model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE")
model = model.to(device="cuda")
prediction = model.inference(images)          # images = list of paths 或数组

prediction.depth        # [N, H, W] float32
prediction.conf         # [N, H, W] float32  ← 自带置信度
prediction.extrinsics   # [N, 3, 4]          ← 还能估位姿
prediction.intrinsics   # [N, 3, 3]
```

**直接吃 `list of paths`，支持多视图，输出带置信度。** 没有目录结构要求，没有坐标系转换。

### 关键优势：**Apache 2.0**

| 模型 | 参数量 | 授权 |
|---|---|---|
| `DA3-BASE` | 0.12B | **Apache 2.0** ✅ |
| `DA3-SMALL` | 0.08B | **Apache 2.0** ✅ |
| `DA3-LARGE` | 0.35B | CC BY-NC 4.0 |
| `DA3-GIANT` | 1.15B | CC BY-NC 4.0 |
| `DA3NESTED-GIANT-LARGE` | 1.40B | CC BY-NC 4.0 |

**BASE 和 SMALL 是 Apache 2.0，可商用。** 这是四个模型里唯一没有授权障碍的。

### 尺度问题

官方 FAQ：

> **Monocular Metric Depth**: To obtain metric depth in meters from `DA3METRIC-LARGE`, use
> `metric_depth = focal * net_output / 300.`

- `DA3METRIC-LARGE`（单目）需要 `focal * net_output / 300` 才是米制
- `DA3NESTED-*` 输出**已经是米制**
- 其他 Any-view 模型输出**相对深度**

**接入时必须明确用哪个模型、要不要自己乘尺度。** 用错了不会报错，只会差一个常数倍。

### 环境

```bash
pip install xformers torch>=2 torchvision
pip install -e .
```

### 阻塞项

1. **尺度**：相对深度 vs 米制必须分清（见上）。
2. **xformers**：老 GPU 有兼容问题（官方 issue #11）。
3. 与 `triview_3d` 的契合度：它输出 `[N,H,W]` per-view 深度 + 置信度，正好是 `pairs` 后端
   需要的输入格式。

### 价值

- 唯一 Apache 2.0
- 自带 `conf` 置信度，可以直接进我们的不确定度体系
- 还能交叉校验我们的相机标定（它估的 `extrinsics` 可以和 RoboDojo 给的对比）

---

## 建议的接入顺序

| 顺序 | 模型 | 理由 |
|---|---|---|
| **1** | **FoundationStereo** | 接口最干净（校正双目 → 视差），与我们现有 `_rectify` 直接对接；精度上限最高 |
| **2** | **Depth Anything 3 (BASE)** | Apache 2.0 无授权障碍；自带置信度；能交叉验证标定 |
| 3 | Fast-FoundationStereo | 与 D3 同源，接口类似，可作为速度对照 |
| 4 | MVSAnywhere | 权重最好拿，但坐标转换 + 磁盘 IO 成本最高；作为"原生三视角"的对照 |

---

## 三个共同的技术前提

**1. 都需要正版权重，且都不小**

FoundationStereo ViT-large、DA3-GIANT 1.4B。下载和存储要预留空间（本机磁盘只剩 132 GB，
需确认）。

**2. 都要独立环境**

和 SAM3 一样走 `uv_envs/<model>/`。理由相同：torch/CUDA 版本互相冲突，且不能让主库依赖变重。

- FoundationStereo：conda + `flash-attn`（单独装）
- Fast-FoundationStereo：cu124 + xformers
- MVSAnywhere：mamba
- DA3：xformers

**四个都要从 `environment.yml` / `requirements.txt` 翻译成 `pyproject.toml` + `uv.lock`。**

**3. 只有 `pairs` 路线不用等模型**

`triview_3d` 的 `pairs` 后端（三组 SGBM 融合）**现在就能跑**，不需要任何权重。它给了
模型接入后的对照基线。

---

## 环境实测（已核实，非估计）

在 `ls-aigc-08`（本工作区所在机器）上实测：

| 项 | 结果 | 影响 |
|---|---|---|
| 磁盘 `/mnt` | **100 GB 可用**（2.0T 用了 1.8T，95%） | 偏紧。四个模型权重 + 环境估计 30–60 GB，装得下但没余量 |
| 现有 `checkpoints/` | 3.3 GB（SAM3 权重） | — |
| `storage.googleapis.com` | **可达**（HTTP 400 是路径问题，不是网络） | MVSAnywhere 权重能下 ✅ |
| `drive.google.com` | **可达**（302 重定向正常） | FoundationStereo 权重能下 ✅ |
| `huggingface.co` | **可达**（200） | DA3 权重能下 ✅ |

**网络不是障碍**（走了 `127.0.0.1:8118` 代理）。**磁盘是需要注意的一项**：100 GB 装四个模型
要精打细算，建议先装一个、验证通了再装下一个，不要一次全下。

---

## 需要你决定的

1. **优先级**：先接 FoundationStereo（精度上限最高、接口最贴合），还是先做 DA3-BASE
   （Apache 2.0 无授权障碍）？

   我倾向 **FoundationStereo**：它的输入要求（校正后的左右图）正好是 `stereo_3d` 已经产出的，
   接线成本最低，而且能立刻验证"神经立体能不能改善我们实测的 0.5–1.7 mm"。

2. **授权**：非商业研究用途没问题吧？要是有商用打算，四个里只有 `DA3-BASE` / `DA3-SMALL`
   （Apache 2.0）能用。

3. **磁盘**：100 GB 够装 1–2 个模型。要不要先清理一些空间，或者只装先验证的那一个？

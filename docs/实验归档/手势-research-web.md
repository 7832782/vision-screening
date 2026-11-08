> **归档说明（2026-11-08）**：这是原始实验报告的原样拷贝。原始实验目录
> （`_gesture_lab/`、`单音识别方案/`）已于同日删除，所以**本文里的相对链接都打不开**，
> 且其中「下一步」「待办」之类的内容已不再适用。要看最终结论请回
> [../AI模块验收.md](../AI模块验收.md)。

# 外部方案调研：4 方向「食指指向」手势识别（轻量 / 准确 / 离线 / 国内可用）

- 调研人：gesture-survey（teammate）
- 日期：2026-11-07
- 共享任务：task-1
- 范围：只做外部资料核实与公开数字整理，**不跑本机模型**；本机文件仅做只读检查（包体积、包结构）。
- 唯一写入文件：本文件。

---

## 0. 证据等级约定（全文通用）

| 标记 | 含义 |
|---|---|
| **【官方】** | Google / MediaPipe / MMPose / Ultralytics / caniuse 等一手文档或官方模型卡 |
| **【官方包】** | 我直接读 / 拉取官方发布的包（.task 文件、npm 包、d.ts），数字由文件本身得出 |
| **【社区】** | 开发者博客、GitHub issue、社区模型作者自述；可信度不等，明确标注 |
| **【推算】** | 我基于公开参数做的算术推导，不是实测 |
| **【未验证】** | 没有可靠公开数字，或无法在本机复现；请 Lead 本地实测 |

> 说明：Google 官方文档域名已从 `ai.google.dev/edge/...` 迁移到 `developers.google.com/edge/...`（部分旧链接 301 到新页或直接消失）。下文同时给新旧链接。

---

## 1. 结论先行（TL;DR）

1. **没有现成的「4 方向指向」公开模型**。MediaPipe Gesture Recognizer 的内置手势只有 8 类，其中与指向有关的**只有 `Pointing_Up` 一个**（没有 down / left / right）。所以「手部关键点」路线一定是**HandLandmarker + 自研 4 类判定**，不是拿一个模型直接用。【官方】
2. **这条路线在本机最划算的形态是**：`hand_landmarker.task`（7.46 MiB，palm detector 2.23 MiB + landmark 5.23 MiB）跑 21 个手部关键点，再在**关键点坐标上**做 4 分类。理由是 Google 自己的手势分类头就是这么做的：官方 `gesture_recognizer.task` 里，8 类手势分类器只有 **7,773 字节**（TFLite），embedder 546,000 字节——整个"手势智能"不到 0.6 MB，输入是 21×3 关键点而不是像素。【官方包】
3. **精度证据足够支撑**：官方手势分类模型卡给出 SS-F1 **95.5%（肤色加权）/ 93.9%（性别加权）**，且该模型卡明确写"pipeline 只吃 hand landmarks，不直接处理任何图像像素"。【官方】
4. **风险点不是"能不能分 4 类"，而是远距离**。社区权威实现（depthai_hand_tracker）明确写：*"The palm detector model from Google Mediapipe was trained to detect hands that are less than 2 meters away from the camera"*，并给出手腕/手掌上下文预聚焦（Body Pre Focusing）作为远距补救。【社区】本机 2 米、手约 100px 的场景正好压在这条边界上，**必须实测**。
5. **老 Android WebView 跑 tasks-vision 有实打实的体积风险**：`@mediapipe/tasks-vision@1.2.2` 的 wasm 是 **14,603,354 字节（13.93 MiB）**，且该版本**不再发布 `nosimd` 版本**；0.10.21 还有 `nosimd`（9,448,638 B）。SIMD 需要 Chrome/Edge ≥ 91（2021 年）、Safari/iOS ≥ 16.4。【官方包】【官方数据】
6. **Model Maker 已官方弃用**（Google 员工 2025-11 在 issue 中写 "MediaPipe Model Maker is now deprecated and is no longer actively maintained"），且 PyPI 包要求 `tensorflow>=2.10,<2.16`、分类器只到 Python 3.10 —— **本机 Python 3.12 装不上**。所以「Model Maker 自定义手势」在本项目里应直接排除，改用自己在 landmark 上训 MLP/逻辑回归。【官方】【官方包】

**推荐排序（详见 §9）**：
> A. HandLandmarker + 自研 4 类判定（规则角度 / 小 MLP，同一底座二选一）
> B. HandLandmarker + GestureRecognizer 内置头当"是否在指"的闸门（只覆盖 Pointing_Up，4 方向仍需自研）
> C. RTMPose hand（精度更硬，但需要检测器 + 2.58 GFLOPs，重）
> D. TFJS handpose + fingerpose（浏览器路线，权重 ~12 MB，老设备性能差）
> E. YOLO11n-pose 手部社区权重（AGPL 许可地雷 + 开手掌偏好）
> F. OpenVINO hand pose（无现成手部模型，只算推理运行时）
> G. OpenCV 肤色 + 凸包/凹陷（只适合演示/兜底）

---

## 2. 项目侧已知约束（来自任务描述与只读检查，供对照）

- 目标：受试者只伸食指，指向上/下/左/右；**4 类**；越轻量越准确越好；**离线**；国内可用（无云端）。
- 现有 v0.5：MediaPipe Pose（肩→腕向量）4 方向。
- 本机：Windows 11，`D:\tools\python312\python.exe`，mediapipe **1.2.2**、opencv **5.0.0**、onnxruntime **1.30.0**（本次只读检查确认）。
- 已下载（本次只读检查，字节数精确）：
  - `_gesture_lab/models/hand_landmarker.task` = **7,819,105 B**（7.46 MiB）
  - `_gesture_lab/models/gesture_recognizer.task` = **8,373,440 B**（7.99 MiB）
- 包内结构（本次用 zipfile 读出，**【官方包】**证据）：

| 外层包 | 内部文件 | 字节数 | 说明 |
|---|---|---|---|
| hand_landmarker.task | `hand_detector.tflite` | 2,339,878 | palm detector |
| hand_landmarker.task | `hand_landmarks_detector.tflite` | 5,478,949 | 21 点回归 |
| gesture_recognizer.task | `hand_landmarker.task` | 7,819,105 | 内嵌上面那包 |
| gesture_recognizer.task | `hand_gesture_recognizer.task` | 554,062 | 手势头（内层 zip） |
| └ hand_gesture_recognizer.task | `gesture_embedder.tflite` | 546,000 | 21 点 → 128 维嵌入 |
| └ hand_gesture_recognizer.task | `canned_gesture_classifier.tflite` | **7,773** | 8 类分类头 |

> 也就是说：**"从关键点判断手势"这件事，Google 的生产级分类头只有 7.8 KB。** 这是本项目"自研 4 类"路线可行性的最强体积证据。

---

## 3. 方案逐条清单

### 3.1 MediaPipe Hand Landmarker（Python Tasks API）——**首选底座**

【官方】文档：<https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker>
（旧链 <https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker>）

**官方给出的事实（逐条可核对）**

| 项 | 值 | 来源 |
|---|---|---|
| 模型结构 | 模型束 = palm detection + hand landmarks（两段式） | 官方文档 Models 节 |
| 输入尺寸 | 192×192（检测器）、224×224（landmark 回归） | 官方文档 Models 表 + 模型卡 |
| 量化 | float16（`hand_landmarker/float16/latest`） | 官方文档 Models 表 |
| 输出 | 21 个 3D 关键点（图像归一化坐标）+ 21 个世界坐标（米）+ 左右手 | 官方文档 Task output |
| numHands | 默认 **1**，可设 >0 | 官方文档 Configuration options |
| 阈值默认 | minHandDetectionConfidence 0.5 / minHandPresenceConfidence 0.5 / minTrackingConfidence 0.5 | 官方文档 Configuration options |
| 运行模式 | `IMAGE` / `VIDEO` / `LIVE_STREAM`（Python 三种都有，LIVE_STREAM 需 `result_callback`） | 官方文档 |
| VIDEO 关键机制 | 用上一帧 landmark 推的 bbox 定位手，**只有跟踪失败/手消失才重跑 palm detection** | 官方文档「Because running the palm detection model is time consuming…」 |
| 官方延迟 | Pixel 6：CPU **17.12 ms** / GPU **12.27 ms**（整条 pipeline 平均） | 官方 Task benchmarks |
| 训练数据 | ~30K 真实图片 + 合成手模型 | 官方文档 / 模型卡 |
| License | 模型卡写 **Apache License 2.0**；npm 包 `license: Apache-2.0` | 模型卡 PDF / npmmirror 元数据 |

**精度（模型卡，MNAE = 按手掌尺寸归一化的平均绝对误差，越小越好）**【官方】
模型卡：<https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf>

| 评测集 | Lite MNAE | Full MNAE |
|---|---|---|
| 14 个地理区域（700 图） | 12.02% | **10.09%** |
| 6 种肤色（Fitzpatrick，420 图） | 5.67% | **5.08%** |
| 性别 | 5.67% | **5.09%** |

> 解读：官方自己没有给"4 方向分类准确率"，只有关键点误差。MNAE 10% 量级意味着手指尖会有可见抖动；所以**不要指望直接把 `INDEX_FINGER_TIP - INDEX_FINGER_MCP` 的 atan2 四舍五入就稳**，需要时序平滑 + 死区（建议 Lead 在评测里对比"单帧规则" vs "N 帧投票"）。

**模型卡明确写的 Out-of-scope / 限制（对本项目非常关键）**【官方】
- 不适合：手套/遮挡（手里拿东西、首饰、纹身、美甲）；人群计数；监控/身份识别。
- "The model has not been tested in 'in-the-wild' smartphone camera conditions, including low-end devices, low light, motion blur"。
- "Predicting landmarks within the crop of **prominently displayed** hands" —— 官方定位就是"画面里显著的手"，**没有承诺远距小手**。

**Python 用法（官方文档原文，VIDEO 模式）**【官方】
```python
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

with HandLandmarker.create_from_options(options) as landmarker:
    for mp_image, timestamp_ms in decoded_video_frames:   # 时间戳必须单调递增
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
```
- 关键点索引（官方图）：`0` WRIST，`5` INDEX_FINGER_MCP，`6` PIP，`7` DIP，`8` INDEX_FINGER_TIP（官方文档 21 点表）。
- 左右手注意：官方文档写明 handedness 是"受试者的手"，**传入水平镜像图会翻转左右手**；旧文档还说 handedness 假定输入是自拍镜像。做"左右指向"时要先固定是否 flip，否则逻辑会反。（社区 issue 也复现过这点：<https://github.com/google-ai-edge/mediapipe/issues/6376>）

**优点**：官方维护、Apache-2.0、Python/JS/Android/iOS 同 API、单包 7.5 MiB、VIDEO 模式省掉重复 palm detection、本机已装 mediapipe 1.2.2。
**风险**：① 远距/小手（见 §6）；② 必须能看到手掌（见 §6.2）；③ 关键点抖动；④ 官方遥测（见 §7.3）。

---

### 3.2 MediaPipe Hand Landmarker（浏览器 tasks-vision）——同一模型，浏览器壳

【官方】文档同上（页面含 Web 代码）；包：`npm i @mediapipe/tasks-vision`（<https://developers.google.com/edge/mediapipe/solutions/setup_web>）。

- Web API：`HandLandmarker.createFromOptions(vision, { baseOptions: { modelAssetPath, delegate: "GPU" }, runningMode: "VIDEO", numHands: 2 })` → `detectForVideo(video, performance.now())`。**Model asset 可指向本地相对路径**（`modelAssetPath` 也支持 `modelAssetBuffer`），所以可完全离线自托管。
- 体积与浏览器限制见 §7（这是本方案唯一真正的"重"点）。

---

### 3.3 MediaPipe Gesture Recognizer（内置手势 / 自定义头）

【官方】文档：<https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer>

**内置手势只有 8 类（官方表格，逐行抄录）**

| ID | Label | 手势 |
|---|---|---|
| 0 | `Unknown` / `None` | 不认识的姿态 |
| 1 | `Closed_Fist` | 握拳 |
| 2 | `Open_Palm` | 张开手掌 |
| 3 | **`Pointing_Up`** | 食指向上指 |
| 4 | `Thumb_Down` | 拇指下 |
| 5 | `Thumb_Up` | 拇指上 |
| 6 | `Victory` | 剪刀手 |
| 7 | `ILoveYou` | ASL "I love you" |

> **结论：内置模型只认"指上"，不认"指下/左/右"。** 直接用内置 8 类无法满足 4 方向需求。

**但它支持接自研分类头**（官方配置项 `customGesturesClassifierOptions` / Python `custom_gesture_classifier_options`）："You can extend the model bundle to recognize more gestures by bundling your own custom gesture classifier."；且"both canned and custom 都识别时优先 custom"。这意味着**理论上**可以把 4 方向分类头塞进 gesture_recognizer.task。但官方 Model Maker 已弃用（§3.4），要自己产出合规 `.task` 包袱较重；**不建议**为此走这条路，直接用 HandLandmarker + 自己分类更简单。

**官方延迟**：Pixel 6 CPU **16.76 ms** / GPU **20.87 ms**（整条 pipeline）。【官方】

**手势分类模型卡（关键证据）**【官方】
<https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf>
- 输入：`1×63` 屏幕关键点 + handedness 标量 + `1×63` 世界关键点；**"This pipeline runs on hand landmarks and related information for a single hand, but does not directly process any images (i.e. RGB pixel data)."**
- 结构：embedder（FC + residual）→ 128×1 嵌入 → 分类头（FC）→ 8 维概率。
- 精度：SS-F1 **95.5%**（Monk 肤色加权，范围 94.3%–97.8%）；**93.9%**（性别表达加权，93.1%–94.4%）。
- 明确不适用：多手手势、**有动作的手势**（挥手）、手语翻译。
- 训练数据规模（用于比较）：肤色评测约 2000 图，性别评测约 8000 图。

> 这张卡基本就是"在 21 个关键点上训一个小分类器能到什么水平"的官方答案：**8 类、SS-F1 95% 量级、模型 7.8 KB**。4 类（更简单）理论上不应更差，但那是在他们的评测分布上；本项目老人手型/距离/背景属于分布外，**必须以本地数据实测为准**。

---

### 3.4 MediaPipe Model Maker（自定义手势训练）——**已弃用，本项目不可用**

| 事实 | 证据 |
|---|---|
| 官方定位：迁移学习，支持 object detection / **gesture recognition** / image·text·audio 分类；"done with less data" | 【官方】<https://developers.google.com/edge/mediapipe/solutions/model_maker>（注：该 URL 现已 301 到 solutions 首页，内容来自搜索索引快照；存档链接 <https://web.archive.org/web/20251220232629/https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer> 的导航仍列有 Model Maker 与 "Customize" 入口） |
| 自定义手势指南（HaGRID 数据集，端到端 Colab） | 【官方历史页】<https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer>（现 301 到 gesture_recognizer 概览页） |
| PyPI 版本 **0.2.1.4**，发布于 **2024-04-22**；`requires: tensorflow>=2.10,<2.16`、`tf-models-official<2.16`；classifiers 只列 **Python 3.8 / 3.9 / 3.10** | 【官方包】<https://pypi.org/project/mediapipe-model-maker/> |
| **已弃用**：Google 员工（kuaashish，2025-11）在 issue 中写 "MediaPipe Model Maker is now deprecated and is no longer actively maintained"；建议 Python 3.10、避免 3.11+ | 【官方（issue 内回复）】<https://github.com/google-ai-edge/mediapipe/issues/5214> |
| Model Maker 对 **hand landmarker 不可用**：官方回复 "Model Maker tool, though it is currently unavailable for hand landmarker" | 【官方（issue 内回复）】<https://github.com/google-ai-edge/mediapipe/issues/5667> |
| 数据量口径："只需约 100 张/类" | 【官方视频】<https://www.youtube.com/watch?v=vODSFXEP-XY>（"you only need about 100 images for each category"）；【社区】<https://samproell.io/posts/ai/asl-detector-with-mediapipe-wsl>（引 Model Maker 文档 "approximately 100 examples per class"） |

> **结论**：Model Maker 这条路在本项目里是死路——① 本机 Python 3.12 不在支持矩阵；② TF 版本被钉死 <2.16；③ 官方已弃用、不做维护；④ 它本来也**不训练 hand landmarker**，只训练手势分类头。**要自定义就在 landmark 特征上自己训**（§3.10）。

---

### 3.5 TensorFlow.js handpose + fingerpose（纯浏览器备选）

| 事实 | 数值 | 来源 |
|---|---|---|
| `@tensorflow-models/handpose` 模型权重 | **约 12 MB** | 【官方】Kaggle 模型卡 <https://www.kaggle.com/models/tensorflow/handdetector>（"~12MB of weights"） |
| 官方性能口径 | 40 FPS（2018 MacBook Pro）、35 FPS（iPhone 11）、**6 FPS（Pixel 3）** | 同上 |
| TFJS 3D HandPose（GHUM）2D mAP | Lite **79.2** / Full **83.8**；上一代 TFJS HandPose **66.5** | 【官方博客】<https://blog.tensorflow.org/2021/11/3D-handpose.html> |
| 3D 平均误差 | Lite 1.4 cm / Full 1.3 cm | 同上 |
| fingerpose | 基于 21 点做**规则型**手势描述：每个手指给 curl（No/Half/Full）+ direction（**Vertical Up / Vertical Down / Horizontal Left / Horizontal Right** + 4 个对角）| 【社区/开源】<https://github.com/andypotato/fingerpose> |
| fingerpose License | **MIT**（GitHub license API 确认） | 【官方包】<https://api.github.com/repos/andypotato/fingerpose/license> |

> 形态："handpose 出 21 点 → fingerpose 用方向+弯曲度打分 → 自定义 4 个 GestureDescription（Index: NoCurl + Vertical Up/Down/Horizontal Left/Right，其它手指 FullCurl）"。
> 这是**现成代码层面最接近 Pointing 4 方向**的组合，但代价是 12 MB 权重 + 老设备 6 FPS 级别的性能（Pixel 3 官方数字），且 TFJS 依赖 WebGL。**只适合当第二备选**。
> 另注：fingerpose 的方向枚举天生就是 8 向，做 4 向只需选 4 个方向并调 match score。

---

### 3.6 Ultralytics YOLO 手部关键点（yolo11n-pose 手部权重）

| 事实 | 数值/说明 | 来源 |
|---|---|---|
| 官方手部数据集 | **26,768 张**（18,776 train / 7,992 val），21 关键点/手，下载 **~369 MB**；标注**由 Google MediaPipe 生成** | 【官方】<https://docs.ultralytics.com/datasets/pose/hand-keypoints> |
| 训练流程博客 | "YOLO11 Hand Pose Estimation" | 【官方博客】<https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11> |
| 社区训练好的权重 | `chrismuntean/YOLO11n-pose-hands`（~99 stars，Kaggle 26K 数据集） | 【社区】<https://github.com/chrismuntean/YOLO11n-pose-hands> |
| 作者自述局限 | "the trained model **struggles to accurately detect keypoints for gestures like pinching and swiping**. However, it performs well in identifying keypoints on **open hands**, whether facing forward or backward." | 同上 README |
| YOLO11n-pose 基准 | 2.9 M 参数；CPU ONNX **52.4 ms**；权重文件 **~6 MB**；mAPpose 50-95 = 50.0（COCO 17 点，非手部） | 【官方】<https://huggingface.co/Ultralytics/YOLO11>、<https://docs.ultralytics.com/models/yolo11> |
| License | **AGPL-3.0**（强 copyleft）："permits distribution and commercial use with source-code disclosure and prohibits sublicensing; proprietary use requires a commercial license"；官方在 issue 里进一步坚持"任何使用模型/架构/代码都需 AGPL 或企业授权" | 【官方】<https://www.ultralytics.com/legal/responsible-responsible-use-policy>、<https://github.com/ultralytics/ultralytics/issues/21505> |

> 判断：**"只伸食指、其它手指弯曲"恰好落在作者说的"pinching/swiping 这类不擅长"的区间**（不是 open hand）。再叠加 AGPL 对"闭源离线产品"的传染性要求、以及训练标签本身就是 MediaPipe 产出（精度上限 = MediaPipe），**不推荐**。
> 另注：数据集页面未明确写数据许可，商用前需核对（**未验证**）。

---

### 3.7 RTMPose / mmpose hand（精度更高的重型备选）

| 事实 | 数值 | 来源 |
|---|---|---|
| 模型 | RTMPose-m hand5（**手部专用**），输入 **256×256**，21 点 | 【官方】<https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose_hand5.md> |
| 精度 | PCK@0.2 **81.5**（COCO-WholeBody-Hand）、**96.4**（Hand5 合并集）；AUC 83.9；EPE 5.06 | 同上 |
| 计算量 | **2.581 GFLOPs** | 同上 |
| 权重 | checkpoint `rtmpose-m_simcc-hand5_...pth`（体积未在该页给出，**未验证**） | 同上 |
| 部署形态 | 官方是 **top-down**：需要手/人检测器（RTMDet-nano 等）先给 bbox | 【官方】mmpose demo / model zoo |
| 推理速度参考 | RTMPose 论文表（**body 模型**，i7-11700 + ONNXRuntime 1 线程）：RTMPose-t 3.204 ms、RTMPose-m 11.060 ms（**不是 hand 模型数字**） | 【官方论文】<https://arxiv.org/html/2303.07399v2> |
| 移动端现成转换 | `litert-community/RTMPose-Hand-LiteRT`：256×256 NCHW fp16，Pixel 8a GPU 已验证跑通 | 【社区/官方社区】<https://huggingface.co/litert-community/RTMPose-Hand-LiteRT> |
| License | MMPose **Apache-2.0**（开源库通用许可） | <https://github.com/open-mmlab/mmpose> |

> 判断：**如果 MediaPipe 在 2 米处关键点抖到无法分方向，RTMPose-hand 是最合理升级路径**（PCK 96.4 在 Hand5 上明显强于 MediaPipe 的 MNAE 口径，但两者指标不同，不能直接换算）。代价：需要检测器、2.58 GFLOPs、ONNX/OpenVINO 部署链、以及自己接 4 分类。

---

### 3.8 OpenVINO hand pose

- **Open Model Zoo 里没有 21 点手部关键点模型**；OMZ 的手相关条目是 body pose（如 `human-pose-estimation-0001`：AP 42.8%、15.435 GFLOPs、4.099 M 参数）。【官方】<https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/human-pose-estimation-0001/README.md>
- 实际可行做法是**把 MediaPipe/RTMPose 的 tflite/ONNX 转成 OpenVINO IR**：社区项目 `depthai_hand_tracker` 就是"tflite → OpenVINO IR → MyriadX blob"的完整脚本链（含归一化参数说明：palm detector 期望 RGB [-1,1]，landmark 期望 RGB [0,1]）。【社区】<https://github.com/geaxgx/depthai_hand_tracker>
- 本机是 **AMD R5 5600 + RX 7650 GRE**，没有 Intel GPU/NPU 红利，OpenVINO 只剩"CPU 推理可能略快"这一点点空间（**未验证**，本机可实测）。
- 判断：**OpenVINO 不是方案，只是运行时**。除非要上 Intel 一体机/工控机，否则不建议为它增加一条构建链。

---

### 3.9 OpenCV 肤色 + 凸包 / 轮廓凹陷（传统启发式）

- 原理链：肤色/YCrCb 阈值分割 → 最大轮廓 → `convexHull` + `convexityDefects` → 用凹陷点数量/位置数手指。教程级资料：
  - OpenCV 官方轮廓教程（含 convexityDefects）：<https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html>
  - 论文（IJARCCE 2022，肤色分割 + 凸包 + 凹陷数指）：<https://ijarce.com/wp-content/uploads/2022/05/IJARCCE.2022.114214.pdf>（检索结果给出的域名是 `ijarce.com`，期刊站常写作 `ijarcce.com`，**链接可用性未验证**）
  - 工程博客（凸包 + 凹陷找指尖）：<https://medium.com/netscape/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0>
- **精度证据：没有任何公开基准**（这些教程都是"干净背景 + 手掌正对"的演示）。对"老人手部皮肤、室内杂背景、2 米远 100px、逆光"的场景，**未验证且大概率不可用**。
- 优点：零模型、几乎 0 延迟、可解释；适合当"摄像头是否有人/手在画面"的极低成本兜底，不适合做 4 方向判定。

---

### 3.10 自研 4 类判定（在 21 个 landmark 上）——**推荐落地形态**

**为什么可行（三条独立证据）**
1. Google 自己的手势分类器就是 landmark-only 的 8 类 FC 网络，模型 **7,773 B**，SS-F1 **95.5%/93.9%**。【官方】【官方包】
2. 社区成熟实现 `kinivi/hand-gesture-recognition-mediapipe`（Apache-2.0）：MediaPipe 21 点 → 归一化到 wrist → **小型 MLP** → 3 类（含 "pointing"）；仓库里 `keypoint_classifier.tflite` 只有 **6,352 B**，训练数据 `keypoint.csv` 约 **3.6 MB**。<https://github.com/kinivi/hand-gesture-recognition-mediapipe>
3. Instructables 完整项目（Windows/Pi4）：恰好就是 **Up / Down / Left / Right** + 4 个对角 + Fire 共 9 类；每类**采集 60 组**（共 540 条 CSV），Keras MLP（128-64-softmax、Dropout 0.3），Windows 约 30 FPS、Pi4 约 6 FPS；作者结论"landmark 方案训练/推理都比像素方案快得多"。<https://www.instructables.com/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/>（代码：<https://github.com/Thomas9363/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe>）

**两种判定实现（建议都做，互为校验）**

| 实现 | 方法 | 需要的标注 | 优点 | 风险 |
|---|---|---|---|---|
| A1 规则法 | 取 `INDEX_FINGER_TIP(8) - INDEX_FINGER_MCP(5)`（或 MCP→TIP 的 PCA 主轴）求 `atan2(-dy, dx)`，落到 4 象限；用 `PIP/DIP` 的伸展度 + 其余手指弯曲度做"确实只伸食指"的闸门；再做 N 帧中值/投票 | 0 张（无需训练） | 零训练、可解释、可 debug、立刻能跑 | 手旋转/腕屈时方向会歪；单帧抖；老人可能"手指伸不直" |
| A2 学习法 | 用 A1 的特征（21×3 归一化坐标 + 若干几何特征：各指间角度、伸展度）训 **逻辑回归 / 2 层 MLP（≤64 单元）**；输出 4 类 + "none" | 50–100 段/类（社区经验：100/类；60/类也跑通） | 抗抖动、可吸收个体差异；模型 <10 KB | 需要采集真实数据（老人手）；有分布外风险 |

**体积预期【推算】**：4 类、输入 63 维、隐层 32–64，权重 ≈ 2–5 k 参数，fp32 约 **10–25 KB**（同量级证据：kinivi 6,352 B 的 3 类模型、官方 7,773 B 的 8 类模型）。对 7.5 MB 的主模型完全不构成负担。

**数据量参考汇总**
- 官方视频口径：约 100 张/类（Model Maker）。
- 社区项目：60 组/类（9 类）即可跑通演示；Model Maker 博客约 100/类。
- Google 手势模型卡的评测集规模（肤色 2000 图 / 性别 8000 图）说明"要证明泛化"需要上千级样本，但**要跑通 4 类**几十级就够。【社区】【官方】

---

## 4. 有没有直接做「Pointing Up/Down/Left/Right」的现成项目 / 模型？

**结论：没有可直接拿来用的成品模型。** 有的是"用现成 landmark 拼出来的项目"，清单如下：

| 项目 | 形态 | 覆盖 | License | 说明 |
|---|---|---|---|---|
| Instructables「How to Train Custom Hand Gestures Using Mediapipe」 | Python + MediaPipe + Keras，9 类（含 Up/Down/Left/Right） | 4 方向 ✓ | 文章 CC BY-NC-SA（代码在 GitHub，许可未标） | 与本项目需求几乎一一对应，但**未针对老人/2 米/小目标**做验证；作者自述 Pi4 只有 6 FPS |
| `kinivi/hand-gesture-recognition-mediapipe` | Python，landmark MLP，3 类（含 pointing） | 只到"在指"，无方向 | Apache-2.0 | 工程结构可直接抄（CSV 采集 + TFLite 推理） |
| `andypotato/fingerpose` | JS 规则库，8 个 FingerDirection 枚举 | 4 方向可组合 ✓ | MIT | 不提供模型，只提供"方向评分"逻辑；依赖 TFJS handpose |
| MediaPipe Gesture Recognizer 内置 | TFLite | 只有 `Pointing_Up` | Apache-2.0（随包） | **缺 down/left/right** |
| MDPI「Dynamic Hand Gesture Recognition Using MediaPipe and Transformer」 | 论文，8 类（含 swipe up/down/left/right） | 动态滑动方向，非静态指向 | 论文 | 参考价值有限（要序列输入 + Transformer，重） |

> 这解释了为什么"评估手部关键点路线"必然落到"自己写 4 类判定"：**上游没有成品，但上游把"判定"这件事的成本压到了 8 KB 级别**。

---

## 5. 与现有 Pose（肩→腕）路线的关系（设计层面对比）

| 维度 | Pose 肩→腕（v0.5） | HandLandmarker + 4 类（候选） |
|---|---|---|
| 能否确认"只伸食指" | **不能**：握拳、张掌、竖拇指、指人的手臂方向都一样 | 能：PIP/DIP 伸展度 + 其余手指弯曲度 |
| 方向语义 | 手臂方向；若受试者腕部弯曲，方向与手指方向不一致 | 手指本身方向 |
| 2 米远距 | Pose 全身模型本来就为远距设计（人体在画面里占比大），更稳 | 手只占 ~100px，压在 palm detector 的边界上（§6） |
| 体积 | pose_landmarker_lite ~5 MB 级（本次未核实，**未验证**） | 7.46 MiB（已实测） |
| 依赖 | mediapipe（已装） | mediapipe（已装） |
| 失败模式 | 只露手不露肩/身体 → 无法判 | 手背对镜头/戴手套/画面里手太小 → 无法判 |

> 建议评估时**同时测两条**并看"错误类型"：Pose 路线的错误是"方向对但手势错"（假接受），手部路线的错误是"检测不到"（假拒绝）。对老人筛查，**假拒绝比假接受更可接受**（可以提示"把手举高一点"），所以手部路线在需求覆盖度上更优。

---

## 6. 远距（2 米 / 手约 100px）与近距（40 cm 特写）的可行性

### 6.1 官方怎么说

| 说法 | 来源 |
|---|---|
| palm detector 必须覆盖"**large scale span (~20x) relative to the image frame**"（约 20 倍尺度跨度），并能处理遮挡/自遮挡 | 【官方历史文档】<https://mediapipe.readthedocs.io/en/latest/solutions/hands.html>（该页是官方 v0.7.5 文档镜像，标注 forwards 到新 hand_landmarker 页） |
| palm detection AP **95.7%**（对比无 decoder 基线 86.22%） | 同上 |
| 模型卡定位："**prominently displayed** hands"；未在低端设备/低光/运动模糊下测试 | 【官方模型卡】PDF 链接见 §3.1 |
| **没有**任何官方"最小手部像素 / 最远距离"参数 | 官方 hand_landmarker 页无此配置项（只有 3 个置信度阈值）——**这一点本身就是证据：官方不承诺距离** |

### 6.2 社区怎么说（最有价值的一条）

`geaxgx/depthai_hand_tracker` README（作者把 MediaPipe 模型移植到 OAK-D，长期维护）：
> - "The palm detector model from Google Mediapipe was **trained to detect hands that are less than 2 meters away from the camera**. So if the person stands further away, his hands may not be detected."
> - "And **the padding used to make the image square before feeding the network contributes even more to the problem**."
> - 补救：**Body Pre Focusing**（用 Movenet 找手腕 → 把手部 ROI 裁出来再喂 palm detector）；"Use Body Pre Focusing when the person can be at more than 1.5 meter from the camera."
> - "When the person is far, **good image definition and ambient lighting have a big impact**… Using a higher resolution (`--resolution ultra`) may help."
> - "It is important to note that **the further the distance, the more difficult the tracking is**, because fast move appear fuzzier on the image."
> 【社区】<https://github.com/geaxgx/depthai_hand_tracker>（README "Body Pre Focusing" 节）

> **这是本次调研里对"2 米"最直接的一条公开结论**：2 米正好是官方训练分布的上界，超出就要靠身体上下文/更高分辨率救。本项目 v0.5 已经有 Pose，恰好可以复用"肩/腕"做**和 depthai 同款的预聚焦**——这是低成本的增强项，不是必须项。

### 6.3 近距（40 cm 大特写）的风险

- 官方模型卡的输入是"palm detector 在整图上找手掌 → 裁 224×224 手部区域"。**40 cm 时手可能超出画面或只露出部分**，palm detector 的锚框/NMS 对"半只手"不友好（**推算**，未实测）。
- 官方对"只露手指不露手掌"的答复：*"detecting only fingers without a palm is beyond the scope of the MediaPipe hand landmarker Task API"* —— 谷歌员工在 issue 中的原话。【官方（issue 回复）】<https://github.com/google-ai-edge/mediapipe/issues/5667>
- 实际建议：40 cm 特写时**要求受试者把手完整放在画面内、手背/掌心朝向镜头**；评测集里应有"半只手出画"的负样本（Lead 的 dataset 已有 `ai_neg_corner_*` 之类的角落小目标，建议再补"半手出画"）。

### 6.4 小目标几何推算【推算】

以 640×480 采集、手掌约 100px 宽为例：
- palm detector 输入 192×192：100px / 640px ≈ 15.6% 画面宽 → 缩放后约 **30 px**（≈ 检测器输入的 15.6%）。能否稳定检出**取决于训练时的尺度分布**（官方只说覆盖 ~20x 尺度跨度，未给下限），**必须实测**。
- landmark 模型输入 224×224 是从检测框裁出后缩放：若检测框约 150px，放大到 224 约 1.5 倍；若检测框只有 60px，放大近 4 倍 → **插值模糊 → 指尖抖动放大**，这正好对应"方向判定抖"的现象。
- 提高分辨率的收益是线性的：把采集从 640×480 提到 1280×720，100px 的手在 192×192 输入里变成约 60px，**强烈建议评测时对比 640 vs 1280 两档**（depthai 作者也是这个建议：用更高分辨率）。

---

## 7. 浏览器端（老旧 Android WebView）跑 tasks-vision 的可行性

### 7.1 体积（官方包精确数字）

`@mediapipe/tasks-vision` npm 包（jsDelivr data API + npmmirror 元数据）【官方包】：

| 版本 | wasm 文件 | 字节数 | JS glue | bundle |
|---|---|---|---|---|
| **1.2.2**（latest，与 Python mediapipe 1.2.2 对齐） | `vision_wasm_internal.wasm` | **14,603,354**（13.93 MiB） | 334,128 | 156,062 |
| | `vision_wasm_module_internal.wasm` | 14,603,372（13.93 MiB） | 334,166 | — |
| | **`nosimd` 版本** | **未发布**（包内无此文件，package.json exports 也只列 internal/module） | — | — |
| 0.10.21（旧版） | `vision_wasm_internal.wasm` | 9,574,032（9.13 MiB） | 204,284 | 137,735 |
| | `vision_wasm_nosimd_internal.wasm` | **9,448,638（9.01 MiB）** | 204,137 | — |

- 整个包 unpacked **31,269,184 B**（31.3 MB），tarball 8,743,225 B（npmmirror `dist`）。【官方包】<https://registry.npmmirror.com/@mediapipe/tasks-vision/latest>
- 数据来源：<https://data.jsdelivr.com/v1/packages/npm/@mediapipe/tasks-vision@1.2.2?structure=flat>、<https://data.jsdelivr.com/v1/packages/npm/@mediapipe/tasks-vision@0.10.21?structure=flat>

**离线 Web 应用的首次加载量级【推算】**：
> wasm 14.6 MB + 模型 7.5 MB ≈ **22 MB**（若用 0.10.21 的 nosimd 则 ≈ 17 MB）。对比现在的 `index.html` 单文件零依赖，这是数量级变化——**这是浏览器路线最大的产品成本，不是技术难度**。

### 7.2 SIMD / WebGL2 要求

- **SIMD 是可选但有版本门槛**：官方 `FilesetResolver` 文档："verifies whether SIMD is supported in the current environment and **loads the SIMD files only if support is detected**"。【官方】<https://developers.google.com/edge/api/mediapipe/js/tasks-vision.filesetresolver>（此句来自官方 API 文档正文；我未逐字复核页面正文，引用自检索索引，标注为"官方文档，未逐字复核"）
- 浏览器支持（caniuse 数据 JSON）：Chrome **91+**、Edge **91+**、Firefox **89+**、Safari/iOS **16.4+**；全球支持率 96.78%。【官方数据】<https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/wasm-simd.json>
  - **老 Android WebView 的坑**：WebView 版本跟系统/厂商更新走，Android 7–9 设备常见 Chromium 60–90，**没有 SIMD**。1.2.2 又不发 nosimd 包 → 这些设备可能直接跑不起来（**未验证**，需真机实测）。
- **WebGL2**：GPU delegate 与 `MPImage` 的 ImageBitmap 转换要求 WebGL2 —— 官方 `vision.d.ts` 原文：*"Converting to and from ImageBitmap requires that the MediaPipe task is initialized with an `OffscreenCanvas`. As we require WebGL2 support, this places some limitations on Browser support"*。【官方包】<https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision.d.ts>
  - WebGL2 在 Android WebView 上通常可用（2017+ 设备），但老设备的 GPU 驱动常有 `glReadPixels` 兼容性问题——社区 issue 有实例：<https://github.com/google-ai-edge/mediapipe/issues/5950>（"GL_INVALID_OPERATION: glReadPixels: format and type incompatible…"，gles2_cmd_decoder）。【社区】
- **降级路径**：
  1. `delegate: "CPU"`（不用 GPU，纯 wasm）——最稳，但老设备更慢；
  2. 固定 **0.10.21**（有 nosimd 构建）并自托管 wasm，文件命名不能改（官方 `FilesetResolver` 要求 "Wasm files are published without renaming"）；
  3. 完全不用浏览器：走 Python/本地 exe（本机已装 mediapipe）。

### 7.3 一个容易忽略的合规点：遥测

官方 npm README 的 Privacy Notice 原文：
> "processing of the input data … takes place on device, and MediaPipe does not send that input data to Google servers."
> "**MediaPipe Tasks APIs send metrics about the performance and utilization of the APIs in your app to Google.**"
【官方包】<https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/README.md>

> 对本项目：① 图像不出设备，隐私主体没问题；② 但"完全离线"严格说会被这层**性能遥测**打破，医院/筛查场景的网络审计可能追问。可控（断网运行即可），但要在设计决策里**明确写出来**，别让评审认为"零联网"。Python 包是否同样上报，**未验证**（建议 Lead 抓包确认）。

### 7.4 国内可用性

| 资源 | 现状 | 证据 |
|---|---|---|
| Python wheel | 本机已装 mediapipe 1.2.2（PyPI 源，国内可走清华/阿里镜像） | 本机只读检查（【官方包】级别：已安装） |
| npm 包 | **npmmirror 已同步 1.2.2**，`npm i @mediapipe/tasks-vision --registry=https://registry.npmmirror.com` 可用；license Apache-2.0、tarball 8.7 MB | 本次实测 HTTP 200：<https://registry.npmmirror.com/@mediapipe/tasks-vision/latest> |
| 模型 `.task` | 官方 URL 在 `storage.googleapis.com`（Google 存储域名）。国内直连通常不可用/极慢（**社区共识，本次未在国内网络实测**）。**但一旦下载即完全离线**：本机两个 .task 已下载，代码用本地相对路径加载即可 | 官方 Models 表 URL；本机文件存在 |
| wasm | 可自托管（官方明确支持 `FilesetResolver.forVisionTasks("/local/wasm")`），**不要依赖 jsDelivr CDN**（国内 jsDelivr 可用性不稳定，**未验证**） | 官方 setup_web 文档 |
| 遥测 | 见 §7.3 | 官方 README |

> 结论：**"国内可用"在本项目里的正确解法是"把 7.5 MB 模型（+ 浏览器时再加 14 MB wasm）随包分发/预装"**，开发期用镜像下载一次即可。不要写成"运行时从 CDN 加载"。

---

## 8. 对比表（核心）

> 体积单位：MiB（1 MiB = 1,048,576 B）；"延迟"一列只写有出处的数字，其余写"未验证"。
> 精度一列区分"关键点误差"与"分类准确率"，两者不可混用。

| 方案 | 模型体积 | 依赖 | 平台 | 输出 | 延迟（公开数字） | 精度证据 | License | 国内离线可行性 |
|---|---|---|---|---|---|---|---|---|
| **HandLandmarker + 自研 4 类**（首选） | 7.46 MiB（detector 2.23 + landmark 5.23）+ 分类器 0.01 MB | mediapipe 1.2.2（已装） | Python / JS / Android / iOS | 21×3 关键点 → 自算 4 类 | Pixel 6：CPU 17.12 ms / GPU 12.27 ms（官方） | 关键点 MNAE full 10.09%（官方）；4 类准确率无公开数字（**未验证**） | Apache-2.0 | ✅ 模型本地加载，全离线 |
| GestureRecognizer（内置 8 类） | 7.99 MiB（手势头 0.53） | mediapipe 1.2.2 | Python / JS / Android / iOS | 8 类（只有 Pointing_Up） | Pixel 6：CPU 16.76 ms / GPU 20.87 ms（官方） | SS-F1 95.5% / 93.9%（官方，8 类） | Apache-2.0 | ✅ 但**不满足 4 方向** |
| GestureRecognizer + 自定义头 | 7.99 MiB + 自训头（<0.1 MB） | 需自造 .task；Model Maker 已弃用 | 同上 | 自定义类别 ✓ | 同上 | 无官方数字（**未验证**） | Apache-2.0（包） | ⚠️ 造包链复杂 |
| TFJS handpose + fingerpose | **~12 MB** 权重 | @tensorflow-models/handpose（+tfjs）、fingerpose | 浏览器 | 21 点 + 规则手势 | 官方：40 FPS（MBP2019）/35（iPhone11）/**6（Pixel 3）** | 2D mAP 66.5（旧 TFJS HandPose）/79.2/83.8（3D GHUM） | handpose Apache-2.0；fingerpose MIT | ⚠️ 12 MB + WebGL 依赖 |
| YOLO11n-pose 手部社区权重 | ~6 MB（COCO 版）；手部权重体积见仓库 | ultralytics + PyTorch/ONNX | Python / ONNX | 21 点 | CPU ONNX 52.4 ms（官方，COCO 17 点） | 无公开手部 mAP；作者自述**不擅长 pinching/swiping，擅长开手** | **AGPL-3.0**（商用需企业授权） | ⚠️ 许可风险 + 手型偏差 |
| RTMPose-m hand5 | 未在官方页给出（**未验证**） | mmpose +（top-down）检测器 + ONNX/OpenVINO | Python / ONNX / LiteRT | 21 点 | 论文 body 模型 CPU 3.2–11 ms（非 hand） | PCK@0.2 **81.5**（COCO-WholeBody-Hand）/ **96.4**（Hand5），AUC 83.9 | Apache-2.0 | ✅ 但部署链最重 |
| OpenVINO hand pose | 无现成手部模型 | OpenVINO + 自转 IR | Intel CPU/GPU/NPU | 取决于所转模型 | 无（**未验证**） | 无 | Apache-2.0（工具） | ⚠️ 本机无 Intel 硬件红利 |
| OpenCV 肤色+凸包/凹陷 | 0 | opencv（已装） | Python / 任意 | 手部轮廓/手指计数 | 无模型，几乎 0 | **无任何公开基准** | Apache-2.0 | ✅ 但精度不可用 |
| MediaPipe Model Maker 自定义 | 需 TF 训练环境 | tensorflow>=2.10,<2.16；Python 3.8–3.10 | Colab/Linux | 自定义 .task | — | — | Apache-2.0 | ❌ 官方弃用 + Py3.12 装不上 |
| Pose（现有 v0.5） | pose 模型（**未核实**） | mediapipe | 任意 | 33 点 → 肩腕向量 | 未核实 | 未核实 | Apache-2.0 | ✅ 已上线 |

---

## 9. 推荐排序与理由

### 9.1 总体排序（Python 本机 / 一体机优先）

**1）HandLandmarker(VIDEO) + 自研 4 类判定（规则 A1 与 MLP A2 都实现，取二者交集或投票）**
- 轻量：7.46 MiB 模型（已下载），分类器 10–25 KB。
- 准确：官方 pipeline 的 21 点本身是 SOTA 级移动端方案；"判定"部分 Google 自己用 7.8 KB 的头做到 SS-F1 95% 量级。
- 离线/国内：Apache-2.0，模型本地文件，零 CDN。
- 需求覆盖：能验证"只伸食指"（Pose 做不到）。
- 风险与对策：远距（§6）→ 提高采集分辨率 + 可选的 Pose 预聚焦；抖动 → 时序平滑 + 死区；近距 → 引导"手完整入画"。

**2）HandLandmarker + GestureRecognizer 内置头当闸门**
- 只用来回答"这是不是在指"（Pointing_Up），再用自研 4 类断开"上/下/左/右"。好处是有一个独立模型的第二意见；坏处是多 0.53 MB + 一次分类开销，且内置头只有"指上"样本（对"指下"可能判成 None）。**可选，不是必需。**

**3）RTMPose-m hand5（+ 手部检测器）**
- 只在 MediaPipe 的 2 米表现被实测证伪时启用。精度口径最硬（Hand5 96.4 PCK），Apache-2.0；但部署链、GFLOPs、依赖都重一个档。

**4）TFJS handpose + fingerpose**
- 浏览器路线的最小改动版（fingerpose 直接有 4 方向枚举），但 12 MB 权重 + Pixel 3 级 6 FPS 的官方数字，**不建议给"老旧 Android WebView"用**；新平板/新手机可作备选。

**5）YOLO11n-pose 手部社区权重**
- 仅当愿意接受 AGPL 或买企业授权、且能自己重训时考虑；现成权重对"非 open hand"不擅长，恰好是最需要的姿态。

**6）OpenVINO hand pose**
- 不是独立方案；只在目标机器确定是 Intel 平台时作为加速层。

**7）OpenCV 肤色+凸包**
- 只做 0 成本兜底/演示，不参与主判定。

### 9.2 分场景建议

- **本机 Python 评估（当前 Lead 在做的事）**：直接 `hand_landmarker.task` + VIDEO 模式 + 自研判定；先跑 dataset 里 4 方向 + 负样本，产出混淆矩阵；再决定是否需要 A2 学习法。
- **安卓一体机 / kiosk**：同上（MediaPipe Android tasks-vision 同 API，Apache-2.0）；或用 Python 起本地服务 + 摄像头（若一体机是 Windows）。
- **保持 `index.html` 单文件零依赖**：手部模型路线在浏览器端意味着 **~22 MB 资源**与 WebGL2/WebGL 兼容性检查，属于产品级决策；如果一定要做，**固定 tasks-vision 0.10.21（有 nosimd）+ 自托管 wasm + CPU delegate** 是兼容老 WebView 的唯一稳妥组合（**需真机验证**）。
- **如果最终选"不改浏览器、不加资源"**：继续用 Pose 路线，但要接受"无法验证只伸食指"这个需求缺口，并把判定逻辑改成"手臂方向 + 置信度门槛 + 提示重做"。

---

## 10. 需要 Lead 本地实测 / 我标记为「未验证」的清单

1. **2 米 / 手 ~100px 的检出率与方向抖动**：官方无最小手部像素，社区说 palm detector 训练上限约 2 米。建议在 dataset 的 `far` 组上跑 HandLandmarker，统计"检出率 / 4 类混淆矩阵 / 指尖像素抖动（相邻帧标准差）"。采集分辨率建议 640 与 1280 各跑一遍。
2. **40 cm 特写**：手是否完整入画、palm 是否可见；`hand_landmarks_detector` 对超大手部裁切后的效果。
3. **规则法（A1）的准确率**：单帧 vs N 帧投票；4 象限边界附近（45° 附近）的错误率。
4. **自研分类器（A2）所需数据量**：50 / 100 / 200 段 per class 的学习曲线（本项目老人手型与社区数据分布不同）。
5. **Python HandLandmarker 延迟**：官方只有 Pixel 6（17.12 ms CPU）。本机 R5 5600 的实测值缺失。
6. **tasks-vision 1.2.2 在无 SIMD 环境是否还能跑**：1.2.2 包内没有 nosimd wasm，`FilesetResolver` 的降级行为在无 SIMD 的 WebView 上未验证；建议直接测 0.10.21 作为兜底。
7. **WebGL2/GPU delegate 在老 Android WebView 的稳定性**：社区有 `glReadPixels` 报错案例（issue #5950），需真机。
8. **Python 包是否也发遥测**：官方只对 Tasks JS 有明示 privacy notice；Python 行为未验证（可断网抓包）。
9. **`index.html` 零依赖约束下引入 ~22 MB 资源的可行性**（GitHub Pages 首屏、离线缓存策略）。
10. **Pose 路线的模型体积/延迟数字**（本报告未核实 v0.5 所用 pose 模型的具体文件）。

---

## 11. 参考链接汇总（按证据等级）

**官方文档 / 模型卡**
1. Hand Landmarker 概览：<https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker>
2. Gesture Recognizer 概览（8 类手势表）：<https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer>
3. Hand Tracking 模型卡（MNAE / Apache-2.0 / out-of-scope）：<https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf>
4. 手势分类模型卡（landmark-only / SS-F1 / 128 维嵌入）：<https://storage.googleapis.com/mediapipe-assets/gesture_recognizer/model_card_hand_gesture_classification_with_faireness_2022.pdf>
5. MediaPipe Hands 历史文档（palm AP 95.7% / ~20x 尺度跨度 / model_complexity）：<https://mediapipe.readthedocs.io/en/latest/solutions/hands.html>
6. Web 端安装与 wasm 托管：<https://developers.google.com/edge/mediapipe/solutions/setup_web>
7. FilesetResolver（SIMD 检测说明）：<https://developers.google.com/edge/api/mediapipe/js/tasks-vision.filesetresolver>
8. Model Maker 概览（现 301 到 solutions 首页）：<https://developers.google.com/edge/mediapipe/solutions/model_maker>
9. 自定义手势指南（HaGRID，旧链）：<https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer>
10. Model Maker PyPI 元数据（TF <2.16、Py 3.8–3.10、0.2.1.4）：<https://pypi.org/project/mediapipe-model-maker/>
11. Ultralytics Hand Keypoints 数据集：<https://docs.ultralytics.com/datasets/pose/hand-keypoints>
12. Ultralytics 合规策略（AGPL 强 copyleft）：<https://www.ultralytics.com/legal/responsible-responsible-use-policy>
13. Ultralytics YOLO11 基准表：<https://docs.ultralytics.com/models/yolo11>、<https://huggingface.co/Ultralytics/YOLO11>
14. TFJS 3D HandPose 官方博客（mAP / 3D 误差）：<https://blog.tensorflow.org/2021/11/3D-handpose.html>
15. TFJS handpose 官方模型卡（~12 MB / FPS）：<https://www.kaggle.com/models/tensorflow/handdetector>
16. caniuse WebAssembly SIMD 数据：<https://raw.githubusercontent.com/Fyrd/caniuse/main/features-json/wasm-simd.json>（页面：<https://caniuse.com/wasm-simd>）
17. RTMPose 论文（CPU 延迟表）：<https://arxiv.org/html/2303.07399v2>
18. MMPose RTMPose-hand5 精度表：<https://github.com/open-mmlab/mmpose/blob/main/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose_hand5.md>
19. OpenVINO OMZ body pose（说明无手部 21 点模型）：<https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/intel/human-pose-estimation-0001/README.md>
20. OpenCV 轮廓/convexityDefects 教程：<https://docs.opencv.org/4.x/d5/d45/tutorial_py_contours_more_functions.html>

**官方发布的包（我用数据 API / 文件本身核对）**
21. `@mediapipe/tasks-vision@1.2.2` 文件与体积：<https://data.jsdelivr.com/v1/packages/npm/@mediapipe/tasks-vision@1.2.2?structure=flat>
22. `@mediapipe/tasks-vision@0.10.21` 文件与体积（含 nosimd）：<https://data.jsdelivr.com/v1/packages/npm/@mediapipe/tasks-vision@0.10.21?structure=flat>
23. npmmirror（国内 npm 镜像）元数据：<https://registry.npmmirror.com/@mediapipe/tasks-vision/latest>
24. tasks-vision 1.2.2 的 `vision.d.ts`（WebGL2/OffscreenCanvas 原文）：<https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/vision.d.ts>
25. tasks-vision 1.2.2 的 README（privacy notice / 遥测）：<https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.2/README.md>

**社区（开发者/issue）**
26. depthai_hand_tracker（"palm detector 训练上限 2 米" / Body Pre Focusing / 分辨率建议）：<https://github.com/geaxgx/depthai_hand_tracker>
27. MediaPipe issue #5667（只露手指不露手掌 → out of scope；Model Maker 不用于 hand landmarker）：<https://github.com/google-ai-edge/mediapipe/issues/5667>
28. MediaPipe issue #5214（Model Maker 已弃用）：<https://github.com/google-ai-edge/mediapipe/issues/5214>
29. MediaPipe issue #5950（老 Android WebGL `glReadPixels` 报错）：<https://github.com/google-ai-edge/mediapipe/issues/5950>
30. MediaPipe issue #6376（handedness 与镜像）：<https://github.com/google-ai-edge/mediapipe/issues/6376>
31. `kinivi/hand-gesture-recognition-mediapipe`（landmark MLP，3 类含 pointing，Apache-2.0）：<https://github.com/kinivi/hand-gesture-recognition-mediapipe>
32. Instructables（Up/Down/Left/Right 9 类 + 每类 60 样本 + Pi4 6 FPS）：<https://www.instructables.com/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/>
33. `andypotato/fingerpose`（8 方向枚举，MIT）：<https://github.com/andypotato/fingerpose>
34. `chrismuntean/YOLO11n-pose-hands`（"不擅长 pinching/swiping，擅长开手"）：<https://github.com/chrismuntean/YOLO11n-pose-hands>
35. Ultralytics YOLO11 手部关键点博客：<https://www.ultralytics.com/blog/enhancing-hand-keypoints-estimation-with-ultralytics-yolo11>
36. `litert-community/RTMPose-Hand-LiteRT`（Android LiteRT 跑通）：<https://huggingface.co/litert-community/RTMPose-Hand-LiteRT>
37. MediaPipe Model Maker ASL 实战博客（~100 图/类）：<https://samproell.io/posts/ai/asl-detector-with-mediapipe-wsl>
38. 官方 YouTube「Model training with MediaPipe Model Maker」（"about 100 images for each category"）：<https://www.youtube.com/watch?v=vODSFXEP-XY>
39. OpenCV 手部肤色+凸包论文（IJARCCE 2022）：<https://ijarce.com/wp-content/uploads/2022/05/IJARCCE.2022.114214.pdf>（域名拼写待核，**链接可用性未验证**）
40. 凸包+凹陷找指尖（工程博客）：<https://medium.com/netscape/simple-hand-gesture-recognition-using-opencv-and-javascript-eb3d6ced28a0>

---

## 12. 一句话给 Lead

**没有现成的 4 方向指向模型；最轻最稳的路线是「MediaPipe HandLandmarker（7.46 MiB，Apache-2.0，本机已装）+ 自研 4 类判定（Google 自家 8 类头只有 7.8 KB，社区 3 类 MLP 只有 6.4 KB）」，最大未知数是 2 米/100px 下的检出率与抖动——这必须在本地 dataset 上实测，公开资料只到"palm detector 训练上限约 2 米"。**

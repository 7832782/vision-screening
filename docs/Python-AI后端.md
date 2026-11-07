# Python AI 后端（语音 + 手势）

> 版本 v0.5.0 · 2026-11-07
>
> 前端仍然是单文件 `index.html`；语音识别与手势识别改由本机的 Python 服务完成。
> 这样做的核心原因：浏览器里用 MFCC + DTW 做「四个词」的音素级识别天花板很低，
> 而 SenseVoice / MediaPipe 这类模型在 Python 生态里成熟、离线、CPU 就能跑。

## 1. 为什么换后端

v0.4 的离线语音流程是：使用者先念两遍「上/下/左/右」当模板 → 提取 13 维 MFCC +
13 维差分 → DTW 匹配。它解决了「不联网」，但有三个无法靠调参消除的问题：

1. **单字信息量太少**：四个都是单音节，声母/韵母差异小，DTW 对语速、口音、麦克风很敏感。
2. **必须录模板**：老人第一次使用就要学一套「按住说话」的录制流程，适老化上不友好。
3. **只有四个词**：说「看不清」「看不见」「不知道」无法处理，而这正是测试中最常见的回答。

换成 Python 后端后：

- **识别模型**：sherpa-onnx 自带的 **SenseVoice int8**（阿里 FunASR 开源，中英日韩粤，239 MB）。
- **断句**：sherpa-onnx 的 **Silero VAD**（0.6 MB）。
- **同音字处理**：SenseVoice 会把单字「左」写成「做」、「右」写成「有」；用 `pypinyin`
  转无调拼音后做序列包含匹配即可稳定归一（`zuo → left`、`you → right`）。
- **不再需要录模板**，并且能识别「看不清 / 看不见 / 不知道」→ `unsure`（记为猜错）。

## 2. 架构

```
手机 / 一体机浏览器（index.html）
   │  WebSocket /ws/asr      16 kHz int16 PCM 二进制帧
   │  WebSocket /ws/pose     320×240 JPEG 帧（约 8 fps）
   ▼
FastAPI（server/app.py）── 同源托管 index.html
   ├── AsrEngine（server/asr_engine.py）
   │     ├── Silero VAD：threshold=0.30, min_silence=0.30s, min_speech=0.12s
   │     ├── SenseVoice int8：CPU 4 线程，RTF≈0.032
   │     └── map_text：pypinyin 无调拼音包含匹配
   └── PoseEngine（server/pose_engine.py）
         ├── MediaPipe PoseLandmarker full（33 点，VIDEO 模式）
         └── 肩→腕向量 + 肘角/躯干归一化 + 6 帧多数投票 + 4 帧迟滞
```

数据只在内存中流转：音频帧不落盘，视频帧不落盘，模型推理完立即丢弃。
前端不做任何上传（除了把帧发到你自己电脑上的这个服务）。

## 3. HTTP / WebSocket 接口

| 接口 | 说明 |
|---|---|
| `GET /api/health` | 返回 `{ok, version, asr:{ready,model}, pose:{ready,model}}`，前端据此启动 AI 模式 |
| `WS /ws/asr` | 客户端发 16 kHz、单声道、int16 little-endian PCM 二进制帧；服务端回 `ready` / `speech_start` / `final` |
| `WS /ws/pose` | 客户端发 JPEG 二进制帧；服务端回 `{type:"pose", pose, gesture, confidence, landmarks, debug}` |
| `POST /api/asr/file` | multipart 上传 WAV，返回 `{text, dir, latencyMs}`（调试用） |
| `POST /api/asr/pcm?sample_rate=16000` | raw int16 PCM 识别（调试 / 压测用） |

### `/ws/asr` 事件

```json
{"type":"ready","engine":"sensevoice"}
{"type":"speech_start"}
{"type":"final","text":"向上。","dir":"up","latencyMs":46.2}
```

`dir` 取值：`up` / `down` / `left` / `right` / `unsure` / `null`。

### 方向词映射

| 说法 | 归一 |
|---|---|
| 上 / 向上 / 上面 / 上边 | `up` |
| 下 / 向下 / 下面 / 下边 | `down` |
| 左 / 向左 / 左边 / 往左（含「做」等同音误写） | `left` |
| 右 / 向右 / 右边 / 往右（含「有」「要」等同音误写） | `right` |
| 看不清 / 看不见 / 看不到 / 不知道 / 不清楚 | `unsure` |

## 4. 实测数据（2026-11-07，R5 5600 CPU）

### 语音（SenseVoice int8）

| 测试 | 结果 |
|---|---|
| `_test/` 48 个单字（3 个 SAPI 声音 × 4 档语速） | **48/48 = 100%** |
| `_test2/` 117 个短语（向上/向下/向左/向右/上面/下面/左边/右边/往左/往右/看不清/看不见/不知道） | **117/117 = 100%** |
| 流式 VAD 165 个文件（前后补静音、512 样本块） | **165/165 每文件恰好 1 个 final，方向全部正确** |
| 离线解码延迟 | p50 42 ms / p95 56 ms |
| 流式「说完 → final」总延迟 | p50 376 ms / p95 668 ms |
| 白噪声 20 / 15 / 10 dB | 85.4% / 79.2% / 79.2% |
| 模型加载 | 1.0–1.5 s（warmup 27 ms） |

### 手势（MediaPipe Pose full）

| 测试 | 结果 |
|---|---|
| `server/test_pose.py` | **36/36 ALL PASS** |
| 定向图（mirror=True，被试视角） | up / down / 左（图像左=被试右→`right`）/ 右 4/4 |
| 合成 33 点单测 | 21/21（含自然下垂 none、遮挡、低可见度、屈肘、侧身、半身） |
| 单帧延迟（1024×1536 JPEG 解码+推理） | p50 20–22 ms / p95 21–25 ms |
| 稳定策略 | 6 帧滑窗多数投票 + 连续 4 帧确认，切换方向约 4 帧（≈0.5 s） |

## 5. 运行

```powershell
# 1. 安装依赖（只需一次）
cd "F:\DeepSeek Harness\项目\vision-screening"
python -m pip install -r server/requirements.txt

# 2. 下载模型（约 165 MB，只需一次）
python server/download_models.py

# 3. 启动（会自动把模型复制到 %LOCALAPPDATA%\vision-screening\models）
cd server
python app.py
# 或双击 server/start.bat
```

启动后：

- 本机：<http://127.0.0.1:8000/>
- 同一 Wi-Fi 的手机：`http://<电脑局域网 IP>:8000/`（页面会打印 IP）

> ⚠️ 手机浏览器只在 **HTTPS** 或 `localhost` 下允许麦克风/摄像头。
> 局域网 `http://192.168.x.x` 在手机上会被判定为不安全上下文，麦克风和摄像头会打不开。
> 手机测试请用 HTTPS 隧道（见下一节）。

## 6. 手机访问（HTTPS）

开发/演示用 Cloudflare 快速隧道（无需账号）：

```powershell
tools\cloudflared.exe tunnel --url http://127.0.0.1:8000
```

它会输出 `https://xxxx.trycloudflare.com`，手机打开这个地址即可，语音和手势都能用。
隧道地址每次重启都会变；正式部署建议用固定域名 + 反向代理（Caddy / Nginx）。

也可以继续用 GitHub Pages 的静态页面，但需要把 Python 后端地址填进设置页：

```
https://7832782.github.io/vision-screening/?backend=https://xxxx.trycloudflare.com
```

### 部署形态（重要）

- **手机 / Android 一体机**：只跑浏览器，负责显示视标、采集麦克风/摄像头帧。
- **Python 后端**：跑在同一 Wi-Fi 下的 Windows/Linux 电脑上（诊所电脑、笔记本、迷你主机都行）。
  Android 一体机本身不运行 Python；如果现场没有电脑，只能用浏览器离线语音（v0.4 的 DTW fallback），
  手势不可用。
- 正式部署：后端放固定电脑/服务器 + 固定 HTTPS 域名，前端可以继续用 GitHub Pages 或直接由后端托管。
- 隐私：音频和视频帧只在内存中处理；但如果走公网隧道，帧会经过 Cloudflare 转发，正式医疗场景建议局域网内直连 HTTPS。

## 7. 前端行为

- 打开页面时自动请求 `/api/health`；连接成功则显示「已连接 · 语音 ✅ · 手势 ✅」。
- 设置页可开关：
  - **Python 语音识别（推荐）**：测试时说「向上 / 向下 / 向左 / 向右」，无需模板。
  - **浏览器离线语音（备用）**：没有 Python 后端时，录模板 + DTW，可离线使用。
  - **手势作答（Python 视觉）**：测试时打开摄像头，用手臂指向作答；测试页右下角有画中画预览和骨架。
- 「语音试音」「手势试音」页面用于现场验证识别效果：说到/指到哪个方向，哪个方块亮。
- 语音播报期间（`ttsBusy`）不向识别服务送音频，避免把设备自己的声音当成答案。

## 8. 已知限制

1. **单字「右」在远场/重采样后可能被听成「要」**：已把 `yao` 归一到 `right`；仍建议提示用户说两个字「向右」，比单字稳。
2. **手势没有手指信息**：只看手臂方向；侧身/半身、自然下垂可能误判 `down`，正面双臂可见时会用对称休息姿态过滤。现场建议 1.5–3 米、手臂伸直。
3. **模型吃 CPU**：SenseVoice int8 单路约 4 核、RTF 0.03；一体机若 CPU 太弱（如老 Atom），建议把后端放在旁边一台电脑上。
4. **隐私**：虽然音频/视频只在内存中处理，但明文 HTTP 局域网可能被同网段抓包；正式使用请走 HTTPS。
5. **模型体积**：SenseVoice int8 约 239 MB，Pose 约 9 MB；需要首次下载。

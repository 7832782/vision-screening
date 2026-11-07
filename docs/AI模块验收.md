# AI 模块验收记录（2026-11-08）

> 本文记录「其他 AI 完成的语音/手势模块」的验收数字，以及整合进 `vision-screening` 的冒烟结果。
> 原始实验目录：`F:\DeepSeek Harness\项目\单音识别方案`、`F:\DeepSeek Harness\项目\vision-screening\_gesture_lab`。

## 1. 语音

### 1.1 服务端双模型（Vosk + SenseVoice）

- 实现：`单音识别方案/server/app.py` + `ensemble.py`
- 模型：Vosk small-cn（65 MB，四字硬语法）+ SenseVoice int8（228 MB，拼音兜底）
- 验收：
  - 真人 24 条：**24/24 = 100%**，来源 both 14 / vosk 6 / sensevoice 4
  - TTS 72 条：72/72 = 100%
  - legacy SAPI 48 条：48/48 = 100%
  - 「看不清」18 条：18/18 判为 unsure
  - 推理 p50 **102 ms**，HTTP 端到端 p50 **111 ms / p95 151 ms**
- 局限：需要本机 Python 后端；只有 1 位真人 24 条样本；负样本误接受约 25%。

### 1.2 纯浏览器 Vosk WASM（本次整合版）

- 资源：`voice/vosk.js`（5.5 MB）+ `voice/model.tar.gz`（43.8 MB，Vosk small-cn）
- 验收（原 `demo-wasm?test=1`，本机 Chromium）：
  - 真人 24 条：**20/24 = 83.3%**
  - TTS 12 条：12/12 = 100%
  - 合计：**32/36 = 88.9%**
  - 模型加载 1.2 s（缓存后 0.5–0.7 s），单条识别 130–350 ms
- 为什么选它：v0.8 要保持 GitHub Pages 纯静态，不引入后端；83.3% 是 Vosk 单模型上限，
  与 Python Vosk 的失败样本完全一致。若现场有电脑，可后续接入服务端 100% 模式。

## 2. 手势

- 实现：`_gesture_lab/pwa` + `hand_rules.js` / `gesture_gate.js` / `hand_landmarker.task`
- 验收：
  - PWA 示例图：**4/4**（GPU delegate，单张 19–246 ms）
  - 离线重载：4/4，0 外部请求（Service Worker 缓存 17 个资源）
  - 闸门单测：**7/7 PASS**（静止垂手不触发、每题只答一次、离开可重新武装等）
  - 原报告数据集：排除 40 cm 纯手部特写后，检出后方向准确率 97%（29/29 次给出方向均正确）
  - 2 m 小目标：640×480 / 480×640 8/8；320×240 不够
- 限制：40 cm 纯手特写检不到；静止垂手需要动作闸门；真实老人数据仍需试用。

## 3. 整合进主页面后的冒烟

| 项目 | 结果 |
|---|---|
| 页面加载 | `window.VoiceEngine` / `window.GestureEngine` 均就绪，0 JS 错误 |
| 语音设置开关 | 首次加载约 44 MB，开关变绿，提示更新 |
| 手势设置开关 | 首次加载约 17 MB，开关变绿 |
| 语音试音页 | 上传/注入 `up_00.wav` → 识别「上」，方块亮起（约 253 ms） |
| 手势试音页 | 示例图 `ai_up_far_01` → 当前「上」，置信度 0.92，方块亮起 |
| 测试流程 · 语音 | 测试页说「上」→ `session.trials` 0 → 1 |
| 手势 PWA 验收页 | 4/4 示例图通过（独立进程 8095） |

> 语音/手势只在设置打开时才懒加载；默认关闭，不申请麦克风/摄像头权限。

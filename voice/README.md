# 离线语音作答模块（Vosk WASM）

本目录是原「单音识别方案」中 `demo-wasm` 的整合版资源（原工程已于 2026-11-08 删除，报告归档在
[`../docs/实验归档/语音-REPORT.md`](../docs/实验归档/语音-REPORT.md)）：

- `vosk.js`：vosk-browser 运行时（含 WASM worker）
- `model.tar.gz`：Vosk 中文小模型 `vosk-model-small-cn-0.22`
- `engine.js`：给主页面用的封装（懒加载模型、麦克风 VAD、16 kHz 识别）

调用方式见 `engine.js` 顶部注释。主页面 `index.html` 的「语音作答（离线）」开关会懒加载本模块；
首次约 44 MB，浏览器缓存后再次打开很快。

**实测（原模块报告，2026-11-08）**：真人 24 条 20/24 = 83.3%；TTS 12/12 = 100%；合计 32/36 = 88.9%。
Vosk 模型页面标注 Apache-2.0；生产前建议向 Alpha Cephei 确认模型许可。

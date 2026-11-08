# 离线手势作答模块（MediaPipe Hand Landmarker）

本目录整合自 `_gesture_lab/pwa`：

- `vendor/`：`@mediapipe/tasks-vision@0.10.21`（本地 WASM）
- `models/hand_landmarker.task`：手部 21 点模型（Apache-2.0）
- `hand_rules.js`：21 点 → 上/下/左/右 的规则判定
- `gesture_gate.js`：动作激活 + 保持确认 + 每题一次
- `engine.js`：给主页面用的摄像头封装（镜像、骨架绘制、VIDEO 模式）

主页面「手势作答（离线）」开关会懒加载本模块；模型和 WASM 首次约 17 MB（封装代码约 145 KB 随首屏加载）。
原 PWA 实测：示例图 4/4，离线重载 4/4，闸门单测 7/7。远距 2 m 建议 640×480；近距 40 cm 请用
「脸 + 手都在画面」的构图，纯手部大特写可能检不出手。

**预览不裁切（v0.9.5）**：视频用 `object-fit: contain`，预览框的 `aspect-ratio` 跟随
`video.videoWidth/videoHeight`；骨架用 `Math.min(vrw/vw, vrh/vh)` 缩放 + 居中。
不要改回 `cover` —— 手机竖屏的 480×640 流放进 4:3 横幅会被上下裁掉约 105 px，举起的手会直接出框。

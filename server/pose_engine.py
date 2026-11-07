# -*- coding: utf-8 -*-
"""手势方向识别引擎（MediaPipe Pose Landmarker）。

职责
----
把前端摄像头/画布送来的 JPEG 帧转成「上/下/左/右」手势：
- ``PoseEngine.detect``：解码 JPEG → PoseLandmarker → 方向分类 → 时间稳定。
- ``classify_landmarks``：纯函数，输入 33 个姿态点即可判定方向，方便单元测试。

坐标约定
--------
MediaPipe 的归一化坐标 x/y 都以「画面左上角」为原点、y 向下。
``mirror=True`` 会先把 x 翻转为 ``1-x`` 再判定：前端视频用 CSS ``scaleX(-1)``
做了自拍镜像，翻转后「被试举右手」会落在预览右侧，判为 ``right``。
返回的 ``landmarks`` 始终是**未镜像**的原始坐标，供前端自行决定绘制方式。

时间稳定策略
------------
VIDEO 模式下使用窗口 6 帧、连续确认 4 帧的「多数投票 + 迟滞」：
- 只有同一方向在最近窗口内出现 >= 4 帧、且连续 >= 4 帧，才切换稳定输出；
- 单帧抖动不会改变稳定方向，保持上一稳定值；
- 稳定方向需要连续 4 帧不同的原始结果才会被替换（包括切到 ``none``）。
视频 10~15 FPS 时，手势确认延迟约 0.3~0.4 秒，不会出现左右来回跳。
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

# MediaPipe Pose 的 33 点里的关键索引
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

LANDMARK_COUNT = 33

# 判定阈值（集中放这里，方便调参）
MIN_VISIBILITY = 0.35        # 肩/腕的最低可见度，低于此值认为该手臂不可用
MIN_ELBOW_VISIBILITY = 0.25  # 肘部可见度低于此值时只用「肩腕距离」判断伸展
EXTEND_ELBOW_ANGLE = 140.0   # 肘角大于该值视为手臂伸直
EXTEND_DISTANCE_RATIO = 0.60 # 肩腕距离 / 躯干长度 大于该值视为手臂伸直
MIN_DIRECTION_AXIS = 0.75    # |主方向分量|/肩腕距离，低于该值视为斜向、方向不明
MIN_SCALE_PX = 12.0          # 尺度太小时不可靠

_GESTURES = ("up", "down", "left", "right", "none")


def _empty_debug() -> dict:
    return {"arm": None, "dx": 0.0, "dy": 0.0, "angle": 0.0}


def _no_pose_result() -> dict:
    return {
        "pose": False,
        "gesture": "none",
        "confidence": 0.0,
        "landmarks": None,
        "debug": _empty_debug(),
    }


def _read_point(p) -> tuple[float, float, float]:
    """兼容 [x, y, visibility] 序列、{'x','y','visibility'} 字典与带同名属性的对象。"""
    if isinstance(p, dict):
        return float(p["x"]), float(p["y"]), float(p.get("visibility", 1.0))
    if hasattr(p, "x"):
        return float(p.x), float(p.y), float(getattr(p, "visibility", 1.0))
    values = list(p)
    x, y = float(values[0]), float(values[1])
    vis = float(values[2]) if len(values) > 2 else 1.0
    return x, y, vis


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """向量 ba 与 bc 的夹角（度），b 为顶点。"""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 0.0
    cos = float(np.dot(ba, bc) / denom)
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _arm_candidate(pts: np.ndarray, side: str, scale: float) -> dict | None:
    """评估单条手臂：是否伸直、指向哪个方向。

    ``pts`` 为 [(x_px, y_px, visibility), ...] 33 点。
    返回 None 表示该手臂不可用或没有明显方向。
    """
    if side == "left":
        si, ei, wi = L_SHOULDER, L_ELBOW, L_WRIST
    else:
        si, ei, wi = R_SHOULDER, R_ELBOW, R_WRIST

    sx, sy, sv = pts[si]
    ex, ey, ev = pts[ei]
    wx, wy, wv = pts[wi]

    if sv < MIN_VISIBILITY or wv < MIN_VISIBILITY:
        return None

    dx = wx - sx
    dy = wy - sy
    dist = float(np.hypot(dx, dy))
    if dist < max(MIN_SCALE_PX, 0.15 * scale):
        return None

    ratio = dist / scale
    elbow_angle = _angle_at(pts[si][:2], pts[ei][:2], pts[wi][:2]) if ev >= MIN_ELBOW_VISIBILITY else 180.0
    extended = (elbow_angle >= EXTEND_ELBOW_ANGLE) or (ratio >= EXTEND_DISTANCE_RATIO)
    if not extended:
        return None

    # 画面坐标：0°=向右，90°=向上，180°=向左，-90°=向下
    theta = float(np.degrees(np.arctan2(-dy, dx)))
    if -45.0 <= theta < 45.0:
        gesture = "right"
    elif 45.0 <= theta < 135.0:
        gesture = "up"
    elif theta >= 135.0 or theta < -135.0:
        gesture = "left"
    else:
        gesture = "down"

    axis = max(abs(dx), abs(dy)) / dist
    if axis < MIN_DIRECTION_AXIS:
        return None

    extension_score = max(
        _clamp01((elbow_angle - 120.0) / 50.0),
        _clamp01((ratio - 0.30) / 0.70),
    )
    direction_score = _clamp01((axis - 0.60) / 0.40)
    visibility_score = _clamp01(min(sv, wv))
    confidence = _clamp01(
        0.50 * extension_score + 0.35 * direction_score + 0.15 * visibility_score
    )

    return {
        "gesture": gesture,
        "confidence": round(confidence, 3),
        "arm": side,
        "dx": float(dx),
        "dy": float(dy),
        "angle": theta,
        "ratio": ratio,
        "elbow_angle": elbow_angle,
        "wrist_vis": wv,
        "shoulder_vis": sv,
    }


def _is_rest_pose(down_candidates: list[dict], pts: np.ndarray, scale: float) -> bool:
    """双臂自然下垂：两条手臂都接近垂直、腕部高度接近 → 属于休息姿态，不是「向下指」。"""
    if len(down_candidates) != 2:
        return False
    a, b = down_candidates
    wrists = (pts[L_WRIST], pts[R_WRIST])
    if abs(wrists[0][1] - wrists[1][1]) > 0.22 * scale:
        return False
    # 自然下垂时手腕基本在肩的正下方，横向偏移很小
    for c in (a, b):
        if abs(c["dx"]) > 0.32 * scale:
            return False
    return True


def classify_landmarks(landmarks: Sequence, width: int, height: int,
                       mirror: bool = True) -> tuple[str, float, dict]:
    """纯函数：由 33 个姿态点判定手势方向。

    参数
    ----
    landmarks : 长度 >= 33 的序列，元素为 ``[x, y, visibility]``（归一化坐标）。
    width, height : 原图像素宽高，用于把归一化坐标换算为像素、按宽高比校正。
    mirror : True 时先把 x 翻转为 ``1-x``，与自拍镜像预览保持一致。

    返回
    ----
    ``(gesture, confidence, debug)``：
    - gesture: ``"up"`` / ``"down"`` / ``"left"`` / ``"right"`` / ``"none"``
    - confidence: 0~1
    - debug: ``{"arm": "left"|"right"|None, "dx": float, "dy": float, "angle": float}``
      dx/dy 为肩→腕向量除以躯干长度后的无量纲值；angle 为画面方向角（度，上为正）。
    """
    debug = _empty_debug()
    try:
        if landmarks is None:
            return "none", 0.0, debug
        try:
            count = len(landmarks)
        except TypeError:
            return "none", 0.0, debug
        if count < LANDMARK_COUNT:
            return "none", 0.0, debug
        if width <= 0 or height <= 0:
            return "none", 0.0, debug

        pts = np.zeros((LANDMARK_COUNT, 3), dtype=np.float64)
        for i in range(LANDMARK_COUNT):
            x, y, vis = _read_point(landmarks[i])
            x = 1.0 - x if mirror else x
            pts[i] = (x * float(width), y * float(height), vis)

        # 尺度：优先躯干长度（肩中点→髋中点），半身照看不到髋时退回肩宽 * 2
        shoulder_w = float(np.hypot(
            pts[L_SHOULDER, 0] - pts[R_SHOULDER, 0],
            pts[L_SHOULDER, 1] - pts[R_SHOULDER, 1],
        ))
        shoulder_mid = (pts[L_SHOULDER, :2] + pts[R_SHOULDER, :2]) / 2.0
        hip_mid = (pts[L_HIP, :2] + pts[R_HIP, :2]) / 2.0
        torso = float(np.linalg.norm(shoulder_mid - hip_mid))
        torso_vis = min(pts[L_SHOULDER, 2], pts[R_SHOULDER, 2],
                        pts[L_HIP, 2], pts[R_HIP, 2])
        torso_valid = (
            torso_vis >= 0.30
            and torso > 0.08 * float(height)
            and abs(pts[L_HIP, 1] - pts[R_HIP, 1]) < 0.15 * float(height)
        )
        if torso_valid:
            scale = torso
        elif shoulder_w >= MIN_SCALE_PX:
            scale = shoulder_w * 2.0
        else:
            return "none", 0.0, debug
        if scale < MIN_SCALE_PX:
            return "none", 0.0, debug

        left = _arm_candidate(pts, "left", scale)
        right = _arm_candidate(pts, "right", scale)
        candidates = [c for c in (left, right) if c is not None]

        # 非 down 的方向优先级更高：避免站立时自然下垂的手臂盖过真正举起的手臂
        non_down = [c for c in candidates if c["gesture"] != "down"]
        down = [c for c in candidates if c["gesture"] == "down"]

        chosen = None
        if non_down:
            chosen = max(non_down, key=lambda c: c["confidence"])
        elif down:
            if not _is_rest_pose(down, pts, scale):
                chosen = max(down, key=lambda c: c["confidence"])

        if chosen is None:
            # 即使没有方向，也把最可能是「伸展手臂」的调试信息带出去
            probe = max(candidates, key=lambda c: c["ratio"]) if candidates else None
            if probe is not None:
                debug.update({
                    "arm": probe["arm"],
                    "dx": round(probe["dx"] / scale, 3),
                    "dy": round(probe["dy"] / scale, 3),
                    "angle": round(probe["angle"], 1),
                })
            return "none", 0.0, debug

        debug.update({
            "arm": chosen["arm"],
            "dx": round(chosen["dx"] / scale, 3),
            "dy": round(chosen["dy"] / scale, 3),
            "angle": round(chosen["angle"], 1),
        })
        return chosen["gesture"], float(chosen["confidence"]), debug
    except Exception:
        # 纯函数面对脏数据（NaN、字段缺失、异常类型）也不能让调用方崩掉
        return "none", 0.0, _empty_debug()


class _TemporalStabilizer:
    """多数投票 + 迟滞：连续 ``confirm`` 帧同向才切换稳定输出。"""

    def __init__(self, window: int = 6, confirm: int = 4):
        self._window = window
        self._confirm = confirm
        self._buf: deque[str] = deque(maxlen=window)
        self._stable = "none"
        self._stable_conf = 0.0
        self._pending: str | None = None
        self._pending_count = 0

    def reset(self) -> None:
        self._buf.clear()
        self._stable = "none"
        self._stable_conf = 0.0
        self._pending = None
        self._pending_count = 0

    def update(self, gesture: str, confidence: float) -> tuple[str, float]:
        if gesture not in _GESTURES:
            gesture = "none"
        self._buf.append(gesture)

        if gesture == self._stable:
            self._pending = None
            self._pending_count = 0
            self._stable_conf = float(confidence) if gesture != "none" else 0.0
            return self._stable, self._stable_conf

        if gesture == self._pending:
            self._pending_count += 1
        else:
            self._pending = gesture
            self._pending_count = 1

        if (self._pending_count >= self._confirm
                and self._buf.count(gesture) >= self._confirm):
            self._stable = gesture
            self._stable_conf = float(confidence) if gesture != "none" else 0.0
            self._pending = None
            self._pending_count = 0
        return self._stable, self._stable_conf


def _resolve_model_path(model_path: str | None) -> str:
    """按优先级寻找模型：显式参数 → VS_POSE_MODEL → LOCALAPPDATA → server/models。"""
    candidates: list[str] = []
    if model_path:
        candidates.append(os.path.expanduser(model_path))
    env = os.environ.get("VS_POSE_MODEL")
    if env:
        candidates.append(os.path.expanduser(env))
    local = os.environ.get("LOCALAPPDATA")
    if local:
        candidates.append(os.path.join(local, "vision-screening", "models", "pose_landmarker_full.task"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "models", "pose_landmarker_full.task"))

    for c in candidates:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "找不到 pose_landmarker_full.task，请设置 VS_POSE_MODEL 或把模型放到 "
        "%LOCALAPPDATA%\\vision-screening\\models\\pose_landmarker_full.task"
    )


def _ascii_safe_model_path(path: str) -> str:
    """MediaPipe 的 C++ 层打不开含中文的路径；必要时把模型复制到 LOCALAPPDATA。"""
    try:
        path.encode("ascii")
        return path
    except UnicodeEncodeError:
        pass
    local = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or "."
    target = os.path.join(local, "vision-screening", "models", "pose_landmarker_full.task")
    if os.path.abspath(path) == os.path.abspath(target) and os.path.isfile(target):
        return target
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.isfile(target) or os.path.getmtime(path) > os.path.getmtime(target):
        import shutil
        shutil.copy2(path, target)
    return target


class PoseEngine:
    """MediaPipe Pose 手势识别引擎。

    Parameters
    ----------
    model_path : str | None
        模型路径；None 时按 VS_POSE_MODEL → LOCALAPPDATA → server/models 查找。
    running_mode : str
        ``"VIDEO"``（默认，内部用 detect_for_video + 递增 timestamp，带时间稳定）
        或 ``"IMAGE"``（单张图片，无时间稳定）。

    线程安全：detect 内部加锁，允许 WebSocket 线程池串行调用同一个 engine。
    注意 VIDEO 模式下所有调用共享同一条时间轴，建议每个视频会话一个 engine。
    """

    def __init__(self, model_path: str | None = None, running_mode: str = "VIDEO"):
        mode = str(running_mode).upper()
        if mode not in ("VIDEO", "IMAGE"):
            raise ValueError("running_mode 必须是 'VIDEO' 或 'IMAGE'")
        self._requested_model_path = model_path
        self._running_mode = mode
        self._landmarker: vision.PoseLandmarker | None = None
        self._model_path: str | None = None
        self._lock = threading.RLock()
        self._last_ts = -1
        self._stabilizer = _TemporalStabilizer()
        self.load_time_ms: float | None = None

    # ------------------------------------------------------------------ 生命周期
    def load(self) -> None:
        """加载模型（幂等）。首次调用打印模型路径与耗时。"""
        with self._lock:
            if self._landmarker is not None:
                return
            path = _ascii_safe_model_path(_resolve_model_path(self._requested_model_path))
            options = vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=path),
                running_mode=(vision.RunningMode.VIDEO if self._running_mode == "VIDEO"
                              else vision.RunningMode.IMAGE),
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            t0 = time.perf_counter()
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
            self.load_time_ms = (time.perf_counter() - t0) * 1000.0
            self._model_path = path
            print(f"[pose] model loaded: {path} ({self.load_time_ms:.1f} ms)")

    @property
    def model_path(self) -> str | None:
        """实际加载的模型路径（load 之后才有值）。"""
        return self._model_path

    def close(self) -> None:
        with self._lock:
            if self._landmarker is not None:
                try:
                    self._landmarker.close()
                finally:
                    self._landmarker = None

    def reset(self) -> None:
        """清空时间稳定状态（换人/重新开始视频时调用）。

        注意：只清空投票窗口，不回退 MediaPipe 的 timestamp 时间轴；同一
        PoseLandmarker 的时间戳必须单调递增，需要全新时间轴时请新建 engine。
        """
        with self._lock:
            self._stabilizer.reset()

    # ------------------------------------------------------------------ 推理
    def _next_timestamp(self, timestamp_ms: int | None) -> int:
        if timestamp_ms is not None:
            try:
                ts = int(timestamp_ms)
            except (TypeError, ValueError):
                ts = self._last_ts + 1
        else:
            ts = int(time.monotonic() * 1000.0)
        # MediaPipe 要求严格递增；调用方给了回退的时间戳时顺延一毫秒，避免抛错
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def detect(self, jpg_bytes: bytes,
               timestamp_ms: int | None = None,
               mirror: bool = True) -> dict:
        """识别一帧 JPEG。

        - ``jpg_bytes``: 任意 JPEG 字节（canvas.toBlob 产物即可）。
        - ``timestamp_ms``: VIDEO 模式可选；不传则内部用单调时钟生成递增时间戳。
        - ``mirror``: 默认 True，按自拍镜像语义把 x 翻转后再判方向。

        返回字段见模块文档；没有检测到人、解码失败、模型异常都返回
        ``pose=False, gesture="none"``，不向调用方抛异常。
        """
        with self._lock:
            if self._landmarker is None:
                self.load()
            try:
                if not jpg_bytes:
                    raise ValueError("empty jpeg bytes")
                buf = np.frombuffer(jpg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is None or bgr.size == 0:
                    raise ValueError("cv2.imdecode failed")
                height, width = bgr.shape[:2]
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                if self._running_mode == "VIDEO":
                    ts = self._next_timestamp(timestamp_ms)
                    result = self._landmarker.detect_for_video(mp_image, ts)
                else:
                    result = self._landmarker.detect(mp_image)
            except Exception:
                # 坏帧按「没有人」处理，前端会继续送下一帧
                self._stabilizer.update("none", 0.0)
                return _no_pose_result()

            if not result.pose_landmarks:
                self._stabilizer.update("none", 0.0)
                return _no_pose_result()

            raw = result.pose_landmarks[0]
            landmarks = [
                [round(float(p.x), 6), round(float(p.y), 6), round(float(getattr(p, "visibility", 0.0)), 6)]
                for p in raw[:LANDMARK_COUNT]
            ]
            if len(landmarks) < LANDMARK_COUNT:
                self._stabilizer.update("none", 0.0)
                return _no_pose_result()

            gesture, confidence, debug = classify_landmarks(
                landmarks, width, height, mirror=mirror
            )
            if self._running_mode == "VIDEO":
                gesture, confidence = self._stabilizer.update(gesture, confidence)
            return {
                "pose": True,
                "gesture": gesture,
                "confidence": round(float(confidence), 3),
                "landmarks": landmarks,
                "debug": debug,
            }

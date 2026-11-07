# -*- coding: utf-8 -*-
"""PoseEngine 自测报告（不依赖 pytest）。

运行：
    python server/test_pose.py

覆盖：
1. 4 张定向图：up / down2 / left / left_flip（mirror=True 语义）；
2. 瑜伽双臂平举图、真人半身图鲁棒性；
3. >=12 个合成 33 点单测（上/下/左/右、自然下垂、遮挡、低可见度、侧身、半身等）；
4. VIDEO 模式 10 帧稳定性、up/right 切换帧数、坏帧容错；
5. 单帧 p50/p95 延迟。
"""

from __future__ import annotations

import os
import statistics
import sys
import time
import traceback

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_engine import PoseEngine, classify_landmarks  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_DIR = os.path.join(ROOT, "_pose_test")

_RESULTS: list[tuple[str, str, bool, str]] = []


def record(section: str, name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((section, name, bool(ok), detail))
    flag = "PASS" if ok else "FAIL"
    suffix = f"  | {detail}" if detail else ""
    print(f"  [{flag}] {name}{suffix}")


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def imread(path: str) -> np.ndarray:
    """OpenCV 不能直接读中文路径，必须 imdecode。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法读取图片: {path}")
    return img


def to_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return buf.tobytes()


def pct(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = max(0, min(len(ordered) - 1, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def fmt_vis(lms: list, arm: str | None) -> str:
    if lms is None:
        return "无关键点"
    mapping = {"left": (11, 13, 15), "right": (12, 14, 16)}
    if arm not in mapping:
        return "无可用手臂"
    si, ei, wi = mapping[arm]
    return f"肩={lms[si][2]:.2f} 肘={lms[ei][2]:.2f} 腕={lms[wi][2]:.2f}"


# --------------------------------------------------------------------------- #
# 1. 静态图片
# --------------------------------------------------------------------------- #
def test_static_images(engine: PoseEngine) -> None:
    section("1. 静态定向图（PoseEngine IMAGE 模式，mirror=True）")
    print("语义：mirror=True = 自拍镜像预览，gesture 是「被试自己视角」的左右。")
    print("  left.png 手臂在图像左侧 -> 被试右手 -> right；left_flip.png 反之 -> left。")

    cases = [
        ("up.png", "up", "双臂上举 -> up"),
        ("down2.png", "down", "侧身、手臂明确指向地面 -> down"),
        ("left.png", "right", "图像左侧手臂 -> 被试右手 -> right"),
        ("left_flip.png", "left", "图像右侧手臂 -> 被试左手 -> left"),
    ]
    for filename, expected, note in cases:
        path = os.path.join(POSE_DIR, filename)
        img = imread(path)
        result = engine.detect(to_jpeg(img), mirror=True)
        ok = result["pose"] and result["gesture"] == expected
        record(
            "静态图", f"{filename}（{note}）", ok,
            f"pose={result['pose']} gesture={result['gesture']} conf={result['confidence']:.3f} "
            f"arm={result['debug']['arm']} angle={result['debug']['angle']} "
            f"vis[{fmt_vis(result['landmarks'], result['debug']['arm'])}]",
        )

    # 反向验证 mirror 语义：同一张图 mirror=False 应为画面方向
    raw_pairs = [("left.png", "left"), ("left_flip.png", "right")]
    for filename, expected_raw in raw_pairs:
        img = imread(os.path.join(POSE_DIR, filename))
        result = engine.detect(to_jpeg(img), mirror=False)
        record(
            "镜像语义", f"{filename} mirror=False（原始画面方向）",
            result["gesture"] == expected_raw,
            f"gesture={result['gesture']} 期望={expected_raw}",
        )

    # 前端实际发送尺寸：320px 宽、JPEG 质量 0.62（canvas.toBlob）
    app_expected = {"up.png": "up", "down2.png": "down", "left.png": "right", "left_flip.png": "left"}
    app_ok = 0
    app_details = []
    for filename, expected in app_expected.items():
        img = imread(os.path.join(POSE_DIR, filename))
        h, w = img.shape[:2]
        nh = max(1, round(h * 320 / w))
        small = cv2.resize(img, (320, nh), interpolation=cv2.INTER_AREA)
        result = engine.detect(to_jpeg(small, quality=62), mirror=True)
        good = result["pose"] and result["gesture"] == expected
        app_ok += int(good)
        app_details.append(f"{filename}:{result['gesture']}/{result['confidence']:.2f}")
    record("压缩帧", "320px / JPEG q62（前端 canvas.toBlob 尺寸）四张全部正确",
           app_ok == 4, " ".join(app_details))

    # down.png：指向镜头前下方，2D 投影基本在肩高，不适合作为 down 的真实图测试
    img = imread(os.path.join(POSE_DIR, "down.png"))
    result = engine.detect(to_jpeg(img), mirror=True)
    print(f"  [DIAG] down.png（不适合 down 图测）：gesture={result['gesture']} "
          f"conf={result['confidence']:.3f} arm={result['debug']['arm']} "
          f"dx={result['debug']['dx']} dy={result['debug']['dy']}；"
          f"该结果来自另一条自然下垂手臂，非指向手臂（指向手臂 2D 投影在肩高、肘角约 89°）。")

    # right.png 与 left.png 实际都指向图像左侧，是生成图片的重复样本，不纳入 4 张计数
    img = imread(os.path.join(POSE_DIR, "right.png"))
    result = engine.detect(to_jpeg(img), mirror=True)
    print(f"  [DIAG] right.png（与 left.png 同向，生成异常，不纳入图测）："
          f"gesture={result['gesture']} conf={result['confidence']:.3f}")

    # 瑜伽图：双臂平举，左右都算合理
    section("2. 瑜伽双臂平举图 / 真实半身图")
    img = imread(os.path.join(ROOT, "_probe_pose.jpg"))
    result = engine.detect(to_jpeg(img), mirror=True)
    ok = result["pose"] and result["gesture"] in ("left", "right")
    record(
        "瑜伽图", "_probe_pose.jpg 双臂平举", ok,
        f"pose={result['pose']} gesture={result['gesture']} conf={result['confidence']:.3f} "
        f"arm={result['debug']['arm']}。左右臂对称水平伸展，判定取置信度更高一侧；左右均可接受。",
    )

    # 真人半身图：只要求不崩，报告是否检测到人
    img = imread(os.path.join(ROOT, "_probe_cam.jpg"))
    try:
        result = engine.detect(to_jpeg(img), mirror=True)
        print(f"  [INFO] _probe_cam.jpg：pose={result['pose']} gesture={result['gesture']} "
              f"（半身/器材画面，检测不到完整人体属预期）")
        record("半身图", "_probe_cam.jpg 不崩溃", True,
               f"pose={result['pose']} gesture={result['gesture']}")
    except Exception as exc:  # noqa: BLE001
        record("半身图", "_probe_cam.jpg 不崩溃", False, f"异常 {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# 2. 合成 33 点单测
# --------------------------------------------------------------------------- #
W, H = 640, 480


def base_points() -> list[list[float]]:
    """正面站姿的 33 点：肩宽 0.2、躯干约 0.32 画面高。"""
    pts = [[0.5, 0.5, 0.0] for _ in range(33)]
    pts[11] = [0.60, 0.30, 1.0]  # 左肩（被试左，画面右）
    pts[12] = [0.40, 0.30, 1.0]  # 右肩（被试右，画面左）
    pts[23] = [0.58, 0.62, 1.0]  # 左髋
    pts[24] = [0.42, 0.62, 1.0]  # 右髋
    # 默认另一条手臂自然弯曲贴身，不构成伸展
    pts[13] = [0.64, 0.42, 1.0]
    pts[15] = [0.60, 0.48, 1.0]
    pts[14] = [0.36, 0.42, 1.0]
    pts[16] = [0.40, 0.48, 1.0]
    return pts


def set_arm(pts, side: str, elbow, wrist, vis: float = 1.0,
            elbow_vis: float | None = None) -> None:
    if side == "left":
        si, ei, wi = 11, 13, 15
    else:
        si, ei, wi = 12, 14, 16
    pts[si][2] = vis
    pts[ei] = [elbow[0], elbow[1], vis if elbow_vis is None else elbow_vis]
    pts[wi] = [wrist[0], wrist[1], vis]


def _pose_up_right():
    p = base_points()
    set_arm(p, "right", (0.38, 0.16), (0.35, 0.02))
    return p


def _pose_up_left():
    p = base_points()
    set_arm(p, "left", (0.62, 0.16), (0.65, 0.02))
    return p


def _pose_down_right():
    p = base_points()
    set_arm(p, "right", (0.33, 0.55), (0.26, 0.80))
    return p


def _pose_down_left():
    p = base_points()
    set_arm(p, "left", (0.67, 0.55), (0.74, 0.80))
    return p


def _pose_screen_left():
    p = base_points()
    set_arm(p, "right", (0.225, 0.30), (0.05, 0.30))
    return p


def _pose_screen_right():
    p = base_points()
    set_arm(p, "left", (0.775, 0.30), (0.95, 0.30))
    return p


def _pose_rest_down():
    p = base_points()
    set_arm(p, "right", (0.36, 0.44), (0.34, 0.60))
    set_arm(p, "left", (0.64, 0.44), (0.66, 0.60))
    return p


def _pose_left_occluded():
    p = base_points()
    set_arm(p, "right", (0.38, 0.16), (0.35, 0.02))
    set_arm(p, "left", (0.64, 0.42), (0.60, 0.48), vis=0.05)
    return p


def _pose_all_lowvis():
    p = base_points()
    set_arm(p, "right", (0.38, 0.16), (0.35, 0.02), vis=0.12)
    return p


def _pose_diagonal():
    p = base_points()
    set_arm(p, "right", (0.31, 0.20), (0.244, 0.092))
    return p


def _pose_bent():
    p = base_points()
    set_arm(p, "right", (0.40, 0.42), (0.45, 0.40))
    return p


def _pose_side_up():
    p = base_points()
    p[11] = [0.51, 0.30, 1.0]
    p[12] = [0.49, 0.30, 1.0]
    set_arm(p, "right", (0.49, 0.16), (0.49, 0.02))
    return p


def _pose_half_body_left():
    p = base_points()
    p[23][2] = 0.0
    p[24][2] = 0.0
    set_arm(p, "right", (0.225, 0.30), (0.05, 0.30))
    return p


def _pose_elbow_invisible():
    p = base_points()
    set_arm(p, "right", (0.38, 0.16), (0.35, 0.02), elbow_vis=0.0)
    return p


def _pose_wrist_lowvis():
    p = base_points()
    set_arm(p, "right", (0.38, 0.16), (0.35, 0.02))
    p[16][2] = 0.2
    return p


def _pose_down_one_arm():
    p = base_points()
    set_arm(p, "right", (0.33, 0.55), (0.26, 0.80))
    return p


def test_synthetic() -> None:
    section("3. 合成 33 点单测（classify_landmarks 纯函数）")
    cases = [
        ("右手向上", _pose_up_right, False, "up"),
        ("左手向上", _pose_up_left, False, "up"),
        ("右手向下", _pose_down_right, False, "down"),
        ("左手向下", _pose_down_left, False, "down"),
        ("手臂指向画面左（原始）", _pose_screen_left, False, "left"),
        ("同上，mirror=True", _pose_screen_left, True, "right"),
        ("手臂指向画面右（原始）", _pose_screen_right, False, "right"),
        ("同上，mirror=True", _pose_screen_right, True, "left"),
        ("双臂自然下垂", _pose_rest_down, False, "none"),
        ("单手遮挡（另一手向上）", _pose_left_occluded, False, "up"),
        ("全部低可见度", _pose_all_lowvis, False, "none"),
        ("45° 斜向（方向不明）", _pose_diagonal, False, "none"),
        ("屈肘未伸展", _pose_bent, False, "none"),
        ("侧身（肩几乎重叠）向上", _pose_side_up, False, "up"),
        ("半身照（髋不可见）指向左", _pose_half_body_left, False, "left"),
        ("肘部不可见但手臂伸直", _pose_elbow_invisible, False, "up"),
        ("腕部低可见度", _pose_wrist_lowvis, False, "none"),
        ("单臂向下（另一臂弯曲）", _pose_down_one_arm, False, "down"),
    ]
    passed = 0
    for name, factory, mirror, expected in cases:
        gesture, confidence, debug = classify_landmarks(factory(), W, H, mirror=mirror)
        ok = gesture == expected
        passed += int(ok)
        record("合成", f"{name}（mirror={mirror}）", ok,
               f"gesture={gesture} 期望={expected} conf={confidence:.3f} "
               f"arm={debug['arm']} dx={debug['dx']} dy={debug['dy']} angle={debug['angle']}")
    # 脏数据/越界输入
    for name, bad in [
        ("少于 33 点", [[0.5, 0.5, 1.0]] * 10),
        ("空输入", []),
        ("NaN 坐标", [[float("nan"), 0.5, 1.0] for _ in range(33)]),
    ]:
        try:
            gesture, confidence, debug = classify_landmarks(bad, W, H)
            ok = gesture == "none"
            record("合成", f"脏输入：{name}", ok, f"gesture={gesture} conf={confidence:.3f}")
            passed += int(ok)
        except Exception as exc:  # noqa: BLE001
            record("合成", f"脏输入：{name}", False, f"抛异常 {type(exc).__name__}: {exc}")
    print(f"  合成用例合计：{passed}/{len(cases) + 3} 通过")


# --------------------------------------------------------------------------- #
# 3. VIDEO 稳定性 / 切换
# --------------------------------------------------------------------------- #
def test_video_stability() -> None:
    section("4. VIDEO 模式时间稳定性（窗口 6 帧、连续确认 4 帧）")
    up_bytes = to_jpeg(imread(os.path.join(POSE_DIR, "up.png")))
    right_bytes = to_jpeg(imread(os.path.join(POSE_DIR, "right.png")))

    engine = PoseEngine(running_mode="VIDEO")
    engine.load()
    sequence = []
    first_stable = None
    for i in range(10):
        result = engine.detect(up_bytes, timestamp_ms=i * 33)
        sequence.append(result["gesture"])
        if first_stable is None and result["gesture"] == "up":
            first_stable = i + 1
    record("视频", "同一张 up 重复 10 帧：最后 6 帧稳定为 up",
           first_stable is not None and len(set(sequence[-6:])) == 1 and sequence[-1] == "up",
           f"first_stable=第{first_stable}帧 sequence={sequence}")

    # 单帧抖动：混入一帧 right，输出应保持 up
    jitter = []
    for i, use_right in enumerate([True] + [False] * 5):
        result = engine.detect(right_bytes if use_right else up_bytes,
                               timestamp_ms=(10 + i) * 33)
        jitter.append(result["gesture"])
    record("视频", "单帧抖动不改变稳定方向", all(g == "up" for g in jitter),
           f"sequence={jitter}")

    # 坏帧：不抛异常，pose=False、gesture=none
    bad = engine.detect(b"this is not a jpeg")
    record("视频", "坏 JPEG 不抛异常", not bad["pose"] and bad["gesture"] == "none",
           f"pose={bad['pose']} gesture={bad['gesture']}")
    engine.close()

    # 切换：up -> right，统计需要几帧
    engine2 = PoseEngine(running_mode="VIDEO")
    engine2.load()
    for i in range(6):
        engine2.detect(up_bytes, timestamp_ms=i * 33)
    switch_seq = []
    switch_at = None
    for i in range(10):
        result = engine2.detect(right_bytes, timestamp_ms=(6 + i) * 33)
        switch_seq.append(result["gesture"])
        if switch_at is None and result["gesture"] == "right":
            switch_at = i + 1
    record("视频", "up -> right 切换帧数", switch_at is not None and switch_at <= 8,
           f"第{switch_at}帧切换 sequence={switch_seq}")

    # 再切回 up
    back_seq = []
    back_at = None
    for i in range(10):
        result = engine2.detect(up_bytes, timestamp_ms=(16 + i) * 33)
        back_seq.append(result["gesture"])
        if back_at is None and result["gesture"] == "up":
            back_at = i + 1
    record("视频", "right -> up 切回帧数", back_at is not None and back_at <= 8,
           f"第{back_at}帧切换 sequence={back_seq}")
    engine2.close()


# --------------------------------------------------------------------------- #
# 4. 性能
# --------------------------------------------------------------------------- #
def test_performance() -> None:
    section("5. 性能（VIDEO 模式，JPEG 解码 + 推理 + 稳定）")
    up_bytes = to_jpeg(imread(os.path.join(POSE_DIR, "up.png")))
    engine = PoseEngine(running_mode="VIDEO")
    t0 = time.perf_counter()
    engine.load()
    load_ms = (time.perf_counter() - t0) * 1000.0
    for i in range(5):  # 预热
        engine.detect(up_bytes, timestamp_ms=i * 33)
    times = []
    for i in range(30):
        t0 = time.perf_counter()
        engine.detect(up_bytes, timestamp_ms=(5 + i) * 33)
        times.append((time.perf_counter() - t0) * 1000.0)
    p50, p95 = statistics.median(times), pct(times, 95)
    record("性能", "单帧延迟 p50/p95", p95 < 200.0,
           f"p50={p50:.1f}ms p95={p95:.1f}ms min={min(times):.1f} max={max(times):.1f} "
           f"模型加载={load_ms:.1f}ms（engine 自带 load_time_ms={engine.load_time_ms:.1f}ms）")
    engine.close()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("PoseEngine 测试报告")
    print(f"Python {sys.version.split()[0]} | OpenCV {cv2.__version__} | "
          f"NumPy {np.__version__} | 工作目录 {os.getcwd()}")
    try:
        engine = PoseEngine(running_mode="IMAGE")
        engine.load()
        print(f"模型路径：{engine.model_path}")
        print(f"模型加载耗时：{engine.load_time_ms:.1f} ms")
        test_static_images(engine)
        engine.close()
        test_synthetic()
        test_video_stability()
        test_performance()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1

    section("SUMMARY")
    total = len(_RESULTS)
    passed = sum(1 for _, _, ok, _ in _RESULTS if ok)
    sections: dict[str, list[bool]] = {}
    for sec, _, ok, _ in _RESULTS:
        sections.setdefault(sec, []).append(ok)
    for sec, values in sections.items():
        print(f"  {sec}: {sum(values)}/{len(values)}")
    print(f"  总计: {passed}/{total}")
    if passed == total:
        print("  结果: ALL PASS")
        return 0
    print("  结果: FAIL")
    for sec, name, ok, detail in _RESULTS:
        if not ok:
            print(f"    FAIL [{sec}] {name}: {detail}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

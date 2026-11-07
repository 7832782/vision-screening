#!/usr/bin/env python3
"""vision-screening ASR 引擎独立测试与基准。

用法::

    python server/test_asr.py

不依赖 pytest。退出码：全部硬性检查通过为 0，否则为 1。
覆盖：文本映射、两套语料准确率、VAD 流式分句与延迟、白噪声鲁棒性、
边界场景（静音/极短音/连续两句/flush）、并发安全、性能。
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

_SERVER_DIR = Path(__file__).resolve().parent
_ROOT = _SERVER_DIR.parent
sys.path.insert(0, str(_SERVER_DIR))

from asr_engine import AsrEngine, SAMPLE_RATE, map_text  # noqa: E402

try:  # 保证在 Windows 终端 / 管道中中文都能正确显示
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass

MODEL_DIR = _ROOT / "server" / "models" / "sense-voice"
VAD_MODEL = _ROOT / "server" / "models" / "silero_vad.onnx"
SINGLE_DIR = _ROOT / "_test"
PHRASE_DIR = _ROOT / "_test2"

# _test2 的文件名前缀 -> 期望方向
PHRASE_TRUTH = {
    "up2": "up", "up3": "up",
    "down2": "down", "down3": "down",
    "left2": "left", "left3": "left", "left4": "left",
    "right2": "right", "right3": "right", "right4": "right",
    "unsure1": "unsure", "unsure2": "unsure", "unsure3": "unsure",
}


class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        suffix = f"  ({detail})" if detail else ""
        if ok:
            self.passed += 1
            print(f"  [PASS] {label}{suffix}")
        else:
            self.failed.append(label)
            print(f"  [FAIL] {label}{suffix}")
        return bool(ok)

    def note(self, label: str, detail: str = "") -> None:
        suffix = f"  ({detail})" if detail else ""
        print(f"  [INFO] {label}{suffix}")

    def summary(self) -> int:
        total = self.passed + len(self.failed)
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        print(f"  硬性检查: {self.passed}/{total} 通过")
        if self.failed:
            print("  失败项:")
            for item in self.failed:
                print(f"    - {item}")
            print("  结果: FAIL")
            return 1
        print("  结果: PASS")
        return 0


CHECKS = Checks()


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("-" * 78)


def fmt_stats(values, unit: str = "ms") -> str:
    if not values:
        return "n/a"
    arr = np.asarray(values, dtype=np.float64)
    return (f"p50 {np.percentile(arr, 50):.1f}{unit} / "
            f"p95 {np.percentile(arr, 95):.1f}{unit} / max {arr.max():.1f}{unit}")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio.astype(np.float32)), int(sr)


def speech_end_index(audio: np.ndarray, sr: int) -> int:
    """按 20ms 窗能量找最后一个有声样本（用于计算「说完」时刻）。"""
    if audio.size == 0:
        return 0
    win = max(1, int(0.02 * sr))
    peak = float(np.max(np.abs(audio)))
    threshold = max(0.01, 0.02 * peak)
    last = 0
    for start in range(0, audio.size, win):
        if float(np.max(np.abs(audio[start:start + win]))) > threshold:
            last = min(start + win, audio.size)
    return last


def truth_for(path: Path) -> str:
    key = path.stem.split("_")[0]
    return PHRASE_TRUTH.get(key, key)


def corpus_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.wav"))


# ---------------------------------------------------------------------------
# 1. map_text 纯函数
# ---------------------------------------------------------------------------

MAP_CASES = [
    # 基础方向
    ("上", "up"), ("下", "down"), ("左", "left"), ("右", "right"),
    # 复合方向词
    ("向上", "up"), ("向下", "down"), ("向左", "left"), ("向右", "right"),
    ("上边", "up"), ("下边", "down"), ("左边", "left"), ("右边", "right"),
    ("上面", "up"), ("下面", "down"), ("往左", "left"), ("往右", "right"),
    # SenseVoice 常见同音误写
    ("做", "left"), ("有", "right"),
    # 标点 / 空格清理
    ("向 左。", "left"), ("向右！", "right"), ("上，", "up"),
    # 不确定类
    ("看不清", "unsure"), ("看不见", "unsure"), ("看不到", "unsure"),
    ("不知道", "unsure"), ("不清楚", "unsure"),
    # 无关文本
    ("你好", None), ("今天天气不错", None), ("", None), ("。。。", None),
]


def test_map_text() -> None:
    section("1. 文本映射 map_text（pypinyin 无调拼音）")
    failures = []
    for text, expected in MAP_CASES:
        got = map_text(text)
        if got != expected:
            failures.append(f"{text!r}: 期望 {expected!r}, 得到 {got!r}")
    CHECKS.check(not failures, f"map_text 用例 {len(MAP_CASES)}/{len(MAP_CASES)}",
                 "全部通过" if not failures else "; ".join(failures[:5]))


# ---------------------------------------------------------------------------
# 2. 离线逐文件准确率 + 解码延迟 / RTF
# ---------------------------------------------------------------------------

def test_direct_corpus(name: str, directory: Path) -> dict:
    files = corpus_files(directory)
    ok = 0
    misses = []
    latencies = []
    rtfs = []
    for path in files:
        audio, sr = read_wav(path)
        result = engine.transcribe(audio, sr)
        latencies.append(result["latencyMs"])
        duration_ms = max(1.0, len(audio) / sr * 1000.0)
        rtfs.append(result["latencyMs"] / duration_ms)
        if result["dir"] == truth_for(path):
            ok += 1
        else:
            misses.append((path.name, truth_for(path), result["text"], result["dir"]))
    pct = 100.0 * ok / max(1, len(files))
    print(f"  {name}: {ok}/{len(files)} = {pct:.1f}%")
    print(f"    解码延迟: {fmt_stats(latencies)}")
    print(f"    RTF: p50 {np.percentile(rtfs, 50):.3f} / p95 {np.percentile(rtfs, 95):.3f}")
    for miss in misses[:10]:
        print(f"    MISS {miss}")
    CHECKS.check(ok == len(files), f"{name} 逐文件准确率 100%", f"{ok}/{len(files)}")
    return {"files": len(files), "latencies": latencies, "rtfs": rtfs}


def test_direct() -> dict:
    section("2. 离线识别：两套语料逐文件准确率")
    single = test_direct_corpus("_test 单字", SINGLE_DIR)
    phrase = test_direct_corpus("_test2 短语", PHRASE_DIR)
    all_lat = single["latencies"] + phrase["latencies"]
    all_rtf = single["rtfs"] + phrase["rtfs"]
    print(f"  合计: {single['files'] + phrase['files']} 文件，"
          f"解码延迟 {fmt_stats(all_lat)}，RTF p50 {np.percentile(all_rtf, 50):.3f}")
    return {"latencies": all_lat, "rtfs": all_rtf}


# ---------------------------------------------------------------------------
# 3. 流式 VAD：每文件恰好一个 final + 延迟
# ---------------------------------------------------------------------------

def run_stream_file(session, audio: np.ndarray, sr: int):
    """前后补 0.5s/1s 静音，按 512 样本块喂入，返回 (finals, speech_starts, speech_end_pad)。"""
    lead = int(0.5 * sr)
    trail = int(1.0 * sr)
    padded = np.concatenate([
        np.zeros(lead, dtype=np.float32), audio, np.zeros(trail, dtype=np.float32)
    ])
    speech_end_pad = lead + speech_end_index(audio, sr)
    finals = []
    speech_starts = 0
    chunk = 512
    for start in range(0, padded.size, chunk):
        events = session.feed(padded[start:start + chunk])
        for event in events:
            if event["type"] == "speech_start":
                speech_starts += 1
            elif event["type"] == "final":
                finals.append((min(start + chunk, padded.size), event, False))
    for event in session.flush():
        if event["type"] == "speech_start":
            speech_starts += 1
        elif event["type"] == "final":
            finals.append((padded.size, event, True))
    return finals, speech_starts, speech_end_pad


def test_stream_corpus(name: str, directory: Path, session) -> dict:
    files = corpus_files(directory)
    one_final = 0
    correct = 0
    one_start = 0
    endpoint_lat = []
    decode_lat = []
    total_lat = []
    flush_finals = 0
    misses = []
    for path in files:
        audio, sr = read_wav(path)
        finals, starts, speech_end = run_stream_file(session, audio, sr)
        session.reset()
        final_count = len(finals)
        if final_count == 1:
            one_final += 1
        if starts == 1:
            one_start += 1
        dirs = [event["dir"] for _, event, _ in finals]
        if final_count == 1 and dirs[0] == truth_for(path):
            correct += 1
        else:
            misses.append((path.name, truth_for(path), [(round(pos / sr, 3), e["text"], e["dir"])
                                                         for pos, e, _ in finals], starts))
        for pos, event, from_flush in finals:
            if from_flush:
                flush_finals += 1
            endpoint = (pos - speech_end) / sr * 1000.0
            endpoint_lat.append(endpoint)
            decode_lat.append(event["latencyMs"])
            total_lat.append(max(0.0, endpoint) + event["latencyMs"])
    print(f"  {name}: final 正确 {correct}/{len(files)}，恰好 1 个 final {one_final}/{len(files)}，"
          f"恰好 1 个 speech_start {one_start}/{len(files)}")
    print(f"    说完→final 端点延迟: {fmt_stats(endpoint_lat)}")
    print(f"    final 解码延迟:     {fmt_stats(decode_lat)}")
    print(f"    端点+解码总延迟:    {fmt_stats(total_lat)}")
    if flush_finals:
        print(f"    其中 flush 收尾产生 {flush_finals} 个 final")
    for miss in misses[:10]:
        print(f"    MISS {miss}")
    CHECKS.check(one_final == len(files), f"{name} 每文件恰好 1 个 final",
                 f"{one_final}/{len(files)}")
    CHECKS.check(correct == len(files), f"{name} 流式方向全部正确", f"{correct}/{len(files)}")
    CHECKS.check(one_start == len(files), f"{name} 每文件恰好 1 个 speech_start",
                 f"{one_start}/{len(files)}")
    return {"endpoint": endpoint_lat, "decode": decode_lat, "total": total_lat}


def test_streaming() -> dict:
    section("3. 流式 VAD：每 WAV 前后补 0.5s/1s 静音，512 样本块")
    session = engine.create_session()
    single = test_stream_corpus("_test 单字", SINGLE_DIR, session)
    phrase = test_stream_corpus("_test2 短语", PHRASE_DIR, session)
    combined = {k: single[k] + phrase[k] for k in single}
    print(f"  合计 165 文件: 端点 {fmt_stats(combined['endpoint'])}，"
          f"总延迟 {fmt_stats(combined['total'])}")
    return combined


# ---------------------------------------------------------------------------
# 4. 白噪声鲁棒性
# ---------------------------------------------------------------------------

def add_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 0:
        return audio
    noise_rms = rms / (10.0 ** (snr_db / 20.0))
    noise = rng.standard_normal(audio.size).astype(np.float32) * noise_rms
    return (audio + noise).astype(np.float32)


def test_noise() -> None:
    section("4. 白噪声鲁棒性（48 个单字 × SNR 20/15/10dB）")
    files = corpus_files(SINGLE_DIR)
    for snr in (20.0, 15.0, 10.0):
        rng = np.random.default_rng(20261107 + int(snr))
        ok = 0
        misses = []
        for path in files:
            audio, sr = read_wav(path)
            noisy = add_white_noise(audio, snr, rng)
            result = engine.transcribe(noisy, sr)
            if result["dir"] == truth_for(path):
                ok += 1
            else:
                misses.append((path.name, truth_for(path), result["text"], result["dir"]))
        pct = 100.0 * ok / max(1, len(files))
        print(f"  SNR {snr:.0f}dB: {ok}/{len(files)} = {pct:.1f}%")
        for miss in misses[:6]:
            print(f"    MISS {miss}")
        CHECKS.check(ok >= int(0.7 * len(files)), f"SNR {snr:.0f}dB 准确率 ≥ 70%",
                     f"{ok}/{len(files)} = {pct:.1f}%")


# ---------------------------------------------------------------------------
# 5. 边界场景
# ---------------------------------------------------------------------------

def test_edge_cases() -> None:
    section("5. 边界场景")

    # 5.1 全静音 3s -> 不产生任何事件
    session = engine.create_session()
    events = session.feed(np.zeros(3 * SAMPLE_RATE, dtype=np.float32))
    tail = session.flush()
    CHECKS.check(not events and not tail, "全静音 3s 不产生事件",
                 f"feed={len(events)}, flush={len(tail)}")

    # 5.2 0.1s 短音不崩溃
    audio, sr = read_wav(SINGLE_DIR / "up_Kangkang_5.wav")
    start = max(0, speech_end_index(audio, sr) - int(0.1 * sr))
    short = audio[start:start + int(0.1 * sr)]
    session = engine.create_session()
    try:
        events = session.feed(short)
        tail = session.flush()
        CHECKS.check(True, "0.1s 短音不崩溃", f"events={len(events) + len(tail)}")
    except Exception as exc:  # pragma: no cover
        CHECKS.check(False, "0.1s 短音不崩溃", repr(exc))

    # 5.3 连续两段话，中间 0.8s 静音 -> 两个 final
    def trimmed(name: str, directory: Path) -> np.ndarray:
        wav, rate = read_wav(directory / name)
        return wav[:speech_end_index(wav, rate)]

    first = trimmed("up_Huihui_0.wav", SINGLE_DIR)
    second = trimmed("left_Kangkang_0.wav", SINGLE_DIR)
    stream = np.concatenate([
        np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32),
        first,
        np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32),
        second,
        np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32),
    ])
    session = engine.create_session()
    finals = []
    starts = 0
    for begin in range(0, stream.size, 512):
        for event in session.feed(stream[begin:begin + 512]):
            if event["type"] == "speech_start":
                starts += 1
            else:
                finals.append(event)
    finals.extend(event for event in session.flush() if event["type"] == "final")
    dirs = [event["dir"] for event in finals]
    CHECKS.check(len(finals) == 2 and dirs == ["up", "left"],
                 "连续两段话（0.8s 间隔）产生两个 final", f"dirs={dirs}, starts={starts}")

    # 5.4 未说完时 flush 强制收尾
    right = trimmed("right_Kangkang_0.wav", SINGLE_DIR)
    session = engine.create_session()
    for begin in range(0, right.size, 512):
        session.feed(right[begin:begin + 512])
    flushed = [event for event in session.flush() if event["type"] == "final"]
    CHECKS.check(len(flushed) == 1 and flushed[0]["dir"] == "right",
                 "flush() 可在句尾静音前强制产出 final",
                 f"finals={[(e['text'], e['dir']) for e in flushed]}")

    # 5.5 采样率不是 16k 时线性重采样
    audio, sr = read_wav(SINGLE_DIR / "up_Huihui_0.wav")
    x48 = np.interp(
        np.linspace(0.0, audio.size - 1.0, audio.size * 3),
        np.arange(audio.size, dtype=np.float64), audio.astype(np.float64),
    ).astype(np.float32)
    result = engine.transcribe(x48, 48000)
    CHECKS.check(result["dir"] == "up", "48kHz 输入自动重采样识别", f"text={result['text']!r}")

    # 5.6 同一 engine 多 session 并发
    sample_files = [
        SINGLE_DIR / "up_Huihui_0.wav", SINGLE_DIR / "down_Kangkang_0.wav",
        SINGLE_DIR / "left_Yaoyao_0.wav", SINGLE_DIR / "right_Huihui_0.wav",
        SINGLE_DIR / "up_Kangkang_0.wav", SINGLE_DIR / "down_Huihui_0.wav",
        SINGLE_DIR / "left_Kangkang_0.wav", SINGLE_DIR / "right_Yaoyao_0.wav",
    ]

    def session_worker(paths):
        session = engine.create_session()
        results = []
        for spath in paths:
            wav, _ = read_wav(spath)
            events = session.feed(wav)
            events.extend(session.flush())
            session.reset()
            finals = [event["dir"] for event in events if event["type"] == "final"]
            results.append((spath.name, finals[0] if len(finals) == 1 else finals, truth_for(spath)))
        return results

    def transcribe_worker(paths):
        results = []
        for spath in paths:
            wav, rate = read_wav(spath)
            results.append((spath.name, engine.transcribe(wav, rate)["dir"], truth_for(spath)))
        return results

    groups = [sample_files[i::4] for i in range(4)]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            flat = [item for group in pool.map(session_worker, groups) for item in group]
        wrong = [item for item in flat if item[1] != item[2]]
        CHECKS.check(not wrong, "4 线程 × 独立 session 并发流式识别", f"全部正确 ({len(flat)} 个)")
    except Exception as exc:  # pragma: no cover
        CHECKS.check(False, "4 线程 × 独立 session 并发流式识别", repr(exc))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            flat = [item for group in pool.map(transcribe_worker, groups) for item in group]
        wrong = [item for item in flat if item[1] != item[2]]
        CHECKS.check(not wrong, "4 线程并发 transcribe 调用", f"全部正确 ({len(flat)} 个)")
    except Exception as exc:  # pragma: no cover
        CHECKS.check(False, "4 线程并发 transcribe 调用", repr(exc))


# ---------------------------------------------------------------------------
# 6. 性能汇总
# ---------------------------------------------------------------------------

def print_performance(direct: dict, stream: dict) -> None:
    section("6. 性能汇总")
    print(f"  模型加载: {engine.load_seconds * 1000:.0f}ms（warmup "
          f"{engine.warmup_seconds * 1000:.0f}ms）")
    print(f"  离线解码 RTF: p50 {np.percentile(direct['rtfs'], 50):.3f} / "
          f"p95 {np.percentile(direct['rtfs'], 95):.3f}")
    print(f"  离线解码延迟: {fmt_stats(direct['latencies'])}")
    print(f"  流式 final 解码延迟: {fmt_stats(stream['decode'])}")
    print(f"  流式说完→final 总延迟: {fmt_stats(stream['total'])}")
    CHECKS.check(engine.load_seconds > 0 and np.median(direct["rtfs"]) < 1.0,
                 "性能基线（加载成功且 RTF p50 < 1.0）",
                 f"load={engine.load_seconds:.2f}s, RTF p50={np.median(direct['rtfs']):.3f}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    global engine

    print("=" * 78)
    print("vision-screening ASR 引擎测试")
    print(f"Python: {sys.version.split()[0]} | 工作目录: {_ROOT}")
    print(f"模型:   {MODEL_DIR}")
    print(f"VAD:    {VAD_MODEL}")
    print("=" * 78)

    for required in (MODEL_DIR / "model.int8.onnx", MODEL_DIR / "tokens.txt", VAD_MODEL):
        if not required.is_file():
            print(f"[FATAL] 缺少文件: {required}")
            return 2

    section("0. 模型加载")
    load_start = time.perf_counter()
    engine = AsrEngine(str(MODEL_DIR), str(VAD_MODEL), num_threads=4)
    engine.load()
    total_load = time.perf_counter() - load_start
    CHECKS.check(engine.is_loaded, "SenseVoice + Silero VAD 加载成功",
                 f"总耗时 {total_load:.2f}s")

    test_map_text()
    direct = test_direct()
    stream = test_streaming()
    test_noise()
    test_edge_cases()
    print_performance(direct, stream)

    return CHECKS.summary()


if __name__ == "__main__":
    sys.exit(main())

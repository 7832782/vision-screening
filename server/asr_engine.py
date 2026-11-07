"""生产级语音识别引擎：SenseVoice + Silero VAD + 方向词映射。

对外接口（server/app.py 按此导入）::

    class AsrEngine:
        def __init__(self, model_dir: str, vad_model: str, num_threads: int = 4)
        def load(self) -> None
        def transcribe(self, samples, sample_rate=16000) -> dict
        def create_session(self) -> "AsrSession"

    class AsrSession:
        def feed(self, samples) -> list[dict]
        def flush(self) -> list[dict]
        def reset(self) -> None

    def map_text(text: str) -> str | None

设计要点：
- SenseVoice 为离线识别器，多线程安全由本模块内部的解码锁保证；每个
  AsrSession 持有独立的 Silero VAD 实例，互不共享状态。
- 方向判定走 pypinyin 无调拼音的序列包含匹配，规避 SenseVoice 对单字
  「左→做」「右→有」的同音误写；不确定短语优先于方向判定。
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import List, Optional, Sequence

import numpy as np
import sherpa_onnx
from pypinyin import lazy_pinyin

SAMPLE_RATE = 16000
_MODEL_FILENAME = "model.int8.onnx"
_TOKENS_FILENAME = "tokens.txt"

# VAD 参数（任务约定范围）：0.30s 静音收句、0.12s 最短语音、8s 最长语音。
# threshold 取 0.30（低于 Silero 默认 0.5）：SAPI 合成的最快语速单字能量低、
# 时长仅 0.2s 左右，0.5 会漏检；0.30 在实测底噪 RMS≤0.02 时仍不误触发。
VAD_THRESHOLD = 0.30
VAD_MIN_SILENCE = 0.30
VAD_MIN_SPEECH = 0.12
VAD_MAX_SPEECH = 8.0
VAD_BUFFER_SECONDS = 30

# 解码每个 VAD 片段前，在片段首尾各补 0.5s 静音。SenseVoice 对片段边界和
# 前导静音较敏感（直接解 0.6s 的「右」会得到「要」），补足上下文后单字
# 48/48、短语 117/117；0.6s 反而会让最快语速的单字退化，因此取 0.5s。
SEGMENT_PAD_SECONDS = 0.5

_CLEAN_RE = re.compile(r"[\s\W_]+", re.UNICODE)

# 不确定/看不清类短语（无调拼音），必须先于方向词匹配。
_UNSURE_PHRASES = (
    ("kan", "bu", "qing"),   # 看不清
    ("kan", "bu", "jian"),   # 看不见
    ("kan", "bu", "dao"),    # 看不到
    ("bu", "zhi", "dao"),    # 不知道
    ("bu", "qing", "chu"),   # 不清楚
)

# 方向词（无调拼音）。单字「上/下/左/右」已覆盖向上、上边、上面、往左等复合词，
# 按固定顺序匹配，避免同音字带来的歧义。
_DIRECTION_PHRASES = (
    ("up", ("shang",)),    # 上 / 向上 / 上边 / 上面
    ("down", ("xia",)),    # 下 / 向下 / 下边 / 下面
    ("left", ("zuo",)),    # 左 / 向左 / 左边 / 往左（含「做」等同音误写）
    ("right", ("you",)),   # 右 / 向右 / 右边 / 往右（含「有」等同音误写）
    ("right", ("yao",)),   # 「右」在远场/重采样后常被听成「要」，归一到右
)


# ---------------------------------------------------------------------------
# 文本映射
# ---------------------------------------------------------------------------

def _contains_sequence(seq: Sequence[str], sub: Sequence[str]) -> bool:
    """判断 sub 是否为 seq 的连续子序列。"""
    n = len(sub)
    if n == 0 or n > len(seq):
        return False
    return any(tuple(seq[i:i + n]) == tuple(sub) for i in range(len(seq) - n + 1))


def map_text(text: str) -> Optional[str]:
    """把识别文本映射为方向。

    返回 ``"up"`` / ``"down"`` / ``"left"`` / ``"right"`` / ``"unsure"``，
    无法判定时返回 ``None``。
    """
    if not text:
        return None
    cleaned = _CLEAN_RE.sub("", str(text))
    if not cleaned:
        return None
    syllables = [s.lower() for s in lazy_pinyin(cleaned) if s.isalpha()]
    if not syllables:
        return None

    for phrase in _UNSURE_PHRASES:
        if _contains_sequence(syllables, phrase):
            return "unsure"
    for direction, phrase in _DIRECTION_PHRASES:
        if _contains_sequence(syllables, phrase):
            return direction
    return None


# ---------------------------------------------------------------------------
# 音频工具
# ---------------------------------------------------------------------------

def _to_float32_mono(samples) -> np.ndarray:
    """把任意常见 PCM 输入转成 float32 单声道 [-1, 1]。"""
    arr = np.asarray(samples)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if arr.ndim == 2:
        # 约定列优先为 (n, channels)；对 (channels, n) 形状的短数组也兼容。
        if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
            arr = arr.mean(axis=0)
        else:
            arr = arr.mean(axis=1)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype == np.int32:
        arr = arr.astype(np.float32) / 2147483648.0
    elif arr.dtype == np.uint8:
        arr = (arr.astype(np.float32) - 128.0) / 128.0
    else:
        arr = arr.astype(np.float32, copy=False)
    return np.ascontiguousarray(arr.reshape(-1))


def _resample_linear(samples: np.ndarray, sample_rate: int,
                     target_rate: int = SAMPLE_RATE) -> np.ndarray:
    """numpy 线性插值重采样（测试/前端偶发 44.1k/48k 输入时兜底）。"""
    if sample_rate == target_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate: {sample_rate}")
    duration = samples.size / float(sample_rate)
    out_len = max(1, int(round(duration * target_rate)))
    src_idx = np.arange(samples.size, dtype=np.float64)
    dst_idx = np.linspace(0.0, samples.size - 1.0, out_len, dtype=np.float64)
    out = np.interp(dst_idx, src_idx, samples.astype(np.float64))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 引擎
# ---------------------------------------------------------------------------

class AsrEngine:
    """SenseVoice 离线识别引擎；可被多个 AsrSession 共享。"""

    def __init__(self, model_dir: str, vad_model: str, num_threads: int = 4):
        self.model_dir = os.path.abspath(os.fspath(model_dir))
        self.model_path = os.path.join(self.model_dir, _MODEL_FILENAME)
        self.tokens_path = os.path.join(self.model_dir, _TOKENS_FILENAME)
        self.vad_model = os.path.abspath(os.fspath(vad_model))
        self.num_threads = int(num_threads)

        self._recognizer = None
        self._load_lock = threading.Lock()
        self._decode_lock = threading.Lock()
        self._loaded = False
        self.load_seconds = 0.0
        self.warmup_seconds = 0.0

    # -- 生命周期 ---------------------------------------------------------

    def load(self) -> None:
        """加载模型并做一次 warmup；重复调用无副作用，可并发调用。"""
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            for path, label in (
                (self.model_path, "SenseVoice model"),
                (self.tokens_path, "SenseVoice tokens"),
                (self.vad_model, "Silero VAD model"),
            ):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"{label} not found: {path}")

            t0 = time.perf_counter()
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self.model_path,
                tokens=self.tokens_path,
                num_threads=self.num_threads,
                use_itn=True,
                language="zh",
                debug=False,
            )
            self._recognizer = recognizer
            self.load_seconds = time.perf_counter() - t0
            self._loaded = True
            print(
                f"[asr] SenseVoice 已加载: {self.model_path} "
                f"(tokens={self.tokens_path}, threads={self.num_threads}) "
                f"耗时 {self.load_seconds:.3f}s"
            )
            print(f"[asr] Silero VAD: {self.vad_model}")
        self.warmup()

    def warmup(self) -> None:
        """用 0.5s 静音预热首个 decode，避免首句延迟过高。"""
        if not self._loaded:
            self.load()
            return
        t0 = time.perf_counter()
        self._decode(np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32))
        self.warmup_seconds = time.perf_counter() - t0
        print(f"[asr] warmup 完成，耗时 {self.warmup_seconds:.3f}s")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- 识别 -------------------------------------------------------------

    def _decode(self, samples: np.ndarray) -> tuple[str, float]:
        """内部解码：返回 (文本, 解码毫秒)。"""
        if not self._loaded:
            self.load()
        with self._decode_lock:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            t0 = time.perf_counter()
            self._recognizer.decode_stream(stream)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        text = stream.result.text or ""
        return text.strip(), latency_ms

    def transcribe(self, samples, sample_rate: int = 16000) -> dict:
        """识别一整段音频，返回 {"text", "dir", "latencyMs"}。"""
        audio = _to_float32_mono(samples)
        if sample_rate != SAMPLE_RATE:
            audio = _resample_linear(audio, int(sample_rate))
        t0 = time.perf_counter()
        text, _ = self._decode(audio)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "text": text,
            "dir": map_text(text),
            "latencyMs": round(latency_ms, 2),
        }

    def create_session(self) -> "AsrSession":
        """创建流式会话；每个会话独立维护 VAD 状态。"""
        return AsrSession(self)


# ---------------------------------------------------------------------------
# 流式会话
# ---------------------------------------------------------------------------

class AsrSession:
    """基于 Silero VAD 的流式会话。

    feed() 接受任意长度 16kHz float32 单声道音频（内部按 512 样本窗口切片），
    返回本次新增的事件列表：
      - {"type": "speech_start"}
      - {"type": "final", "text": ..., "dir": ..., "latencyMs": ...}
    """

    CHUNK_SIZE = 512

    def __init__(self, engine: AsrEngine):
        self.engine = engine
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = engine.vad_model
        config.silero_vad.threshold = VAD_THRESHOLD
        config.silero_vad.min_silence_duration = VAD_MIN_SILENCE
        config.silero_vad.min_speech_duration = VAD_MIN_SPEECH
        config.silero_vad.max_speech_duration = VAD_MAX_SPEECH
        config.sample_rate = SAMPLE_RATE
        config.num_threads = 1
        self._vad = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=VAD_BUFFER_SECONDS
        )
        self._lock = threading.RLock()
        self._speech_active = False

    # -- 内部 -------------------------------------------------------------

    def _drain_finals(self) -> List[dict]:
        events: List[dict] = []
        pad = np.zeros(int(SEGMENT_PAD_SECONDS * SAMPLE_RATE), dtype=np.float32)
        while not self._vad.empty():
            segment = self._vad.front
            samples = np.array(segment.samples, dtype=np.float32)
            self._vad.pop()
            if samples.size == 0:
                continue
            samples = np.concatenate([pad, samples, pad])
            text, decode_ms = self.engine._decode(samples)
            events.append({
                "type": "final",
                "text": text,
                "dir": map_text(text),
                "latencyMs": round(decode_ms, 2),
            })
        return events

    # -- 公开接口 ---------------------------------------------------------

    def feed(self, samples) -> List[dict]:
        """喂入任意长度音频，返回本次产生的事件。"""
        audio = _to_float32_mono(samples)
        events: List[dict] = []
        with self._lock:
            for start in range(0, audio.size, self.CHUNK_SIZE):
                chunk = audio[start:start + self.CHUNK_SIZE]
                was_speech = self._vad.is_speech_detected()
                self._vad.accept_waveform(chunk)
                is_speech = self._vad.is_speech_detected()
                if is_speech and not was_speech:
                    events.append({"type": "speech_start"})
                events.extend(self._drain_finals())
                self._speech_active = bool(is_speech)
        return events

    def flush(self) -> List[dict]:
        """结束当前音频流：强制收尾未完成的语音段并返回 final 事件。"""
        events: List[dict] = []
        with self._lock:
            self._vad.flush()
            events.extend(self._drain_finals())
            self._speech_active = False
            self._vad.reset()
        return events

    def reset(self) -> None:
        """清空会话状态，准备下一段独立音频。"""
        with self._lock:
            self._vad.reset()
            self._speech_active = False

    @property
    def in_speech(self) -> bool:
        """当前 VAD 是否处于语音段中（只读，供上层 UI 指示）。"""
        return bool(self._speech_active)


__all__ = ["AsrEngine", "AsrSession", "map_text", "SAMPLE_RATE"]

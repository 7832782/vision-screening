"""模型路径管理。

Windows 上 MediaPipe 的 C++ 层无法打开包含中文的路径，因此所有模型文件在
首次启动时会被复制/链接到一个纯 ASCII 的缓存目录（默认
``%LOCALAPPDATA%\\vision-screening\\models``）。sherpa-onnx 本身支持中文路径，
但统一走缓存目录可以让部署更省心。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SERVER_DIR.parent
MODEL_DIR = SERVER_DIR / "models"

SENSEVOICE_FILES = ("model.int8.onnx", "tokens.txt")
SENSEVOICE_DIR_NAME = "sense-voice"
VAD_FILE = "silero_vad.onnx"
POSE_FILE = "pose_landmarker_full.task"

MIN_SIZES = {
    "model.int8.onnx": 200 * 1024 * 1024,
    "tokens.txt": 100 * 1024,
    "silero_vad.onnx": 500 * 1024,
    "pose_landmarker_full.task": 8 * 1024 * 1024,
}


def cache_dir() -> Path:
    override = os.environ.get("VS_MODEL_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "vision-screening" / "models"
    return Path.home() / ".cache" / "vision-screening" / "models"


def _looks_complete(path: Path, name: str) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= MIN_SIZES.get(name, 1)
    except OSError:
        return False


def _ensure_file(src: Path, dst: Path, name: str) -> Path:
    if _looks_complete(dst, name):
        return dst
    if not _looks_complete(src, name):
        raise FileNotFoundError(
            f"模型文件缺失或损坏：{src}\n"
            f"请先运行 python server/download_models.py 下载模型。"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    return dst


def ensure_models(verbose: bool = True) -> dict:
    """确保模型在 ASCII 缓存目录中可用，返回各模型路径。"""
    cache = cache_dir()
    sv_src = MODEL_DIR / SENSEVOICE_DIR_NAME
    sv_dst = cache / SENSEVOICE_DIR_NAME

    paths: dict[str, str] = {}
    for name in SENSEVOICE_FILES:
        p = _ensure_file(sv_src / name, sv_dst / name, name)
        paths[name] = str(p)

    paths["vad"] = str(_ensure_file(MODEL_DIR / VAD_FILE, cache / VAD_FILE, VAD_FILE))
    paths["pose"] = str(_ensure_file(MODEL_DIR / POSE_FILE, cache / POSE_FILE, POSE_FILE))
    paths["sensevoice_dir"] = str(sv_dst)
    paths["cache_dir"] = str(cache)

    if verbose:
        print(f"[models] 缓存目录：{cache}")
        print(f"[models] SenseVoice：{paths['model.int8.onnx']}")
        print(f"[models] VAD：{paths['vad']}")
        print(f"[models] Pose：{paths['pose']}")
    return paths


if __name__ == "__main__":
    import json

    print(json.dumps(ensure_models(), ensure_ascii=False, indent=2))

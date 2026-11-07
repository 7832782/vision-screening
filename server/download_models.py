"""下载运行所需的模型文件。

用法：
    python server/download_models.py

默认从 hf-mirror / GitHub 下载到 server/models/，首次启动服务时会自动复制到
ASCII 缓存目录。若已有文件则跳过。
"""

from __future__ import annotations

import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent
MODEL_DIR = SERVER_DIR / "models"
DL_DIR = MODEL_DIR / "_dl"

SENSEVOICE_ARCHIVE = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2"
MIRRORS = {
    SENSEVOICE_ARCHIVE: [
        f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{SENSEVOICE_ARCHIVE}",
        f"https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/{SENSEVOICE_ARCHIVE}",
    ],
    "silero_vad.onnx": [
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    ],
    "pose_landmarker_full.task": [
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ],
}


def _download(url: str, dst: Path) -> None:
    print(f"  下载 {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        t0 = time.time()
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                pct = done * 100 / total
                speed = done / 1024 / 1024 / max(0.001, time.time() - t0)
                print(f"\r  {pct:5.1f}%  {done/1024/1024:7.1f}/{total/1024/1024:.1f} MB  {speed:.2f} MB/s", end="")
        print()
    os.replace(tmp, dst)


def _get(name: str, dst: Path, min_size: int) -> None:
    if dst.is_file() and dst.stat().st_size >= min_size:
        print(f"[skip] {dst.name} 已存在（{dst.stat().st_size/1024/1024:.1f} MB）")
        return
    last_err = None
    for url in MIRRORS[name]:
        try:
            _download(url, dst)
            if dst.stat().st_size >= min_size:
                return
            last_err = RuntimeError("下载文件过小")
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  失败：{e}")
    raise RuntimeError(f"无法下载 {name}: {last_err}")


def _extract_sensevoice() -> None:
    sv_dir = MODEL_DIR / "sense-voice"
    if (sv_dir / "model.int8.onnx").is_file() and (sv_dir / "tokens.txt").is_file():
        print("[skip] SenseVoice 已解压")
        return
    archive = DL_DIR / SENSEVOICE_ARCHIVE
    print("[extract] SenseVoice ...")
    with tarfile.open(archive, "r:bz2") as tf:
        prefix = tf.getnames()[0].split("/")[0] + "/"
        sv_dir.mkdir(parents=True, exist_ok=True)
        for m in tf.getmembers():
            if not m.isfile():
                continue
            rel = m.name[len(prefix):] if m.name.startswith(prefix) else m.name
            target = sv_dir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(target, "wb") as out:
                out.write(src.read())


def main() -> int:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DL_DIR.mkdir(parents=True, exist_ok=True)

    archive = DL_DIR / SENSEVOICE_ARCHIVE
    _get(SENSEVOICE_ARCHIVE, archive, 100 * 1024 * 1024)
    _extract_sensevoice()
    _get("silero_vad.onnx", MODEL_DIR / "silero_vad.onnx", 500 * 1024)
    _get("pose_landmarker_full.task", MODEL_DIR / "pose_landmarker_full.task", 8 * 1024 * 1024)

    print("\n全部模型就绪：")
    for p in sorted(MODEL_DIR.rglob("*")):
        if p.is_file() and "_dl" not in p.parts:
            print(f"  {p.relative_to(MODEL_DIR)}  {p.stat().st_size/1024/1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

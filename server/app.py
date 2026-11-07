"""视力筛查 · Python AI 后端。

启动：
    python server/app.py
或：
    uvicorn server.app:app --host 0.0.0.0 --port 8000

职责：
- 托管前端静态文件（/、/manifest.webmanifest、图标）
- /api/health        健康检查（前端据此判断是否启用 AI 模式）
- /ws/asr            实时语音识别（16kHz int16 单声道 PCM 二进制帧）
- /ws/pose           实时手势识别（JPEG 帧二进制）
- /api/asr/file      上传 WAV 识别（调试用）
- /api/asr/pcm       一段 raw int16 PCM 识别（调试用）
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import socket
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:  # python server/app.py
    from model_paths import PROJECT_DIR, ensure_models
except ImportError:  # uvicorn server.app:app
    from .model_paths import PROJECT_DIR, ensure_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vs.server")

VERSION = "0.5.0"

STATIC_FILES = {
    "index.html",
    "manifest.webmanifest",
    "icon-192.png",
    "icon-512.png",
    "apple-touch-icon.png",
    "sw.js",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.started_at = time.time()
    app.state.asr = None
    app.state.pose_model = None
    app.state.models = None

    try:
        paths = ensure_models()
        app.state.models = paths
    except Exception as e:  # noqa: BLE001
        log.error("模型准备失败：%s", e)
        paths = None

    if paths:
        # ASR 加载较慢（约 1s），放线程里避免阻塞事件循环
        try:
            try:
                from .asr_engine import AsrEngine
            except ImportError:
                from asr_engine import AsrEngine

            engine = AsrEngine(paths["sensevoice_dir"], paths["vad"], num_threads=int(os.environ.get("VS_ASR_THREADS", "4")))
            await asyncio.to_thread(engine.load)
            app.state.asr = engine
            log.info("ASR 引擎就绪")
        except Exception as e:  # noqa: BLE001
            log.exception("ASR 引擎加载失败：%s", e)

        try:
            try:
                from .pose_engine import PoseEngine
            except ImportError:
                from pose_engine import PoseEngine

            pose = PoseEngine(paths["pose"], running_mode="IMAGE")
            await asyncio.to_thread(pose.load)
            app.state.pose_model = pose
            log.info("Pose 引擎就绪")
        except Exception as e:  # noqa: BLE001
            log.exception("Pose 引擎加载失败：%s", e)

    log.info("服务已启动，版本 %s", VERSION)
    yield
    log.info("服务停止")


app = FastAPI(title="视力筛查 AI 后端", version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 健康检查 / 基础信息
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health():
    asr = getattr(app.state, "asr", None)
    pose = getattr(app.state, "pose_model", None)
    return {
        "ok": True,
        "version": VERSION,
        "uptimeSec": round(time.time() - getattr(app.state, "started_at", time.time()), 1),
        "asr": {
            "ready": asr is not None,
            "model": "sensevoice-int8-2024-07-17" if asr is not None else None,
            "sampleRate": 16000,
        },
        "pose": {
            "ready": pose is not None,
            "model": "pose_landmarker_full" if pose is not None else None,
        },
    }


@app.get("/api/ping")
async def ping():
    return {"pong": True, "t": time.time()}


# ---------------------------------------------------------------------------
# 实时语音识别
# ---------------------------------------------------------------------------
@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    engine = getattr(app.state, "asr", None)
    if engine is None:
        await ws.send_json({"type": "error", "message": "语音模型未加载，请检查 server/models"})
        await ws.close()
        return

    session = engine.create_session()
    await ws.send_json({"type": "ready", "engine": "sensevoice"})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw:
                if len(raw) % 2:
                    raw = raw[:-1]
                pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                events = await asyncio.to_thread(session.feed, pcm)
                for ev in events:
                    await ws.send_json(ev)
                continue
            text = msg.get("text")
            if text:
                try:
                    cmd = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "flush":
                    events = await asyncio.to_thread(session.flush)
                    for ev in events:
                        await ws.send_json(ev)
                elif cmd.get("type") == "reset":
                    session.reset()
                    await ws.send_json({"type": "reset_ok"})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("ASR WebSocket 异常：%s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


@app.post("/api/asr/file")
async def asr_file(file: UploadFile = File(...)):
    engine = getattr(app.state, "asr", None)
    if engine is None:
        raise HTTPException(503, "语音模型未加载")
    data = await file.read()
    try:
        import soundfile as sf

        samples, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"无法解码音频：{e}") from e
    result = await asyncio.to_thread(engine.transcribe, samples, int(sr))
    return {"ok": True, **result}


@app.post("/api/asr/pcm")
async def asr_pcm(request: Request, sample_rate: int = 16000):
    engine = getattr(app.state, "asr", None)
    if engine is None:
        raise HTTPException(503, "语音模型未加载")
    body = await request.body()
    if len(body) % 2:
        body = body[:-1]
    samples = np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0
    result = await asyncio.to_thread(engine.transcribe, samples, sample_rate)
    return {"ok": True, **result}


# ---------------------------------------------------------------------------
# 实时手势识别
# ---------------------------------------------------------------------------
@app.websocket("/ws/pose")
async def ws_pose(ws: WebSocket):
    await ws.accept()
    if getattr(app.state, "pose_model", None) is None:
        await ws.send_json({"type": "error", "message": "Pose 模型未加载"})
        await ws.close()
        return

    try:
        from .pose_engine import PoseEngine
    except ImportError:
        from pose_engine import PoseEngine

    engine = PoseEngine(app.state.models["pose"], running_mode="VIDEO")
    await asyncio.to_thread(engine.load)
    await ws.send_json({"type": "ready", "engine": "pose_landmarker_full"})

    last_ts = 0
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            data = msg.get("bytes")
            if not data:
                continue
            ts = int(time.time() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = await asyncio.to_thread(engine.detect, data, ts, True)
            await ws.send_json({"type": "pose", **result})
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        log.exception("Pose WebSocket 异常：%s", e)
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            engine.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 静态文件（只暴露白名单，避免把 server/models 暴露出去）
# ---------------------------------------------------------------------------
@app.get("/")
async def index():
    return FileResponse(PROJECT_DIR / "index.html", media_type="text/html; charset=utf-8")


@app.get("/{name}")
async def static_file(name: str):
    if name not in STATIC_FILES:
        raise HTTPException(404, "Not Found")
    p = PROJECT_DIR / name
    if not p.is_file():
        raise HTTPException(404, "Not Found")
    media = {
        ".html": "text/html; charset=utf-8",
        ".webmanifest": "application/manifest+json",
        ".png": "image/png",
        ".js": "application/javascript",
    }.get(p.suffix, "application/octet-stream")
    return FileResponse(p, media_type=media)


# ---------------------------------------------------------------------------
# 启动入口
# ---------------------------------------------------------------------------
def _lan_ips() -> list[str]:
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def main() -> None:
    import uvicorn

    host = os.environ.get("VS_HOST", "0.0.0.0")
    port = int(os.environ.get("VS_PORT", "8000"))
    print("=" * 64)
    print(f"  视力筛查 AI 后端 v{VERSION}")
    print(f"  本机访问：http://127.0.0.1:{port}/")
    for ip in _lan_ips():
        print(f"  局域网：  http://{ip}:{port}/   （手机需与电脑同一 Wi-Fi）")
    print("=" * 64)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

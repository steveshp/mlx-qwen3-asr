"""FastAPI ASR server with browser microphone transcription UI."""

import asyncio
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

if TYPE_CHECKING:
    from mlx_qwen3_asr import Session

MODEL_PATH = os.environ.get(
    "ASR_MODEL_PATH",
    os.path.expanduser("~/Downloads/Qwen3-ASR-1.7B-8bit"),
)
STATIC_DIR = Path(__file__).parent / "webapp"

session: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    from mlx_qwen3_asr import Session

    print(f"Loading model from {MODEL_PATH} ...")
    start = time.time()
    session = Session(model=MODEL_PATH)
    print(f"Model loaded in {time.time() - start:.2f}s")
    yield
    session = None


app = FastAPI(
    title="MLX Qwen3-ASR API",
    description="Speech-to-text API with browser microphone UI powered by Qwen3-ASR",
    version="0.1.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    """Serve the microphone transcription web app."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


def _stream_payload(state: Any, event: str, is_final: bool) -> dict[str, Any]:
    return {
        "event": event,
        "text": getattr(state, "text", ""),
        "stable_text": getattr(state, "stable_text", ""),
        "language": getattr(state, "language", "unknown"),
        "is_final": is_final,
    }


@app.websocket("/ws/stream")
async def stream_transcribe(websocket: WebSocket):
    """Stream microphone PCM to incremental ASR over WebSocket."""
    await websocket.accept()
    state: Any | None = None

    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                payload = json.loads(message["text"])
                action = payload.get("action")

                if action == "start":
                    language = payload.get("language") or None
                    chunk_size_ms = max(250, int(payload.get("chunk_size_ms", 2000)))
                    state = session.init_streaming(
                        language=language,
                        sample_rate=16000,
                        chunk_size_sec=chunk_size_ms / 1000.0,
                    )
                    await websocket.send_json({
                        "event": "ready",
                        "chunk_size_ms": chunk_size_ms,
                        "sample_rate": 16000,
                    })
                    continue

                if action == "stop":
                    if state is None:
                        await websocket.send_json({"event": "error", "message": "stream not started"})
                        continue

                    state = await asyncio.to_thread(session.finish_streaming, state)
                    await websocket.send_json(_stream_payload(state, event="final", is_final=True))
                    state = None
                    continue

                await websocket.send_json({"event": "error", "message": f"unknown action: {action}"})
                continue

            if "bytes" in message and message["bytes"] is not None:
                if state is None:
                    await websocket.send_json({"event": "error", "message": "stream not started"})
                    continue

                pcm = np.frombuffer(message["bytes"], dtype=np.float32)
                if pcm.size == 0:
                    continue

                state = await asyncio.to_thread(session.feed_audio, pcm, state)
                await websocket.send_json(_stream_payload(state, event="partial", is_final=False))
                continue

            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str | None = Query(None, description="Force language (e.g. Korean, English)"),
):
    """Transcribe an uploaded audio file."""
    start = time.time()

    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = session.transcribe(tmp_path, language=language)
        elapsed = time.time() - start

        return JSONResponse(content={
            "text": result.text,
            "language": result.language,
            "inference_time": round(elapsed, 3),
        })
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe/batch")
async def transcribe_batch(
    files: list[UploadFile] = File(...),
    language: str | None = Query(None),
):
    """Transcribe multiple audio files."""
    results = []
    total_start = time.time()

    for file in files:
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            start = time.time()
            result = session.transcribe(tmp_path, language=language)
            elapsed = time.time() - start
            results.append({
                "filename": file.filename,
                "text": result.text,
                "language": result.language,
                "inference_time": round(elapsed, 3),
            })
        finally:
            os.unlink(tmp_path)

    return JSONResponse(content={
        "results": results,
        "total_time": round(time.time() - total_start, 3),
        "count": len(results),
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

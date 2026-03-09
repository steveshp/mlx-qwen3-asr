"""FastAPI ASR server with batch transcription API."""

import asyncio
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from mlx_qwen3_asr import Session

MODEL_PATH = os.environ.get(
    "ASR_MODEL_PATH",
    os.path.expanduser("~/Downloads/Qwen3-ASR-1.7B-8bit"),
)

session: Any | None = None
_inference_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, _inference_lock
    from mlx_qwen3_asr import Session

    _inference_lock = asyncio.Lock()
    print(f"Loading model from {MODEL_PATH} ...")
    start = time.time()
    session = Session(model=MODEL_PATH)
    print(f"Model loaded in {time.time() - start:.2f}s")
    yield
    session = None


app = FastAPI(
    title="MLX Qwen3-ASR API",
    description="Batch speech-to-text API powered by Qwen3-ASR on Apple Silicon",
    version="0.2.0",
    lifespan=lifespan,
)
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


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
        async with _inference_lock:
            result = await asyncio.to_thread(session.transcribe, tmp_path, language=language)
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
            async with _inference_lock:
                result = await asyncio.to_thread(session.transcribe, tmp_path, language=language)
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

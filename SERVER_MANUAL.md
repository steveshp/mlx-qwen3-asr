# MLX Qwen3-ASR API Server 매뉴얼

## 1. 새 서버에서 처음부터 설치

> 다른 Mac(Apple Silicon)에서 이 프로젝트를 클론하여 서버를 띄우는 전체 과정입니다.

### 필수 요구사항

- macOS (Apple Silicon M1/M2/M3/M4)
- Python 3.10+
- Homebrew

### Step 1: 시스템 도구 설치

```bash
# Homebrew가 없으면 먼저 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# uv (Python 패키지 매니저)
brew install uv

# git-lfs (대용량 모델 파일 다운로드)
brew install git-lfs
git lfs install

# ffmpeg (mp3, m4a 등 비-WAV 오디오 포맷 지원)
brew install ffmpeg
```

### Step 2: 프로젝트 클론

```bash
git clone https://github.com/moona3k/mlx-qwen3-asr.git ~/projects/qe_asr
cd ~/projects/qe_asr
```

### Step 3: Python 가상환경 & 패키지 설치

```bash
uv venv .venv --python 3.13
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Step 4: 모델 다운로드

```bash
git clone https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit ~/Downloads/Qwen3-ASR-1.7B-8bit
cd ~/Downloads/Qwen3-ASR-1.7B-8bit
git lfs pull
cd ~/projects/qe_asr
```

다운로드 확인 (2.3G 이상이어야 정상):

```bash
du -h ~/Downloads/Qwen3-ASR-1.7B-8bit/model.safetensors
```

> 모델 크기: ~2.3GB (8-bit 양자화)

### Step 5: server.py 복사

이 레포에 포함된 `server.py` 파일이 이미 있으므로 별도 작업 불필요.
없는 경우 아래 내용으로 `server.py` 를 프로젝트 루트에 생성:

```python
"""FastAPI ASR server for MLX Qwen3-ASR 1.7B-8bit."""

import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse

from mlx_qwen3_asr import Session

MODEL_PATH = os.environ.get(
    "ASR_MODEL_PATH",
    os.path.expanduser("~/Downloads/Qwen3-ASR-1.7B-8bit"),
)

session: Session | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    print(f"Loading model from {MODEL_PATH} ...")
    start = time.time()
    session = Session(model=MODEL_PATH)
    print(f"Model loaded in {time.time() - start:.2f}s")
    yield
    session = None


app = FastAPI(
    title="MLX Qwen3-ASR API",
    description="Speech-to-text API powered by Qwen3-ASR 1.7B 8-bit on Apple Silicon",
    version="0.1.0",
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
```

### Step 6: 서버 실행 & 검증

```bash
source .venv/bin/activate
python server.py
```

다른 터미널에서 확인:

```bash
curl http://localhost:8000/health
```

### 한 줄 요약 (복사-붙여넣기용)

```bash
brew install uv git-lfs ffmpeg && git lfs install && \
git clone https://github.com/moona3k/mlx-qwen3-asr.git ~/projects/qe_asr && \
cd ~/projects/qe_asr && uv venv .venv --python 3.13 && source .venv/bin/activate && \
pip install -r requirements.txt && pip install -e . && \
git clone https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-8bit ~/Downloads/Qwen3-ASR-1.7B-8bit && \
cd ~/Downloads/Qwen3-ASR-1.7B-8bit && git lfs pull && \
cd ~/projects/qe_asr && python server.py
```

---

## 2. 서버 실행 (이미 설치된 환경)

```bash
cd ~/projects/qe_asr
source .venv/bin/activate
python server.py
```

서버가 `http://localhost:8000` 에서 시작됩니다.

### 웹앱 사용

브라우저에서 아래 주소를 열면 마이크 녹음 후 바로 전사할 수 있는 웹 UI가 표시됩니다.

```bash
open http://localhost:8000/
```

또는 브라우저 주소창에 직접 입력:

```text
http://localhost:8000/
```

### 다른 모델 경로 사용 시

```bash
ASR_MODEL_PATH=/path/to/model python server.py
```

### 백그라운드 실행

```bash
nohup python server.py > server.log 2>&1 &
echo $! > server.pid
```

### 백그라운드 서버 종료

```bash
kill $(cat server.pid)
```

---

## 3. API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/health` | 서버 상태 확인 |
| POST | `/transcribe` | 단일 파일 전사 |
| POST | `/transcribe/batch` | 다중 파일 배치 전사 |
| GET | `/docs` | Swagger UI (웹 브라우저) |

---

## 4. 사용법

### Swagger UI (웹 브라우저)

브라우저에서 열기:
```
http://localhost:8000/docs
```
"Try it out" 클릭 → 파일 업로드 → Execute

### curl 명령어

#### 서버 상태 확인
```bash
curl http://localhost:8000/health
```

#### 단일 파일 전사
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.mp3"
```

응답:
```json
{
  "text": "안녕하세요. 오늘 날씨가 정말 좋습니다.",
  "language": "Korean",
  "inference_time": 0.456
}
```

#### 언어 지정 (선택)
```bash
curl -X POST "http://localhost:8000/transcribe?language=Korean" \
  -F "file=@audio.mp3"
```

#### 배치 전사 (여러 파일)
```bash
curl -X POST http://localhost:8000/transcribe/batch \
  -F "files=@file1.mp3" \
  -F "files=@file2.wav" \
  -F "files=@file3.m4a"
```

응답:
```json
{
  "results": [
    {"filename": "file1.mp3", "text": "...", "language": "Korean", "inference_time": 0.45},
    {"filename": "file2.wav", "text": "...", "language": "English", "inference_time": 0.37},
    {"filename": "file3.m4a", "text": "...", "language": "Japanese", "inference_time": 0.35}
  ],
  "total_time": 1.17,
  "count": 3
}
```

### Python 클라이언트

```python
import requests

# 단일 파일
with open("audio.mp3", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/transcribe",
        files={"file": f},
    )
    print(resp.json())

# 배치
files = [
    ("files", open("a.mp3", "rb")),
    ("files", open("b.wav", "rb")),
]
resp = requests.post("http://localhost:8000/transcribe/batch", files=files)
print(resp.json())
```

---

## 5. 지원 언어

Korean, Vietnamese, Chinese, English, Thai, Japanese, Arabic, Dutch, French, German, Hindi, Italian, Portuguese, Russian, Spanish, Turkish + 22개 중국어 방언

---

## 6. 지원 오디오 포맷

| 포맷 | 확장자 | 필요 조건 |
|------|--------|----------|
| WAV | .wav | 기본 지원 |
| MP3 | .mp3 | ffmpeg 필요 |
| M4A | .m4a | ffmpeg 필요 |
| FLAC | .flac | ffmpeg 필요 |
| MP4 | .mp4 | ffmpeg 필요 |

---

## 7. 성능 참고

Qwen3-ASR-1.7B 8-bit (Apple Silicon 기준):

| 언어 | 평균 응답 시간 |
|------|--------------|
| Korean | ~0.45s |
| English | ~0.37s |
| Chinese | ~0.29s |
| Japanese | ~0.35s |
| Vietnamese | ~0.45s |
| Thai | ~0.48s |

---

## 8. 문제 해결

### 포트 충돌 (Address already in use)
```bash
lsof -ti:8000 | xargs kill -9
python server.py
```

### 모델 로딩 실패
모델 파일이 LFS 포인터가 아닌 실제 파일인지 확인:
```bash
du -h ~/Downloads/Qwen3-ASR-1.7B-8bit/model.safetensors
# 2.3G 이상이어야 정상
```

4KB이면 LFS pull 필요:
```bash
cd ~/Downloads/Qwen3-ASR-1.7B-8bit
git lfs pull
```

### MP3 전사 실패
```bash
brew install ffmpeg
```

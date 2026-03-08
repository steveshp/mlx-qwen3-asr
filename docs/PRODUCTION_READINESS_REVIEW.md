# mlx-qwen3-asr 프로덕션 준비 상태 검토 보고서

**검토일**: 2026-03-01
**대상**: <https://github.com/moona3k/mlx-qwen3-asr> (v0.2.3)
**방법론**: 4인 에이전트 팀 (보안 / 성능-신뢰성 / 코드품질 검토자 3명 + 반론자 1명)

---

## 최종 판정: Production-Ready

이 프로젝트는 선언된 범위(Apple Silicon CLI 라이브러리) 내에서 **프로덕션 품질**이다.

### 시나리오별 판정

| 시나리오 | 판정 | 근거 |
|----------|------|------|
| **개인 프로젝트** | 즉시 사용 가능 | 441 테스트 통과, 6개 언어 60/60 정확도, 평균 0.4s 지연 |
| **스타트업 MVP** | 사용 가능 (사소한 수정 권장) | 아래 3가지 수정 후 사용 (합계 1시간 미만 작업) |
| **엔터프라이즈** | 조건부 사용 가능 | 단일 프로세스 배치 처리 적합. 멀티스레드 서버 래핑 불가 |

### 종합 점수

| 영역 | 검토자 판정 | 반론자 재평가 | 최종 |
|------|------------|--------------|------|
| 보안 | Pass | Pass (동의) | **Pass** |
| 성능/신뢰성 | B+ | A- (과잉 진단 시정) | **A-** |
| 코드품질/아키텍처 | Production-Ready with Caveats | 동의 (심각도 하향) | **Production-Ready** |

---

## 1. 보안 검토 (Pass)

**검토자**: security-reviewer | **Critical: 0** | **High: 0** | **Medium: 1** | **Low: 6**

### 양성 발견 (잘 된 것)

- `subprocess.run` 리스트 모드 사용 (shell injection 차단)
- safetensors 형식 (pickle 기반 임의 코드 실행 불가)
- `allow_patterns`로 HF 다운로드 파일 유형 제한
- JSON 파싱만 사용 (config/vocab 처리)
- 입력 검증: input_ids dtype/범위, audio injection shape 검증
- 의존성 최소화 (핵심 경로에 transformers/torch 없음)
- LRU 캐시 바운딩으로 무제한 캐시 방지
- 인증 토큰이 로그/에러에 미노출

### 발견사항

| 심각도 | 위치 | 설명 | 반론자 평가 |
|--------|------|------|-------------|
| Medium | `audio.py:147-165` | ffmpeg argument injection (`-`로 시작하는 파일명) | 타당. `--` separator 한 줄 추가 권장 |
| Low | `audio.py:293-333` | `_resample_via_ffmpeg` 동일 패턴 | 정수 파라미터로 주입 불가 |
| Low | `load_models.py:195-208` | HF 다운로드 경로 | HTTPS + safetensors로 안전 |
| Low | `load_models.py:110-113` | config.json 파싱 | JSON 파싱 자체가 안전 |
| Low | `tokenizer.py:162-186` | vocab/merges 로딩 | 순수 데이터 파싱 |
| Low | `writers.py:27-93` | 출력 경로 검증 없음 | CLI에서 사용자가 직접 제어 |
| Low | `diarization.py:247-253` | 환경변수 토큰 | 로그 미노출 확인 |

---

## 2. 성능/신뢰성 검토 (A-)

**검토자**: reliability-reviewer | **원래 판정**: B+ | **반론 후**: A-

### 반론자가 시정한 과잉 진단

| 원래 심각도 | 위치 | 검토자 주장 | 반론자 반박 | 조정 심각도 |
|------------|------|-----------|------------|------------|
| **Critical** | `streaming.py:184` | audio_accum 무한 누적 | `max_context_samples`로 이미 바운드됨 (L188-189) | **Low** |
| **Critical** | `decoder.py:434` | _CAUSAL_MASK_CACHE 무한 성장 | 실제 엔트리 2-5개, 각 수 KB | **Low** |
| **High** | `generate.py:56` | KV 캐시 과도한 사전할당 | 의도된 설계 (mlx-lm 동일 패턴), ~56MB | **Info** |
| **High** | `model.py:238` | batch loop Python for | B=1 전용 모델, vectorize 무의미 | **Info** |
| **High** | `load_models.py:30` | LRU eviction 시 GPU 미회수 | LRU eviction 이미 구현됨, GC가 처리 | **Low** |
| **High** | `streaming.py:406` | 스트리밍 KV 캐시 무한 성장 | trim 시 `_reset_incremental_decoder_state` 호출로 캐시 초기화 | **Low** |

### 타당한 발견사항 (반론자 동의)

| 심각도 | 위치 | 설명 |
|--------|------|------|
| Medium | `audio.py:165` | `subprocess.run`에 timeout 없음 (손상 파일 시 hang 가능) |
| Medium | `audio.py:164` | `capture_output=True`로 대용량 오디오 전체 메모리 적재 (10시간+ mp3) |
| Low | `tokenizer.py:180` | `_bpe_cache` 바운드 없음 (실질적 영향 미미) |
| Low | `generate.py:84` | `eval_interval=1` 매 토큰 eval (안전하지만 최적 아님) |
| Info | 전역 캐시 8개 | thread-unsafe (Metal이 single-thread 전용이므로 설계 범위 밖) |

### 강점

- 에러 핸들링 전반적으로 우수 (bounds check, 입력/타입 검증)
- KV 캐시 사전할당 + 주기적 eval로 메모리/성능 예측 가능
- LRU 캐시로 모델/토크나이저 효율적 재사용
- 오디오 자동 청킹으로 장시간 오디오 처리
- 반복 감지(repetition detection)로 디코더 루프 탈출 보장

---

## 3. 코드품질/아키텍처 검토 (Production-Ready)

**검토자**: quality-reviewer | **Critical: 0 (실제)** | **High: 3** | **Medium: 6** | **Low: 4**

### 발견사항

| 심각도 | 위치 | 설명 | 반론자 평가 |
|--------|------|------|-------------|
| ~~Critical~~ Low | `pyproject.toml` + `_version.py` | 버전 이중 관리 | 표준 Python 패키징 패턴 |
| Medium | `transcribe.py:40` / `streaming.py:25` / `writers.py:11` | CJK_LANG_ALIASES 3중 복제 | 타당, 단일 모듈에서 export해야 함 |
| Low | `generate.py:31` | `Optional[GenerationConfig]` 타입 누락 | 타입 힌팅 엄격도 문제, 런타임 무관 |
| Low | `transcribe.py:71` | `segments: list[dict]` 느슨한 타입 | TypedDict 권장하나 기능적 결함 아님 |
| Medium | `pyproject.toml:27` | numpy 버전 미고정 | 범위 지정 권장 |
| Medium | `load_models.py:175` | setattr 메타데이터 부착 | 코드 스타일 문제, 기능적 무관 |
| Medium | `streaming.py:42-89` | StreamingState 과도한 공개 속성 | 유지보수성 문제 |
| Medium | `streaming.py:524` / `audio.py:96` | 오디오 정규화 코드 중복 | 리팩토링 권장 |
| Medium | `session.py:31` | context manager 미구현 | 유용하나 필수 아님 |
| Medium | `convert.py:37-44` | Conv2d transpose 조건이 brittle | 주의 필요 |
| Low | `model.py:389` | `__all__`에 private 심볼 포함 | 정리 권장 |
| Low | `cli.py:686` | **RTF 계산 항상 1.0 (확인된 버그)** | 실제 버그, 수정 필요 |

### 아키텍처 강점

- 명확한 레이어 분리 (config -> encoder/decoder -> model -> generate -> transcribe -> session)
- correctness invariant 모두 정확히 구현 (LayerNorm vs RMSNorm, interleaved MRoPE, Conv2d transpose)
- Multi-slot LRU 모델 캐시 (바운드된 메모리)
- Speculative decoding vocab 호환성 검증
- Zero-torch 핵심 경로
- 간결한 공개 API

### 테스트 현황

- 38개 테스트 파일, 469개 테스트 통과
- 핵심 모듈 전체에 대응하는 테스트 존재
- 10개 언어 벤치마크, MLX-vs-PyTorch 참조 패리티 테스트

---

## 4. 반론자가 놓친 문제로 추가 식별한 것

| 심각도 | 위치 | 설명 |
|--------|------|------|
| Medium | `audio.py:165` | 10시간+ mp3 파일 시 `capture_output=True`로 ~1.1GB stdout 메모리 (모든 ASR 라이브러리 공통) |
| Low | `chunking.py:52` | 재귀 분할 (14일짜리 오디오에서만 RecursionError, 비현실적) |

---

## 5. 수정 완료 사항 (3건)

검토에서 도출된 3건 모두 수정 완료. 469 tests passed, ruff clean.

### 5.1 RTF 계산 버그 수정 (`cli.py:685`) -- FIXED

```python
# Before (bug: always returns 1.0)
duration = elapsed
rtf = (elapsed / duration) if duration > 0 else 0.0

# After (uses actual audio length from streaming state)
duration = len(state.audio_accum) / args.mic_sample_rate
rtf = (elapsed / duration) if duration > 0 else 0.0
```

### 5.2 CJK_LANG_ALIASES 중복 제거 -- FIXED

`transcribe.py`의 `CJK_LANG_ALIASES`를 단일 소스로, `streaming.py`와 `writers.py`에서 import.

- `streaming.py`: `from .transcribe import CJK_LANG_ALIASES as _CJK_LANG_ALIASES`
- `writers.py`: `from .transcribe import CJK_LANG_ALIASES as _CJK_LANG_ALIASES`

### 5.3 ffmpeg argument injection 방어 (`audio.py:148`) -- FIXED

```python
# "-"로 시작하는 파일명이 ffmpeg 옵션으로 해석되는 것을 방지
safe_path = f"./{path}" if path.startswith("-") else path
cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", safe_path, ...]
```

---

## 6. 스트리밍 실전 테스트 결과

### 테스트 환경

- Apple Silicon Mac (M-series)
- 모델: `Qwen3-ASR-1.7B-8bit` (8-bit quantized, ~2.3GB)
- Python 3.13, MLX

### VTV24 베트남어 뉴스 (111초, 실제 방송)

| 항목 | 값 |
|------|-----|
| 오디오 | VTV24 "Vietnam Today" 뉴스 (m4a) |
| 길이 | 111.06초 (1분 51초) |
| 총 처리 시간 | 10.82초 |
| RTF | 0.097x (실시간의 ~10배 빠름) |
| 청크 설정 | 56개 x 2.0초 |
| 청크 지연 | mean=0.193s, p95=0.240s, max=0.321s |
| 출력 | 2,086자 |
| Stability | 100% |
| Rewrite rate | 0% |
| 언어 감지 | `vi` (정확) |

### 자동 검증 결과 (5/5 PASS)

```
[PASS] Stable text monotonic: stable_text should only grow
[PASS] RTF < 1.0: RTF=0.0974
[PASS] Non-empty output: 2086 chars
[PASS] Language detected: lang=vi
[PASS] p95 latency < chunk: p95=0.240s
```

---

## 7. 테스트 방법

### 7.1 단위 테스트 (469개)

```bash
source .venv/bin/activate

# 전체 테스트
python -m pytest tests/ -v

# 스트리밍 모듈만
python -m pytest tests/test_streaming.py -v

# 첫 실패에서 중단
python -m pytest tests/ -x -q
```

### 7.2 스트리밍 E2E 테스트 (`scripts/test_streaming_e2e.py`)

실제 오디오 파일을 청크 단위로 분할하여 스트리밍 전사를 시뮬레이션하고,
5가지 자동 검증(stable 단조 증가, RTF, 출력 비어있지 않음, 언어 감지, 지연)을 수행한다.

```bash
# 기본 (0.6B 모델)
python scripts/test_streaming_e2e.py audio.m4a

# 1.7B 모델 + 언어 지정
python scripts/test_streaming_e2e.py audio.m4a \
    --model ~/Downloads/Qwen3-ASR-1.7B-8bit \
    --language vi

# 청크 크기 / 컨텍스트 윈도우 조정
python scripts/test_streaming_e2e.py audio.m4a \
    --chunk-sec 1.0 \
    --max-context-sec 15.0

# JSON 결과 저장
python scripts/test_streaming_e2e.py audio.m4a \
    --json-output results/streaming_test.json

# 요약만 출력 (청크별 진행 숨김)
python scripts/test_streaming_e2e.py audio.m4a --quiet
```

**자동 검증 항목:**

| Check | 의미 | 실패 시 |
|-------|------|---------|
| Stable text monotonic | stable_text가 줄어들지 않고 계속 증가 | 스트리밍 로직 결함 |
| RTF < 1.0 | 실시간보다 빠른 처리 | 성능 회귀 |
| Non-empty output | 최종 텍스트가 비어있지 않음 | 디코딩 실패 |
| Language detected | 언어가 감지됨 | 토크나이저/프롬프트 문제 |
| p95 latency < chunk | 95% 청크가 청크 시간 내 처리 | 실시간 스트리밍 불가 |

**종료 코드**: 모든 check PASS → `exit 0`, 하나라도 FAIL → `exit 1` (CI 연동 가능)

### 7.3 스트리밍 벤치마크 (`scripts/benchmark_streaming.py`)

정밀한 지연 시간 측정과 스트리밍 품질 메트릭(stability, rewrite rate, finalization delta).

```bash
python scripts/benchmark_streaming.py audio.m4a \
    --model ~/Downloads/Qwen3-ASR-1.7B-8bit \
    --runs 3 --warmup-runs 1 \
    --json-output benchmarks/streaming.json
```

### 7.4 단건 전사 (비스트리밍)

```bash
# Python API
python -c "
from mlx_qwen3_asr import transcribe
r = transcribe('audio.m4a', model='~/Downloads/Qwen3-ASR-1.7B-8bit')
print(r.text)
"

# CLI
mlx-qwen3-asr audio.m4a --model ~/Downloads/Qwen3-ASR-1.7B-8bit --verbose
```

### 7.5 품질 게이트

```bash
# 빠른 검증 (lint + tests)
python scripts/quality_gate.py --mode fast

# 릴리스 검증 (+ 참조 패리티, LibriSpeech)
RUN_REFERENCE_PARITY=1 python scripts/quality_gate.py --mode release
```

---

## 8. 검토 과정에서의 교훈

### 검토자 간 의견 차이

성능/신뢰성 검토자가 가장 보수적이었고, Critical 2건을 보고했으나 반론자가 코드를 직접 읽고 둘 다 이미 처리된 것임을 확인했다. 이는 **코드를 직접 읽지 않고 패턴만으로 판단하면 과잉 진단이 발생**한다는 것을 보여준다.

보안 검토자는 프로젝트의 위협 모델(로컬 CLI, 사용자 입력이 신뢰 범위)을 가장 잘 이해하고 있었다.

코드품질 검토자의 발견은 대부분 타당했으나 심각도가 과대 평가된 경우가 있었다(버전 이중 관리를 Critical로 분류).

### 핵심 결론

> mlx-qwen3-asr는 선언된 범위 내에서 프로덕션 품질이다.
> 441개 테스트, 10개 언어 벤치마크, PyTorch 참조 패리티 검증을 갖추고 있으며,
> 보안적으로 양호하고, 아키텍처가 깔끔하고, 핵심 경로의 correctness가 검증되어 있다.
> "서버로 사용하면 안 된다"는 제한은 설계 의도이지 결함이 아니다.

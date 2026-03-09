const recordButton = document.getElementById("record-btn");
const clearButton = document.getElementById("clear-btn");
const streamButton = document.getElementById("stream-btn");
const streamStatus = document.getElementById("stream-status");
const languageSelect = document.getElementById("language");
const transcriptBox = document.getElementById("transcript");
const messageLine = document.getElementById("message");
const serverStatus = document.getElementById("server-status");
const recorderStatus = document.getElementById("recorder-status");
const clipLength = document.getElementById("clip-length");
const detectedLanguage = document.getElementById("detected-language");
const inferenceTime = document.getElementById("inference-time");
const partialMode = document.getElementById("partial-mode");
const levelMeter = document.getElementById("level-meter");

let mediaRecorder = null;
let audioChunks = [];
let audioStream = null;
let audioContext = null;
let analyser = null;
let sourceNode = null;
let levelTimer = null;
let recordingStartedAt = 0;

// Streaming state (batch-only, no WebSocket)
let streamingActive = false;
let vadInstance = null;
let vadListening = false;

let allCapturedPcm = [];
let batchCorrectionTimer = null;
let batchInFlight = false;

// Sliding window: transcribe all audio up to MAX_WINDOW_SEC.
// When buffer exceeds this, trim oldest audio & lock previous text as confirmed.
let confirmedText = "";
const MAX_WINDOW_SEC = 30;
const MAX_WINDOW_SAMPLES = MAX_WINDOW_SEC * 16000;

const BATCH_INTERVAL_MS = 3000;

function setStatus(element, text, tone) {
  element.textContent = text;
  element.className = `pill ${tone}`;
}

function setMessage(text) {
  messageLine.textContent = text;
}

function resetMeter() {
  levelMeter.style.width = "1%";
}

function chooseMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((c) => MediaRecorder.isTypeSupported(c)) || "";
}

function blobExtension(blobType) {
  if (blobType.includes("mp4")) return "m4a";
  if (blobType.includes("ogg")) return "ogg";
  return "webm";
}

function updateDuration() {
  const seconds = (Date.now() - recordingStartedAt) / 1000;
  clipLength.textContent = `${seconds.toFixed(1)}s`;
}

function startLevelMeter() {
  if (!analyser) return;
  const data = new Uint8Array(analyser.frequencyBinCount);
  levelTimer = window.setInterval(() => {
    analyser.getByteTimeDomainData(data);
    let peak = 0;
    for (const sample of data) {
      peak = Math.max(peak, Math.abs(sample - 128));
    }
    const width = Math.max(1, Math.min(100, Math.round((peak / 128) * 100)));
    levelMeter.style.width = `${width}%`;
    updateDuration();
  }, 80);
}

function stopLevelMeter() {
  if (levelTimer) {
    window.clearInterval(levelTimer);
    levelTimer = null;
  }
  resetMeter();
}

async function ensureServer() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`상태 확인 실패: ${response.status}`);
    const payload = await response.json();
    setStatus(serverStatus, "서버 준비 완료", "pill-ok");
    setMessage(`로드된 모델: ${payload.model}`);
  } catch (error) {
    setStatus(serverStatus, "서버 연결 불가", "pill-warn");
    setMessage(`서버에 연결할 수 없습니다: ${error.message}`);
  }
}

// ── Utility: Float32 PCM → 16-bit WAV Blob ──

function pcmToWavBlob(float32Array, sampleRate) {
  const numSamples = float32Array.length;
  const buffer = new ArrayBuffer(44 + numSamples * 2);
  const view = new DataView(buffer);

  function writeString(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + numSamples * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, numSamples * 2, true);

  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

// ── Record mode ──

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
    return;
  }
  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new AudioContext();
    sourceNode = audioContext.createMediaStreamSource(audioStream);
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 1024;
    sourceNode.connect(analyser);

    const mimeType = chooseMimeType();
    mediaRecorder = mimeType ? new MediaRecorder(audioStream, { mimeType }) : new MediaRecorder(audioStream);
    audioChunks = [];
    recordingStartedAt = Date.now();

    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    });
    mediaRecorder.addEventListener("stop", async () => {
      await submitRecording();
      cleanupRecorder();
    });

    mediaRecorder.start();
    recordButton.textContent = "녹음 종료";
    recordButton.classList.add("recording");
    setStatus(recorderStatus, "녹음 중", "pill-live");
    setMessage("녹음 중입니다. 끝나면 녹음 종료를 누르세요.");
    detectedLanguage.textContent = "-";
    inferenceTime.textContent = "-";
    startLevelMeter();
  } catch (error) {
    cleanupRecorder();
    setMessage(`마이크 접근 실패: ${error.message}`);
  }
}

function stopRecording() {
  if (!mediaRecorder) return;
  recordButton.disabled = true;
  setMessage("ASR 서버로 오디오를 업로드하는 중입니다...");
  mediaRecorder.stop();
}

function cleanupRecorder() {
  stopLevelMeter();
  if (sourceNode) { sourceNode.disconnect(); sourceNode = null; }
  if (audioStream) { audioStream.getTracks().forEach((t) => t.stop()); audioStream = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  analyser = null;
  mediaRecorder = null;
  recordButton.disabled = false;
  recordButton.textContent = "녹음 시작";
  recordButton.classList.remove("recording");
  setStatus(recorderStatus, "녹음 대기", "pill-muted");
}

async function submitRecording() {
  if (!audioChunks.length) { setMessage("녹음된 오디오가 없습니다."); return; }
  const blobType = audioChunks[0].type || "audio/webm";
  const audioBlob = new Blob(audioChunks, { type: blobType });
  const formData = new FormData();
  formData.append("file", audioBlob, `mic-input.${blobExtension(blobType)}`);
  const language = languageSelect.value.trim();
  const endpoint = language ? `/transcribe?language=${encodeURIComponent(language)}` : "/transcribe";
  try {
    const response = await fetch(endpoint, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`전사 실패: ${response.status}`);
    const payload = await response.json();
    transcriptBox.value = payload.text || "";
    detectedLanguage.textContent = payload.language || "-";
    inferenceTime.textContent = payload.inference_time ? `${payload.inference_time}s` : "-";
    setMessage("전사가 완료되었습니다.");
  } catch (error) {
    setMessage(error.message);
  }
}

// ── Streaming mode (batch-only, no WebSocket) ──
//
// No WebSocket streaming — avoids Metal GPU concurrent access crash.
// VAD captures 16kHz audio. Every 5s, accumulated PCM → WAV → POST /transcribe.
// Single model access at a time. Simple and stable.

function cleanupStreaming() {
  streamingActive = false;
  if (batchCorrectionTimer) { clearInterval(batchCorrectionTimer); batchCorrectionTimer = null; }
  allCapturedPcm = [];
  batchInFlight = false;
  confirmedText = "";
  if (vadInstance && vadListening) { vadInstance.pause(); vadListening = false; }
  streamButton.disabled = false;
  streamButton.textContent = "스트리밍 시작";
  streamButton.classList.remove("recording");
  setStatus(streamStatus, "스트리밍 대기", "pill-muted");
  partialMode.textContent = "비활성";
  resetMeter();
}

async function doBatchCorrection() {
  if (batchInFlight || allCapturedPcm.length === 0) return;

  const totalLen = allCapturedPcm.reduce((sum, f) => sum + f.length, 0);
  if (totalLen < 16000) return; // At least 1 second

  batchInFlight = true;

  // If buffer exceeds window, trim oldest audio and lock current text
  if (totalLen > MAX_WINDOW_SAMPLES) {
    const excess = totalLen - MAX_WINDOW_SAMPLES;
    let dropped = 0;
    while (allCapturedPcm.length > 0 && dropped + allCapturedPcm[0].length <= excess) {
      dropped += allCapturedPcm.shift().length;
    }
    // Lock whatever was on screen as confirmed text (will be prepended)
    confirmedText = transcriptBox.value;
    console.log(`[batch] trimmed ${(dropped/16000).toFixed(1)}s, locked ${confirmedText.length} chars`);
  }

  // Merge all remaining PCM (up to ~30s)
  const windowLen = allCapturedPcm.reduce((sum, f) => sum + f.length, 0);
  const merged = new Float32Array(windowLen);
  let offset = 0;
  for (const f of allCapturedPcm) {
    merged.set(f, offset);
    offset += f.length;
  }

  const wavBlob = pcmToWavBlob(merged, 16000);
  const formData = new FormData();
  formData.append("file", wavBlob, "stream-batch.wav");
  const language = languageSelect.value.trim();
  const endpoint = language ? `/transcribe?language=${encodeURIComponent(language)}` : "/transcribe";

  try {
    console.log(`[batch] sending ${(windowLen/16000).toFixed(1)}s audio...`);
    partialMode.textContent = "전사 중...";
    const response = await fetch(endpoint, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`batch failed: ${response.status}`);
    const payload = await response.json();

    const windowText = (payload.text || "").trim();
    // Display: confirmed (locked old text) + window result
    if (confirmedText) {
      transcriptBox.value = confirmedText + "\n" + windowText;
    } else {
      transcriptBox.value = windowText;
    }
    detectedLanguage.textContent = payload.language || "-";
    inferenceTime.textContent = payload.inference_time ? `${payload.inference_time}s` : "-";
    partialMode.textContent = "전사 완료";
    console.log(`[batch] result: ${windowText.slice(0, 80)}`);
  } catch (error) {
    console.error("[batch] error:", error.message);
  } finally {
    batchInFlight = false;
  }
}

async function startStreaming() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
    return;
  }

  try {
    streamButton.disabled = true;
    setMessage("오디오 파이프라인 로딩 중...");

    allCapturedPcm = [];
    batchInFlight = false;
    confirmedText = "";

    // Periodic batch transcription every 3 seconds
    batchCorrectionTimer = setInterval(() => {
      if (streamingActive && !batchInFlight) {
        doBatchCorrection();
      }
    }, BATCH_INTERVAL_MS);

    // VAD as 16kHz audio pipeline only
    vadInstance = await vad.MicVAD.new({
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      minSpeechMs: 300,
      preSpeechPadMs: 300,
      redemptionMs: 500,
      startOnLoad: false,

      onSpeechStart: () => {},
      onSpeechEnd: () => {},
      onVADMisfire: () => {},

      onFrameProcessed: (probs, frame) => {
        const width = Math.max(1, Math.min(100, Math.round(probs.isSpeech * 100)));
        levelMeter.style.width = `${width}%`;

        if (frame) {
          allCapturedPcm.push(new Float32Array(frame));
        }
      },
    });

    streamingActive = true;
    recordingStartedAt = Date.now();
    vadInstance.start();
    vadListening = true;

    streamButton.disabled = false;
    streamButton.textContent = "스트리밍 종료";
    streamButton.classList.add("recording");
    setStatus(streamStatus, "실시간 캡처 중", "pill-live");
    partialMode.textContent = "캡처 중";
    setMessage("마이크 캡처 중. 5초마다 자동 전사합니다.");
  } catch (error) {
    cleanupStreaming();
    setMessage(`스트리밍 시작 실패: ${error.message}`);
  }
}

async function stopStreaming() {
  if (vadInstance && vadListening) { vadInstance.pause(); vadListening = false; }
  if (batchCorrectionTimer) { clearInterval(batchCorrectionTimer); batchCorrectionTimer = null; }

  streamButton.disabled = true;
  setMessage("최종 전사 중...");
  partialMode.textContent = "최종 전사 중";
  streamingActive = false;

  // Final batch transcription of all audio
  await doBatchCorrection();

  setMessage("전사 완료.");
  partialMode.textContent = "전사 완료";
  streamButton.disabled = false;
  streamButton.textContent = "스트리밍 시작";
  streamButton.classList.remove("recording");
  setStatus(streamStatus, "스트리밍 대기", "pill-muted");
  resetMeter();

  allCapturedPcm = [];
  batchInFlight = false;
  // confirmedText is intentionally kept so transcript remains visible after stop
}

// ── Event listeners ──

recordButton.addEventListener("click", async () => {
  if (streamingActive || vadListening) {
    setMessage("스트리밍 중에는 일반 녹음을 시작할 수 없습니다.");
    return;
  }
  if (mediaRecorder && mediaRecorder.state === "recording") {
    stopRecording();
    return;
  }
  await startRecording();
});

streamButton.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    setMessage("일반 녹음 중에는 스트리밍을 시작할 수 없습니다.");
    return;
  }
  if (streamingActive || vadListening) {
    await stopStreaming();
    return;
  }
  await startStreaming();
});

clearButton.addEventListener("click", () => {
  transcriptBox.value = "";
  confirmedText = "";
  clipLength.textContent = "0.0s";
  detectedLanguage.textContent = "-";
  inferenceTime.textContent = "-";
  partialMode.textContent = streamingActive ? "캡처 중" : "비활성";
  setMessage("전사 결과를 지웠습니다.");
});

window.addEventListener("beforeunload", () => {
  cleanupRecorder();
  if (vadInstance) { vadInstance.destroy(); vadInstance = null; }
  cleanupStreaming();
});

ensureServer();

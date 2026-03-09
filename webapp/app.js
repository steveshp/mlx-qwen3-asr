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

// Streaming state
let streamSocket = null;
let streamingActive = false;
let vadInstance = null;
let vadListening = false;
let flushTimer = null;
let pendingFrames = [];

// Periodic batch correction state
let allCapturedPcm = [];       // All 16kHz PCM frames since stream start
let batchCorrectionTimer = null;
let batchInFlight = false;      // Prevent overlapping batch requests
let correctedText = "";         // Latest batch-corrected text
let correctedSamples = 0;       // How many samples have been batch-corrected

const BATCH_INTERVAL_MS = 5000; // Batch correct every 5 seconds

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
  view.setUint32(16, 16, true);         // chunk size
  view.setUint16(20, 1, true);          // PCM
  view.setUint16(22, 1, true);          // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);          // block align
  view.setUint16(34, 16, true);         // bits per sample
  writeString(36, "data");
  view.setUint32(40, numSamples * 2, true);

  // Convert float32 [-1,1] to int16
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

// ── Streaming mode ──
//
// Dual pipeline:
//   1. WebSocket streaming → real-time partial text (fast, less accurate)
//   2. Every 5s → POST accumulated PCM as WAV to /transcribe (accurate batch)
//      → batch result replaces partial text
// Manual stop → one final batch correction of all audio

function cleanupStreaming() {
  streamingActive = false;
  if (flushTimer) { clearInterval(flushTimer); flushTimer = null; }
  if (batchCorrectionTimer) { clearInterval(batchCorrectionTimer); batchCorrectionTimer = null; }
  pendingFrames = [];
  allCapturedPcm = [];
  batchInFlight = false;
  correctedText = "";
  correctedSamples = 0;
  if (vadInstance && vadListening) { vadInstance.pause(); vadListening = false; }
  if (streamSocket) { streamSocket.close(); streamSocket = null; }
  streamButton.disabled = false;
  streamButton.textContent = "스트리밍 시작";
  streamButton.classList.remove("recording");
  setStatus(streamStatus, "스트리밍 대기", "pill-muted");
  partialMode.textContent = "비활성";
  resetMeter();
}

// Send accumulated PCM to /transcribe for batch correction
async function doBatchCorrection() {
  if (batchInFlight || allCapturedPcm.length === 0) return;

  const totalLen = allCapturedPcm.reduce((sum, f) => sum + f.length, 0);
  if (totalLen < 16000) return; // At least 1 second

  batchInFlight = true;
  const snapshotLen = totalLen;

  // Merge all PCM into one array
  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const f of allCapturedPcm) {
    merged.set(f, offset);
    offset += f.length;
  }

  // Convert to WAV and POST
  const wavBlob = pcmToWavBlob(merged, 16000);
  const formData = new FormData();
  formData.append("file", wavBlob, "stream-batch.wav");
  const language = languageSelect.value.trim();
  const endpoint = language ? `/transcribe?language=${encodeURIComponent(language)}` : "/transcribe";

  try {
    console.log(`[batch] sending ${(totalLen/16000).toFixed(1)}s audio for correction...`);
    const response = await fetch(endpoint, { method: "POST", body: formData });
    if (!response.ok) throw new Error(`batch failed: ${response.status}`);
    const payload = await response.json();

    correctedText = payload.text || "";
    correctedSamples = snapshotLen;
    detectedLanguage.textContent = payload.language || "-";
    inferenceTime.textContent = payload.inference_time ? `${payload.inference_time}s` : "-";

    // Replace displayed text with batch-corrected version
    transcriptBox.value = correctedText;
    partialMode.textContent = "보정 완료";
    console.log(`[batch] corrected: ${correctedText.slice(0, 80)}`);
  } catch (error) {
    console.error("[batch] error:", error.message);
  } finally {
    batchInFlight = false;
  }
}

function handleStreamMessage(event) {
  const payload = JSON.parse(event.data);

  if (payload.event === "ready") {
    streamingActive = true;
    streamButton.disabled = false;
    streamButton.textContent = "스트리밍 종료";
    streamButton.classList.add("recording");
    setStatus(streamStatus, "실시간 전사 중", "pill-live");
    partialMode.textContent = "실시간";
    setMessage("실시간 전사 중. 5초마다 자동 보정합니다.");
    recordingStartedAt = Date.now();
    return;
  }

  if (payload.event === "partial") {
    // Only show streaming partial if no batch correction yet or streaming is ahead
    if (!batchInFlight) {
      const streamText = payload.text || "";
      // If batch-corrected text exists and streaming text is longer, show corrected + new part
      if (correctedText && streamText.length > correctedText.length) {
        transcriptBox.value = correctedText + streamText.slice(correctedText.length);
      } else if (!correctedText) {
        transcriptBox.value = streamText;
      }
      // If streaming text is shorter/same as corrected, keep corrected (it's more accurate)
    }
    updateDuration();
    return;
  }

  if (payload.event === "final") {
    // Streaming final — will be overridden by last batch correction
    return;
  }

  if (payload.event === "refined") {
    // Server-side refined — but we're using our own batch correction, ignore
    return;
  }

  if (payload.event === "error") {
    console.error("[ws] error:", payload.message);
    setMessage(`스트리밍 오류: ${payload.message}`);
  }
}

function openStreamSocket() {
  return new Promise((resolve, reject) => {
    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      console.log("[ws] connected, sending start");
      ws.send(JSON.stringify({
        action: "start",
        language: languageSelect.value.trim(),
        chunk_size_ms: 2000,
      }));
    };

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.event === "ready" && !streamingActive) {
        handleStreamMessage(event);
        resolve(ws);
        return;
      }
      handleStreamMessage(event);
    };

    ws.onclose = () => {
      console.log("[ws] closed");
      if (streamingActive) cleanupStreaming();
    };

    ws.onerror = () => reject(new Error("WebSocket 연결 실패"));
  });
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
    correctedText = "";
    correctedSamples = 0;
    batchInFlight = false;

    // Buffer VAD frames, flush to WebSocket every 500ms
    pendingFrames = [];
    flushTimer = setInterval(() => {
      if (!streamingActive || !streamSocket || streamSocket.readyState !== WebSocket.OPEN) return;
      if (pendingFrames.length === 0) return;

      const totalLen = pendingFrames.reduce((sum, f) => sum + f.length, 0);
      const merged = new Float32Array(totalLen);
      let offset = 0;
      for (const f of pendingFrames) {
        merged.set(f, offset);
        offset += f.length;
      }
      pendingFrames = [];

      streamSocket.send(merged.buffer);
      updateDuration();
    }, 500);

    // Periodic batch correction every 5 seconds
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
          const copy = new Float32Array(frame);
          pendingFrames.push(copy);
          allCapturedPcm.push(copy);
        }
      },
    });

    // Open WebSocket for streaming partials
    streamSocket = await openStreamSocket();

    vadInstance.start();
    vadListening = true;
    streamButton.disabled = false;
  } catch (error) {
    cleanupStreaming();
    setMessage(`스트리밍 시작 실패: ${error.message}`);
  }
}

async function stopStreaming() {
  if (vadInstance && vadListening) { vadInstance.pause(); vadListening = false; }
  if (flushTimer) { clearInterval(flushTimer); flushTimer = null; }
  if (batchCorrectionTimer) { clearInterval(batchCorrectionTimer); batchCorrectionTimer = null; }

  // Final batch correction of all audio
  streamButton.disabled = true;
  setMessage("최종 배치 보정 중...");
  partialMode.textContent = "최종 보정 중";

  // Close WebSocket (don't need streaming anymore)
  if (streamSocket) { streamSocket.close(); streamSocket = null; }
  streamingActive = false;

  // Do final full batch correction
  await doBatchCorrection();

  setMessage("최종 보정 완료.");
  partialMode.textContent = "최종 보정 완료";
  streamButton.disabled = false;
  streamButton.textContent = "스트리밍 시작";
  streamButton.classList.remove("recording");
  setStatus(streamStatus, "스트리밍 대기", "pill-muted");
  resetMeter();

  // Cleanup remaining state
  allCapturedPcm = [];
  correctedText = "";
  correctedSamples = 0;
  pendingFrames = [];
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
  clipLength.textContent = "0.0s";
  detectedLanguage.textContent = "-";
  inferenceTime.textContent = "-";
  partialMode.textContent = streamingActive ? "활성" : "비활성";
  setMessage("전사 결과를 지웠습니다.");
});

window.addEventListener("beforeunload", () => {
  cleanupRecorder();
  if (vadInstance) { vadInstance.destroy(); vadInstance = null; }
  cleanupStreaming();
});

ensureServer();

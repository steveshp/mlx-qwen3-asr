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
let streamSocket = null;
let processorNode = null;
let streamingActive = false;
let streamAudioContext = null;
let streamSourceNode = null;

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

  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) || "";
}

function blobExtension(blobType) {
  if (blobType.includes("mp4")) {
    return "m4a";
  }
  if (blobType.includes("ogg")) {
    return "ogg";
  }
  return "webm";
}

function updateDuration() {
  const seconds = (Date.now() - recordingStartedAt) / 1000;
  clipLength.textContent = `${seconds.toFixed(1)}s`;
}

function startLevelMeter() {
  if (!analyser) {
    return;
  }

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
    if (!response.ok) {
      throw new Error(`상태 확인 실패: ${response.status}`);
    }
    const payload = await response.json();
    setStatus(serverStatus, "서버 준비 완료", "pill pill-ok");
    setMessage(`로드된 모델: ${payload.model}`);
  } catch (error) {
    setStatus(serverStatus, "서버 연결 불가", "pill pill-warn");
    setMessage(`서버에 연결할 수 없습니다: ${error.message}`);
  }
}

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

    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        audioChunks.push(event.data);
      }
    });

    mediaRecorder.addEventListener("stop", async () => {
      await submitRecording();
      cleanupRecorder();
    });

    mediaRecorder.start();
    recordButton.textContent = "녹음 종료";
    recordButton.classList.add("recording");
    setStatus(recorderStatus, "녹음 중", "pill pill-live");
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
  if (!mediaRecorder) {
    return;
  }
  recordButton.disabled = true;
  setMessage("ASR 서버로 오디오를 업로드하는 중입니다...");
  mediaRecorder.stop();
}

function cleanupRecorder() {
  stopLevelMeter();
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  analyser = null;
  mediaRecorder = null;
  recordButton.disabled = false;
  recordButton.textContent = "녹음 시작";
  recordButton.classList.remove("recording");
  setStatus(recorderStatus, "녹음 대기", "pill pill-muted");
}

function closeStreamSocket() {
  if (streamSocket) {
    streamSocket.close();
    streamSocket = null;
  }
}

function cleanupStreaming() {
  if (processorNode) {
    processorNode.disconnect();
    processorNode.onaudioprocess = null;
    processorNode = null;
  }
  if (streamSourceNode) {
    streamSourceNode.disconnect();
    streamSourceNode = null;
  }
  if (audioStream) {
    audioStream.getTracks().forEach((track) => track.stop());
    audioStream = null;
  }
  if (streamAudioContext) {
    streamAudioContext.close();
    streamAudioContext = null;
  }
  streamingActive = false;
  closeStreamSocket();
  streamButton.disabled = false;
  streamButton.textContent = "스트리밍 시작";
  streamButton.classList.remove("recording");
  setStatus(streamStatus, "스트리밍 대기", "pill pill-muted");
  partialMode.textContent = "비활성";
  resetMeter();
}

async function submitRecording() {
  if (!audioChunks.length) {
    setMessage("녹음된 오디오가 없습니다.");
    return;
  }

  const blobType = audioChunks[0].type || "audio/webm";
  const audioBlob = new Blob(audioChunks, { type: blobType });
  const formData = new FormData();
  formData.append("file", audioBlob, `mic-input.${blobExtension(blobType)}`);

  const language = languageSelect.value.trim();
  const endpoint = language ? `/transcribe?language=${encodeURIComponent(language)}` : "/transcribe";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`전사 실패: ${response.status}`);
    }

    const payload = await response.json();
    transcriptBox.value = payload.text || "";
    detectedLanguage.textContent = payload.language || "-";
    inferenceTime.textContent = payload.inference_time ? `${payload.inference_time}s` : "-";
    setMessage("전사가 완료되었습니다.");
  } catch (error) {
    setMessage(error.message);
  }
}

async function startStreaming() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setMessage("이 브라우저는 마이크 녹음을 지원하지 않습니다.");
    return;
  }

  try {
    audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    streamAudioContext = new AudioContext({ sampleRate: 16000 });
    streamSourceNode = streamAudioContext.createMediaStreamSource(audioStream);
    analyser = streamAudioContext.createAnalyser();
    analyser.fftSize = 1024;
    streamSourceNode.connect(analyser);

    processorNode = streamAudioContext.createScriptProcessor(4096, 1, 1);
    streamSourceNode.connect(processorNode);
    processorNode.connect(streamAudioContext.destination);

    const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    streamSocket = new WebSocket(`${wsProtocol}//${window.location.host}/ws/stream`);
    streamSocket.binaryType = "arraybuffer";

    streamSocket.addEventListener("open", () => {
      streamSocket.send(JSON.stringify({
        action: "start",
        language: languageSelect.value.trim(),
        chunk_size_ms: 2000,
      }));
    });

    streamSocket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (payload.event === "ready") {
        streamingActive = true;
        streamButton.textContent = "스트리밍 종료";
        streamButton.classList.add("recording");
        setStatus(streamStatus, "스트리밍 중", "pill pill-live");
        partialMode.textContent = "활성";
        setMessage("실시간 스트리밍 전사를 시작했습니다.");
        recordingStartedAt = Date.now();
        startLevelMeter();
        return;
      }

      if (payload.event === "partial" || payload.event === "final") {
        transcriptBox.value = payload.text || "";
        detectedLanguage.textContent = payload.language || "-";
        partialMode.textContent = payload.is_final ? "종료" : "활성";
        setMessage(payload.is_final ? "스트리밍 전사가 완료되었습니다." : "부분 전사 업데이트 중입니다.");
        if (payload.is_final) {
          cleanupStreaming();
        }
        return;
      }

      if (payload.event === "error") {
        setMessage(`스트리밍 오류: ${payload.message}`);
      }
    });

    streamSocket.addEventListener("close", () => {
      if (streamingActive) {
        cleanupStreaming();
      }
    });

    processorNode.onaudioprocess = (event) => {
      if (!streamSocket || streamSocket.readyState !== WebSocket.OPEN || !streamingActive) {
        return;
      }
      const input = event.inputBuffer.getChannelData(0);
      const pcm = new Float32Array(input.length);
      pcm.set(input);
      streamSocket.send(pcm.buffer);
      updateDuration();
    };
  } catch (error) {
    cleanupStreaming();
    setMessage(`스트리밍 시작 실패: ${error.message}`);
  }
}

function stopStreaming() {
  if (!streamSocket || streamSocket.readyState !== WebSocket.OPEN) {
    cleanupStreaming();
    return;
  }
  streamButton.disabled = true;
  setMessage("남은 버퍼를 정리하고 최종 전사를 마무리하는 중입니다...");
  streamSocket.send(JSON.stringify({ action: "stop" }));
}

recordButton.addEventListener("click", async () => {
  if (streamingActive) {
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
  if (streamingActive) {
    stopStreaming();
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
  cleanupStreaming();
});
ensureServer();

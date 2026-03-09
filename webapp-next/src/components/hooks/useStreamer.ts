'use client';

import { useRef, useCallback, useEffect, type Dispatch } from 'react';
import type { ASRAction } from './useASRState';

const BATCH_INTERVAL_MS = 3000;
const SEGMENT_SEC = 30;
const SAMPLE_RATE = 16000;

/** Audio constraints matching MicVAD defaults for consistent noise handling. */
const AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  channelCount: 1,
  echoCancellation: true,
  autoGainControl: true,
  noiseSuppression: true,
};

// MicVAD type (dynamic import to avoid SSR issues)
type MicVADInstance = {
  start: () => Promise<void>;
  pause: () => Promise<void>;
  destroy: () => Promise<void>;
  listening: boolean;
};

/** Convert Float32 PCM to 16-bit WAV Blob. Uses Int16Array for fast conversion. */
function pcmToWavBlob(float32: Float32Array, sampleRate: number): Blob {
  const numSamples = float32.length;

  // WAV header (44 bytes)
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const writeStr = (off: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + numSamples * 2, true);
  writeStr(8, 'WAVE');
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, 'data');
  view.setUint32(40, numSamples * 2, true);

  // PCM data — typed array avoids per-sample DataView.setInt16 overhead
  const pcm16 = new Int16Array(numSamples);
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }

  return new Blob([header, pcm16.buffer], { type: 'audio/wav' });
}

/** Transcribe a PCM buffer via POST /transcribe. Supports AbortSignal for cancellation. */
async function transcribeBuffer(
  pcm: Float32Array[],
  totalLen: number,
  language: string,
  sampleRate: number,
  signal?: AbortSignal,
): Promise<{ text: string; language: string; inferenceTime: string } | null> {
  if (totalLen < sampleRate) return null;

  const merged = new Float32Array(totalLen);
  let offset = 0;
  for (const f of pcm) {
    merged.set(f, offset);
    offset += f.length;
  }

  const wavBlob = pcmToWavBlob(merged, sampleRate);
  const formData = new FormData();
  formData.append('file', wavBlob, 'stream-batch.wav');

  const lang = language.trim();
  const endpoint = lang
    ? `/transcribe?language=${encodeURIComponent(lang)}`
    : '/transcribe';

  const response = await fetch(endpoint, { method: 'POST', body: formData, signal });
  if (!response.ok) return null;
  const payload = await response.json();

  return {
    text: (payload.text || '').trim(),
    language: payload.language || '',
    inferenceTime: payload.inference_time ? `${payload.inference_time}s` : '',
  };
}

export interface StreamerHandle {
  startStreaming(): Promise<void>;
  stopStreaming(): Promise<void>;
}

export function useStreamer(dispatch: Dispatch<ASRAction>, language: string): StreamerHandle {
  const activeRef = useRef(false);
  const pcmChunksRef = useRef<Float32Array[]>([]);
  const segmentsRef = useRef<string[]>([]);
  const batchInFlightRef = useRef(false);
  const batchPromiseRef = useRef<Promise<void> | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const actualSampleRateRef = useRef(SAMPLE_RATE);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const levelTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef(0);
  const languageRef = useRef(language);
  const vadRef = useRef<MicVADInstance | null>(null);
  const isSpeakingRef = useRef(false);
  const speechEndBatchRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  languageRef.current = language;

  const displayText = useCallback(
    (windowText: string, lang: string, inf: string, mode: string) => {
      const parts = [...segmentsRef.current];
      if (windowText) parts.push(windowText);
      const fullText = parts.join('\n');

      dispatch({
        type: 'STREAM_PARTIAL',
        text: fullText,
        language: lang,
        inferenceTime: inf,
        partialMode: mode,
      });
    },
    [dispatch],
  );

  const doBatch = useCallback(async () => {
    if (batchInFlightRef.current || pcmChunksRef.current.length === 0) return;

    const sr = actualSampleRateRef.current;
    const segmentSamples = SEGMENT_SEC * sr;
    const totalLen = pcmChunksRef.current.reduce((sum, f) => sum + f.length, 0);
    if (totalLen < sr) return;

    batchInFlightRef.current = true;
    abortRef.current = new AbortController();

    // If buffer exceeds segment limit, finalize current segment and start fresh
    if (totalLen > segmentSamples) {
      dispatch({ type: 'STREAM_STATUS', partialMode: '세그먼트 확정 중...' });

      try {
        const bufferSnapshot = [...pcmChunksRef.current];
        pcmChunksRef.current = []; // Clear immediately so new audio goes to new segment

        const result = await transcribeBuffer(bufferSnapshot, totalLen, languageRef.current, sr, abortRef.current.signal);
        if (result?.text && activeRef.current) {
          segmentsRef.current.push(result.text);
          displayText('', result.language, result.inferenceTime, '세그먼트 확정');
        }
      } catch {
        // Aborted or failed — ignore
      } finally {
        batchInFlightRef.current = false;
        abortRef.current = null;
      }
      return;
    }

    // Normal periodic transcription of current window
    dispatch({ type: 'STREAM_STATUS', partialMode: '전사 중...' });

    try {
      const bufferSnapshot = [...pcmChunksRef.current];
      const result = await transcribeBuffer(bufferSnapshot, totalLen, languageRef.current, sr, abortRef.current.signal);
      if (result && activeRef.current) {
        displayText(result.text, result.language, result.inferenceTime, '전사 완료');
      }
    } catch {
      // Aborted or failed — ignore
    } finally {
      batchInFlightRef.current = false;
      abortRef.current = null;
    }
  }, [dispatch, displayText]);

  const cleanup = useCallback(() => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (levelTimerRef.current) { clearInterval(levelTimerRef.current); levelTimerRef.current = null; }
    if (speechEndBatchRef.current) { clearTimeout(speechEndBatchRef.current); speechEndBatchRef.current = null; }
    if (vadRef.current) { vadRef.current.destroy(); vadRef.current = null; }
    if (processorRef.current) { processorRef.current.disconnect(); processorRef.current = null; }
    if (sourceRef.current) { sourceRef.current.disconnect(); sourceRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (ctxRef.current) { ctxRef.current.close(); ctxRef.current = null; }
    analyserRef.current = null;
    activeRef.current = false;
    isSpeakingRef.current = false;
    pcmChunksRef.current = [];
    segmentsRef.current = [];
    batchInFlightRef.current = false;
    batchPromiseRef.current = null;
    if (abortRef.current) { abortRef.current.abort(); abortRef.current = null; }
  }, []);

  useEffect(() => {
    return () => { cleanup(); };
  }, [cleanup]);

  const startStreaming = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) return;

    try {
      // If VAD exists from previous session, resume it instead of creating new
      if (vadRef.current && streamRef.current) {
        activeRef.current = true;
        isSpeakingRef.current = false;
        startTimeRef.current = Date.now();
        pcmChunksRef.current = [];
        segmentsRef.current = [];

        dispatch({ type: 'STREAM_START' });

        await vadRef.current.start();
        console.log('VAD resumed');

        // Restart timers
        const analyser = analyserRef.current;
        if (analyser) {
          const data = new Uint8Array(analyser.frequencyBinCount);
          levelTimerRef.current = setInterval(() => {
            if (!analyserRef.current) return;
            analyserRef.current.getByteTimeDomainData(data);
            let peak = 0;
            for (const sample of data) {
              peak = Math.max(peak, Math.abs(sample - 128));
            }
            const pct = Math.max(1, Math.min(100, Math.round((peak / 128) * 100)));
            dispatch({ type: 'SET_METER', pct, seconds: (Date.now() - startTimeRef.current) / 1000 });
          }, 80);
        }

        timerRef.current = setInterval(() => {
          if (activeRef.current && !batchInFlightRef.current && isSpeakingRef.current) {
            batchPromiseRef.current = doBatch();
          }
        }, BATCH_INTERVAL_MS);

        return;
      }

      // First time: create stream, audio pipeline, and VAD
      const stream = await navigator.mediaDevices.getUserMedia({ audio: AUDIO_CONSTRAINTS });
      streamRef.current = stream;

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      ctxRef.current = ctx;
      actualSampleRateRef.current = ctx.sampleRate;

      if (ctx.sampleRate !== SAMPLE_RATE) {
        console.warn(`AudioContext sampleRate: requested ${SAMPLE_RATE}, got ${ctx.sampleRate}`);
      }

      const source = ctx.createMediaStreamSource(stream);
      sourceRef.current = source;

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = ctx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;
      source.connect(processor);
      processor.connect(ctx.destination);

      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        if (!activeRef.current) return;
        // VAD gate: only accumulate PCM when speech is detected
        if (!isSpeakingRef.current) return;
        const input = e.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(input));
      };

      activeRef.current = true;
      isSpeakingRef.current = false;
      startTimeRef.current = Date.now();
      pcmChunksRef.current = [];
      segmentsRef.current = [];

      dispatch({ type: 'STREAM_START' });

      // Initialize Silero VAD — shared stream, custom pause/resume to keep tracks alive
      try {
        const { MicVAD } = await import('@ricky0123/vad-web');
        const sharedStream = stream;
        const vad = await MicVAD.new({
          baseAssetPath: '/',
          onnxWASMBasePath: '/',
          startOnLoad: false,
          // Share AudioContext and mic stream — single context, single stream
          audioContext: ctx,
          getStream: () => Promise.resolve(sharedStream),
          // Custom pause: don't stop tracks (shared stream stays alive)
          pauseStream: async () => { /* noop — keep stream alive */ },
          // Custom resume: return the same shared stream
          resumeStream: async () => sharedStream,
          onSpeechStart: () => {
            console.log('VAD: speech start');
            isSpeakingRef.current = true;
            if (speechEndBatchRef.current) {
              clearTimeout(speechEndBatchRef.current);
              speechEndBatchRef.current = null;
            }
            if (activeRef.current) {
              dispatch({ type: 'STREAM_STATUS', partialMode: '음성 감지' });
            }
          },
          onSpeechEnd: (audio) => {
            console.log('VAD: speech end', audio.length, 'samples');
            isSpeakingRef.current = false;
            if (activeRef.current) {
              dispatch({ type: 'STREAM_STATUS', partialMode: '대기 중' });
              // Trigger batch after speech ends for responsiveness
              if (!batchInFlightRef.current && pcmChunksRef.current.length > 0) {
                speechEndBatchRef.current = setTimeout(() => {
                  speechEndBatchRef.current = null;
                  if (activeRef.current && !batchInFlightRef.current) {
                    batchPromiseRef.current = doBatch();
                  }
                }, 300);
              }
            }
          },
          onVADMisfire: () => {
            console.log('VAD: misfire');
            isSpeakingRef.current = false;
          },
        });
        await vad.start();
        vadRef.current = vad;
        console.log('VAD initialized (shared stream, default thresholds)');
      } catch (err) {
        console.warn('VAD init failed, falling back to always-on capture:', err);
        isSpeakingRef.current = true;
      }

      const data = new Uint8Array(analyser.frequencyBinCount);
      levelTimerRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(data);
        let peak = 0;
        for (const sample of data) {
          peak = Math.max(peak, Math.abs(sample - 128));
        }
        const pct = Math.max(1, Math.min(100, Math.round((peak / 128) * 100)));
        dispatch({ type: 'SET_METER', pct, seconds: (Date.now() - startTimeRef.current) / 1000 });
      }, 80);

      timerRef.current = setInterval(() => {
        if (activeRef.current && !batchInFlightRef.current && isSpeakingRef.current) {
          batchPromiseRef.current = doBatch();
        }
      }, BATCH_INTERVAL_MS);
    } catch {
      cleanup();
    }
  }, [dispatch, doBatch, cleanup]);

  const stopStreaming = useCallback(async () => {
    // 1. Stop capturing new audio, pause VAD (keep it alive for resume)
    activeRef.current = false;
    isSpeakingRef.current = false;
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
    if (levelTimerRef.current) { clearInterval(levelTimerRef.current); levelTimerRef.current = null; }
    if (speechEndBatchRef.current) { clearTimeout(speechEndBatchRef.current); speechEndBatchRef.current = null; }
    if (vadRef.current) {
      await vadRef.current.pause();
      console.log('VAD paused');
    }

    // 2. Cancel any in-flight batch request
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    // 3. Wait for in-flight batch to finish (it will catch the abort and exit cleanly)
    if (batchPromiseRef.current) {
      await batchPromiseRef.current;
      batchPromiseRef.current = null;
    }
    batchInFlightRef.current = false;

    // 4. Final transcription of remaining buffer (fresh request, no race)
    if (pcmChunksRef.current.length > 0) {
      const sr = actualSampleRateRef.current;
      const totalLen = pcmChunksRef.current.reduce((sum, f) => sum + f.length, 0);
      if (totalLen >= sr) {
        dispatch({ type: 'STREAM_STATUS', partialMode: '최종 전사 중...' });
        try {
          const result = await transcribeBuffer(pcmChunksRef.current, totalLen, languageRef.current, sr);
          if (result?.text) {
            segmentsRef.current.push(result.text);
            displayText('', result.language, result.inferenceTime, '전사 완료');
          }
        } catch {
          // ignore
        }
      }
    }

    pcmChunksRef.current = [];
    batchInFlightRef.current = false;

    dispatch({ type: 'STREAM_STOP' });
  }, [dispatch, displayText]);

  return { startStreaming, stopStreaming };
}

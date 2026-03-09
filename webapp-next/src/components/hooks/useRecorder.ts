'use client';

import { useRef, useEffect, useCallback, type Dispatch } from 'react';
import type { ASRAction } from './useASRState';

export interface RecorderHandle {
  startRecording(): Promise<void>;
  stopRecording(): void;
}

/** Choose the best supported MIME type for MediaRecorder. */
function chooseMimeType(): string {
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/mp4',
    'audio/ogg;codecs=opus',
  ];
  return candidates.find((c) => MediaRecorder.isTypeSupported(c)) ?? '';
}

/** Map blob MIME to file extension. */
function blobExtension(blobType: string): string {
  if (blobType.includes('mp4')) return 'm4a';
  if (blobType.includes('ogg')) return 'ogg';
  return 'webm';
}

export function useRecorder(dispatch: Dispatch<ASRAction>, language: string): RecorderHandle {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const audioStreamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const levelTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startTimeRef = useRef<number>(0);
  const languageRef = useRef(language);

  // Keep language ref in sync so the stop handler uses the latest value.
  languageRef.current = language;

  const cleanup = useCallback(() => {
    if (levelTimerRef.current) {
      clearInterval(levelTimerRef.current);
      levelTimerRef.current = null;
    }
    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect();
      sourceNodeRef.current = null;
    }
    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((t) => t.stop());
      audioStreamRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }
    analyserRef.current = null;
    mediaRecorderRef.current = null;
  }, []);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const startRecording = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      dispatch({ type: 'RECORD_DONE', text: '', language: '', inferenceTime: '' });
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioStreamRef.current = stream;

      const ctx = new AudioContext();
      audioContextRef.current = ctx;
      const source = ctx.createMediaStreamSource(stream);
      sourceNodeRef.current = source;
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mimeType = chooseMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      audioChunksRef.current = [];
      startTimeRef.current = Date.now();

      recorder.addEventListener('dataavailable', (e: BlobEvent) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      });

      recorder.addEventListener('stop', async () => {
        await submitRecording();
        cleanup();
      });

      recorder.start();
      dispatch({ type: 'RECORD_START' });

      // Level meter + duration updates every 80ms.
      const data = new Uint8Array(analyser.frequencyBinCount);
      levelTimerRef.current = setInterval(() => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteTimeDomainData(data);
        let peak = 0;
        for (const sample of data) {
          peak = Math.max(peak, Math.abs(sample - 128));
        }
        const pct = Math.max(1, Math.min(100, Math.round((peak / 128) * 100)));
        dispatch({ type: 'SET_LEVEL', pct });

        const seconds = (Date.now() - startTimeRef.current) / 1000;
        dispatch({ type: 'SET_DURATION', seconds });
      }, 80);
    } catch (error) {
      cleanup();
      dispatch({
        type: 'RECORD_DONE',
        text: '',
        language: '',
        inferenceTime: '',
      });
    }
  }, [dispatch, cleanup]);

  const submitRecording = useCallback(async () => {
    const chunks = audioChunksRef.current;
    if (!chunks.length) {
      dispatch({ type: 'RECORD_DONE', text: '', language: '', inferenceTime: '' });
      return;
    }

    dispatch({ type: 'RECORD_STOP' });

    const blobType = chunks[0].type || 'audio/webm';
    const audioBlob = new Blob(chunks, { type: blobType });
    const formData = new FormData();
    formData.append('file', audioBlob, `mic-input.${blobExtension(blobType)}`);

    const lang = languageRef.current.trim();
    const endpoint = lang
      ? `/transcribe?language=${encodeURIComponent(lang)}`
      : '/transcribe';

    try {
      const response = await fetch(endpoint, { method: 'POST', body: formData });
      if (!response.ok) throw new Error(`Transcription failed: ${response.status}`);
      const payload = await response.json();
      dispatch({
        type: 'RECORD_DONE',
        text: payload.text ?? '',
        language: payload.language ?? '',
        inferenceTime: payload.inference_time ? `${payload.inference_time}s` : '',
      });
    } catch (error) {
      dispatch({
        type: 'RECORD_DONE',
        text: '',
        language: '',
        inferenceTime: '',
      });
    }
  }, [dispatch]);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
    }
  }, []);

  return { startRecording, stopRecording };
}

'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useMicVAD } from '@ricky0123/vad-react';

import { useASRState } from './hooks/useASRState';
import { useStreamSocket } from './hooks/useStreamSocket';
import { useRecorder } from './hooks/useRecorder';
import HeroCard from './HeroCard';
import ControlPanel from './ControlPanel';
import TranscriptPanel from './TranscriptPanel';
import type { PillVariant } from './Pill';

export default function ASRConsole() {
  const [state, dispatch] = useASRState();
  const { openSocket, sendAudio, sendStop, sendVadEvent, closeSocket } =
    useStreamSocket(dispatch);
  const { startRecording, stopRecording } = useRecorder(dispatch, state.language);

  const modeRef = useRef(state.mode);
  modeRef.current = state.mode;
  const startTimeRef = useRef<number>(0);
  const frameCountRef = useRef(0);

  // ── Health check on mount ──
  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then((data) => dispatch({ type: 'SERVER_OK', modelName: data.model ?? '' }))
      .catch(() => dispatch({ type: 'SERVER_ERROR', message: '서버에 연결할 수 없습니다' }));
  }, [dispatch]);

  // ── Cleanup on unmount ──
  useEffect(() => {
    return () => closeSocket();
  }, [closeSocket]);

  // ── VAD — runs continuously, gates audio sending ──
  const vad = useMicVAD({
    model: 'v5',
    startOnLoad: false,
    baseAssetPath: '/',
    onnxWASMBasePath: '/',
    positiveSpeechThreshold: 0.5,
    negativeSpeechThreshold: 0.35,
    minSpeechMs: 300,
    preSpeechPadMs: 300,
    redemptionMs: 500,

    onSpeechStart: () => {
      console.log('[vad] speech START');
      dispatch({ type: 'VAD_SPEECH_START' });
      sendVadEvent('speech_start');
    },

    onSpeechEnd: () => {
      // VAD keeps running — this is just a notification, not a stop signal
      console.log('[vad] speech END (VAD continues listening)');
      dispatch({ type: 'VAD_SPEECH_END' });
      sendVadEvent('speech_end');
    },

    onVADMisfire: () => {
      console.log('[vad] misfire');
      dispatch({ type: 'VAD_MISFIRE' });
    },

    onFrameProcessed: (probs, frame) => {
      const pct = Math.max(1, Math.min(100, Math.round(probs.isSpeech * 100)));
      dispatch({ type: 'SET_LEVEL', pct });

      const mode = modeRef.current;
      const isStreamActive = mode === 'speaking' || mode === 'streaming_ready';

      if (isStreamActive && probs.isSpeech > 0.5 && frame) {
        // CRITICAL: frame.buffer may be larger than the frame slice — extract only the frame's portion
        const audioBytes = new Uint8Array(frame.buffer as ArrayBuffer, frame.byteOffset, frame.byteLength);
        sendAudio(audioBytes.buffer as ArrayBuffer);

        frameCountRef.current++;
        if (frameCountRef.current % 30 === 1) {
          console.log(`[vad] sending frame #${frameCountRef.current}, ${frame.length} samples, speech=${probs.isSpeech.toFixed(2)}`);
        }

        dispatch({
          type: 'SET_DURATION',
          seconds: (Date.now() - (startTimeRef.current || Date.now())) / 1000,
        });
      }
    },
  });

  // ── Stream button: start or stop ──
  const handleStream = useCallback(async () => {
    const mode = modeRef.current;

    if (mode === 'idle') {
      try {
        console.log('[stream] opening socket...');
        await openSocket(state.language);
        console.log('[stream] socket ready, starting VAD...');
        startTimeRef.current = Date.now();
        frameCountRef.current = 0;
        vad.start();
        console.log('[stream] VAD started, listening:', vad.listening);
      } catch (err) {
        console.error('[stream] failed:', err);
        dispatch({
          type: 'STREAM_ERROR',
          message: err instanceof Error ? err.message : 'Connection failed',
        });
      }
    } else {
      console.log('[stream] stopping...');
      if (vad.listening) vad.pause();
      sendStop();
    }
  }, [state.language, openSocket, vad, dispatch, sendStop]);

  // ── Record button ──
  const handleRecord = useCallback(async () => {
    if (state.mode === 'recording') {
      stopRecording();
    } else if (state.mode === 'idle') {
      await startRecording();
    }
  }, [state.mode, startRecording, stopRecording]);

  // ── Derived UI state ──
  const isStreaming = !['idle', 'recording'].includes(state.mode);

  const recordLabel = state.mode === 'recording' ? '녹음 종료' : '녹음 시작';
  const recordDisabled = isStreaming;
  const recordActive = state.mode === 'recording';

  const streamLabel = isStreaming ? '스트리밍 종료' : '스트리밍 시작';
  const streamDisabled = state.mode === 'recording';
  const streamActive = isStreaming;

  const recorderVariant: PillVariant = state.mode === 'recording' ? 'live' : 'muted';
  const recorderStatus = state.mode === 'recording' ? '녹음 중' : '녹음 대기';

  const streamStatusVariant: PillVariant = (() => {
    switch (state.mode) {
      case 'speaking': return 'live';
      case 'streaming_ready': return 'ok';
      case 'finalizing':
      case 'refining': return 'warn';
      default: return 'muted';
    }
  })();

  const streamStatusLabel = (() => {
    switch (state.mode) {
      case 'streaming_ready': return '대기 중 (말씀하세요)';
      case 'speaking': return '음성 감지됨';
      case 'finalizing': return '최종 처리 중';
      case 'refining': return '보정 중';
      default: return '스트리밍 대기';
    }
  })();

  return (
    <>
      <HeroCard
        serverStatus={state.serverStatus}
        modelName={state.modelName}
        recorderStatus={recorderStatus}
        recorderVariant={recorderVariant}
      />

      <section className="panel-grid">
        <ControlPanel
          language={state.language}
          onLanguageChange={(v) => dispatch({ type: 'SET_LANGUAGE', value: v })}
          recordLabel={recordLabel}
          recordDisabled={recordDisabled}
          recordActive={recordActive}
          onRecord={handleRecord}
          streamLabel={streamLabel}
          streamDisabled={streamDisabled}
          streamActive={streamActive}
          onStream={handleStream}
          streamStatusVariant={streamStatusVariant}
          streamStatusLabel={streamStatusLabel}
          levelPct={state.levelPct}
          clipDuration={`${state.clipDuration.toFixed(1)}s`}
          detectedLanguage={state.detectedLanguage || '-'}
          inferenceTime={state.inferenceTime || '-'}
          partialMode={state.partialMode || '비활성'}
          onClear={() => dispatch({ type: 'CLEAR_TRANSCRIPT' })}
        />

        <TranscriptPanel
          value={state.transcript}
          message={state.message || '녹음 시작을 누른 뒤 마이크에 말해 주세요.'}
        />
      </section>
    </>
  );
}

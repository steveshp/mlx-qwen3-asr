'use client';

import { useEffect, useCallback } from 'react';

import { useASRState } from './hooks/useASRState';
import { useRecorder } from './hooks/useRecorder';
import { useStreamer } from './hooks/useStreamer';
import HeroCard from './HeroCard';
import ControlPanel from './ControlPanel';
import TranscriptPanel from './TranscriptPanel';

export default function ASRConsole() {
  const [state, dispatch] = useASRState();
  const { startRecording, stopRecording } = useRecorder(dispatch, state.language);
  const { startStreaming, stopStreaming } = useStreamer(dispatch, state.language);

  // ── Health check on mount ──
  useEffect(() => {
    fetch('/health')
      .then((r) => r.json())
      .then((data) => dispatch({ type: 'SERVER_OK', modelName: data.model ?? '' }))
      .catch(() => dispatch({ type: 'SERVER_ERROR', message: '서버에 연결할 수 없습니다' }));
  }, [dispatch]);

  // ── Record button: start or stop ──
  const handleRecord = useCallback(async () => {
    if (state.mode === 'recording') {
      stopRecording();
    } else if (state.mode === 'idle') {
      await startRecording();
    }
  }, [state.mode, startRecording, stopRecording]);

  // ── Stream button: start or stop ──
  const handleStream = useCallback(async () => {
    if (state.mode === 'streaming') {
      await stopStreaming();
    } else if (state.mode === 'idle') {
      await startStreaming();
    }
  }, [state.mode, startStreaming, stopStreaming]);

  const recordLabel = state.mode === 'recording' ? '녹음 종료' : '녹음 시작';
  const recordDisabled = state.mode === 'transcribing' || state.mode === 'streaming';
  const recordActive = state.mode === 'recording';

  const streamLabel = state.mode === 'streaming' ? '스트리밍 종료' : '스트리밍 시작';
  const streamDisabled = state.mode === 'recording' || state.mode === 'transcribing';
  const streamActive = state.mode === 'streaming';

  const recorderStatusText =
    state.mode === 'recording' ? '녹음 중' :
    state.mode === 'transcribing' ? '전사 중' :
    state.mode === 'streaming' ? '스트리밍 중' :
    '대기';

  const recorderVariant =
    state.mode === 'recording' || state.mode === 'streaming' ? 'live' :
    state.mode === 'transcribing' ? 'warn' :
    'muted';

  return (
    <>
      <HeroCard
        serverStatus={state.serverStatus}
        modelName={state.modelName}
        recorderStatus={recorderStatusText}
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
          streamPartialMode={state.streamPartialMode}
          levelPct={state.levelPct}
          clipDuration={`${state.clipDuration.toFixed(1)}s`}
          detectedLanguage={state.detectedLanguage || '-'}
          inferenceTime={state.inferenceTime || '-'}
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

'use client';

import { useReducer } from 'react';

// ── State types ──

export type Mode =
  | 'idle'
  | 'recording'
  | 'streaming_ready'
  | 'speaking'
  | 'silence_countdown'
  | 'finalizing'
  | 'refining';

export type ServerStatus = 'checking' | 'ok' | 'error';

export interface AppState {
  mode: Mode;
  serverStatus: ServerStatus;
  transcript: string;
  message: string;
  detectedLanguage: string;
  inferenceTime: string;
  clipDuration: number;
  levelPct: number;
  language: string;
  partialMode: string;
  modelName: string;
}

// ── Action types ──

export type ASRAction =
  | { type: 'SERVER_OK'; modelName: string }
  | { type: 'SERVER_ERROR'; message: string }
  | { type: 'RECORD_START' }
  | { type: 'RECORD_STOP' }
  | { type: 'RECORD_DONE'; text: string; language: string; inferenceTime: string }
  | { type: 'STREAM_READY' }
  | { type: 'STREAM_PARTIAL'; text: string; language: string }
  | { type: 'STREAM_FINAL'; text: string; language: string }
  | { type: 'STREAM_REFINED'; text: string; language: string }
  | { type: 'STREAM_ERROR'; message: string }
  | { type: 'STREAM_CLEANUP' }
  | { type: 'VAD_SPEECH_START' }
  | { type: 'VAD_SPEECH_END' }
  | { type: 'VAD_MISFIRE' }
  | { type: 'SET_LEVEL'; pct: number }
  | { type: 'SET_DURATION'; seconds: number }
  | { type: 'SET_LANGUAGE'; value: string }
  | { type: 'CLEAR_TRANSCRIPT' };

// ── Initial state ──

const initialState: AppState = {
  mode: 'idle',
  serverStatus: 'checking',
  transcript: '',
  message: '',
  detectedLanguage: '',
  inferenceTime: '',
  clipDuration: 0,
  levelPct: 0,
  language: '',
  partialMode: '',
  modelName: '',
};

// ── Reducer ──

function asrReducer(state: AppState, action: ASRAction): AppState {
  switch (action.type) {
    case 'SERVER_OK':
      return {
        ...state,
        serverStatus: 'ok',
        modelName: action.modelName,
        message: `Model loaded: ${action.modelName}`,
      };

    case 'SERVER_ERROR':
      return {
        ...state,
        serverStatus: 'error',
        message: action.message,
      };

    case 'RECORD_START':
      return {
        ...state,
        mode: 'recording',
        clipDuration: 0,
        levelPct: 0,
        detectedLanguage: '',
        inferenceTime: '',
        message: 'Recording... press stop when finished.',
      };

    case 'RECORD_STOP':
      return {
        ...state,
        message: 'Uploading audio to ASR server...',
      };

    case 'RECORD_DONE':
      return {
        ...state,
        mode: 'idle',
        transcript: action.text,
        detectedLanguage: action.language,
        inferenceTime: action.inferenceTime,
        levelPct: 0,
        message: 'Transcription complete.',
      };

    case 'STREAM_READY':
      return {
        ...state,
        mode: 'streaming_ready',
        partialMode: 'VAD waiting',
        message: 'Silero VAD active. Speak to begin transcription.',
      };

    case 'STREAM_PARTIAL':
      return {
        ...state,
        transcript: action.text,
        detectedLanguage: action.language,
        partialMode: 'Live transcription',
        message: 'Partial transcription updating...',
      };

    case 'STREAM_FINAL':
      return {
        ...state,
        mode: 'refining',
        transcript: action.text,
        detectedLanguage: action.language,
        partialMode: 'Refining',
        message: 'Streaming complete. Batch refinement in progress...',
      };

    case 'STREAM_REFINED':
      return {
        ...state,
        mode: 'idle',
        transcript: action.text,
        detectedLanguage: action.language,
        partialMode: 'Refined',
        message: 'Batch refinement complete.',
      };

    case 'STREAM_ERROR':
      return {
        ...state,
        message: `Streaming error: ${action.message}`,
      };

    case 'STREAM_CLEANUP':
      return {
        ...state,
        mode: 'idle',
        levelPct: 0,
        partialMode: '',
        message: state.mode === 'idle' ? state.message : 'Streaming stopped.',
      };

    case 'VAD_SPEECH_START':
      return {
        ...state,
        mode: 'speaking',
        message: 'Speech detected.',
      };

    case 'VAD_SPEECH_END':
      return {
        ...state,
        mode: 'streaming_ready',
        message: '무음 감지. 계속 말씀하세요...',
      };

    case 'VAD_MISFIRE':
      return {
        ...state,
        mode: 'streaming_ready',
        message: 'Speech too short, ignoring.',
      };

    case 'SET_LEVEL':
      return { ...state, levelPct: action.pct };

    case 'SET_DURATION':
      return { ...state, clipDuration: action.seconds };

    case 'SET_LANGUAGE':
      return { ...state, language: action.value };

    case 'CLEAR_TRANSCRIPT':
      return {
        ...state,
        transcript: '',
        clipDuration: 0,
        detectedLanguage: '',
        inferenceTime: '',
        message: 'Transcript cleared.',
      };

    default:
      return state;
  }
}

// ── Hook ──

export function useASRState() {
  return useReducer(asrReducer, initialState);
}

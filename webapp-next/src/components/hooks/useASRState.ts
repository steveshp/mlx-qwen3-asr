'use client';

import { useReducer } from 'react';

// ── State types ──

export type Mode = 'idle' | 'recording' | 'transcribing' | 'streaming';

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
  modelName: string;
  streamPartialMode: string;
}

// ── Action types ──

export type ASRAction =
  | { type: 'SERVER_OK'; modelName: string }
  | { type: 'SERVER_ERROR'; message: string }
  | { type: 'RECORD_START' }
  | { type: 'RECORD_STOP' }
  | { type: 'RECORD_DONE'; text: string; language: string; inferenceTime: string }
  | { type: 'SET_LEVEL'; pct: number }
  | { type: 'SET_DURATION'; seconds: number }
  | { type: 'SET_LANGUAGE'; value: string }
  | { type: 'CLEAR_TRANSCRIPT' }
  | { type: 'STREAM_START' }
  | { type: 'STREAM_PARTIAL'; text: string; language: string; inferenceTime: string; partialMode: string }
  | { type: 'STREAM_STATUS'; partialMode: string }
  | { type: 'STREAM_STOP' };

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
  modelName: '',
  streamPartialMode: '',
};

// ── Reducer ──

function asrReducer(state: AppState, action: ASRAction): AppState {
  switch (action.type) {
    case 'SERVER_OK':
      return {
        ...state,
        serverStatus: 'ok',
        modelName: action.modelName,
        message: `모델 로드 완료: ${action.modelName}`,
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
        message: '녹음 중... 완료되면 정지를 누르세요.',
      };

    case 'RECORD_STOP':
      return {
        ...state,
        mode: 'transcribing',
        message: '서버에 음성을 전송 중...',
      };

    case 'RECORD_DONE':
      return {
        ...state,
        mode: 'idle',
        transcript: action.text,
        detectedLanguage: action.language,
        inferenceTime: action.inferenceTime,
        levelPct: 0,
        message: action.text ? '전사 완료.' : '전사 결과가 없습니다.',
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
        streamPartialMode: '',
        message: '초기화 완료.',
      };

    case 'STREAM_START':
      return {
        ...state,
        mode: 'streaming',
        clipDuration: 0,
        levelPct: 0,
        detectedLanguage: '',
        inferenceTime: '',
        streamPartialMode: '캡처 중',
        message: '마이크 캡처 중. 3초마다 자동 전사합니다.',
      };

    case 'STREAM_PARTIAL':
      return {
        ...state,
        transcript: action.text,
        detectedLanguage: action.language,
        inferenceTime: action.inferenceTime,
        streamPartialMode: action.partialMode,
      };

    case 'STREAM_STATUS':
      return { ...state, streamPartialMode: action.partialMode };

    case 'STREAM_STOP':
      return {
        ...state,
        mode: 'idle',
        levelPct: 0,
        streamPartialMode: '',
        message: state.transcript ? '전사 완료.' : '전사 결과가 없습니다.',
      };

    default:
      return state;
  }
}

// ── Hook ──

export function useASRState() {
  return useReducer(asrReducer, initialState);
}

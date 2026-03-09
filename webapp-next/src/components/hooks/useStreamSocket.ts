'use client';

import { useRef, useCallback, type Dispatch, type MutableRefObject } from 'react';
import type { ASRAction } from './useASRState';

export interface StreamSocketHandle {
  socketRef: MutableRefObject<WebSocket | null>;
  openSocket(language: string): Promise<void>;
  sendStart(language: string): void;
  sendAudio(buffer: ArrayBuffer): void;
  sendStop(): void;
  sendVadEvent(type: string): void;
  closeSocket(): void;
}

export function useStreamSocket(dispatch: Dispatch<ASRAction>): StreamSocketHandle {
  const socketRef = useRef<WebSocket | null>(null);

  const closeSocket = useCallback(() => {
    if (socketRef.current) {
      console.log('[ws] closeSocket called');
      socketRef.current.close();
      socketRef.current = null;
    }
  }, []);

  const openSocket = useCallback(
    (language: string): Promise<void> => {
      return new Promise<void>((resolve, reject) => {
        // Next.js rewrites don't proxy WebSocket — connect directly to backend
        const wsHost = window.location.hostname + ':8000';
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const url = `${wsProtocol}//${wsHost}/ws/stream`;
        console.log('[ws] connecting to', url);
        const ws = new WebSocket(url);
        ws.binaryType = 'arraybuffer';

        ws.onopen = () => {
          console.log('[ws] connected, sending start');
          ws.send(
            JSON.stringify({
              action: 'start',
              language: language || undefined,
              chunk_size_ms: 4000,
            }),
          );
        };

        ws.onmessage = (event: MessageEvent) => {
          let payload: { event: string; text?: string; language?: string; message?: string };
          try {
            payload = JSON.parse(event.data as string);
          } catch {
            return;
          }

          console.log('[ws] ←', payload.event, payload.text?.slice(0, 50) ?? '');

          switch (payload.event) {
            case 'ready':
              dispatch({ type: 'STREAM_READY' });
              socketRef.current = ws;
              resolve();
              break;

            case 'partial':
              dispatch({
                type: 'STREAM_PARTIAL',
                text: payload.text ?? '',
                language: payload.language ?? '',
              });
              break;

            case 'final':
              dispatch({
                type: 'STREAM_FINAL',
                text: payload.text ?? '',
                language: payload.language ?? '',
              });
              break;

            case 'refined':
              dispatch({
                type: 'STREAM_REFINED',
                text: payload.text ?? '',
                language: payload.language ?? '',
              });
              break;

            case 'error':
              console.error('[ws] server error:', payload.message);
              dispatch({
                type: 'STREAM_ERROR',
                message: payload.message ?? 'Unknown streaming error',
              });
              break;
          }
        };

        ws.onclose = (ev) => {
          console.log('[ws] closed, code:', ev.code, 'reason:', ev.reason);
          socketRef.current = null;
          dispatch({ type: 'STREAM_CLEANUP' });
        };

        ws.onerror = (ev) => {
          console.error('[ws] error:', ev);
          socketRef.current = null;
          reject(new Error('WebSocket connection failed'));
        };
      });
    },
    [dispatch],
  );

  const sendStart = useCallback((language: string) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log('[ws] → start');
      ws.send(
        JSON.stringify({
          action: 'start',
          language: language || undefined,
          chunk_size_ms: 4000,
        }),
      );
    }
  }, []);

  const sendAudio = useCallback((buffer: ArrayBuffer) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(buffer);
    }
  }, []);

  const sendStop = useCallback(() => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      console.log('[ws] → stop');
      ws.send(JSON.stringify({ action: 'stop' }));
    }
  }, []);

  const sendVadEvent = useCallback((type: string) => {
    const ws = socketRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'vad_event', type }));
    }
  }, []);

  return { socketRef, openSocket, sendStart, sendAudio, sendStop, sendVadEvent, closeSocket };
}

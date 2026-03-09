'use client';

import Pill from './Pill';
import type { PillVariant } from './Pill';

interface HeroCardProps {
  serverStatus: 'checking' | 'ok' | 'error';
  modelName: string;
  recorderStatus: string;
  recorderVariant: PillVariant;
}

const SERVER_PILL_MAP: Record<HeroCardProps['serverStatus'], { variant: PillVariant; label: string }> = {
  checking: { variant: 'muted', label: 'Server checking...' },
  ok:       { variant: 'ok',    label: 'Server connected' },
  error:    { variant: 'warn',  label: 'Server offline' },
};

export default function HeroCard({ serverStatus, modelName, recorderStatus, recorderVariant }: HeroCardProps) {
  const server = SERVER_PILL_MAP[serverStatus];

  return (
    <div className="hero-card">
      <p className="eyebrow">애플 실리콘 로컬 ASR</p>
      <h1>타미온 인공지능 ASR Server</h1>
      <p className="hero-copy">
        Apple Silicon에서 실행되는 로컬 음성 인식 서버입니다.
        Qwen3-ASR {modelName} 모델을 사용하여 실시간 전사를 제공합니다.
      </p>
      <div className="status-row">
        <Pill variant={server.variant}>{server.label}</Pill>
        <Pill variant={recorderVariant}>{recorderStatus}</Pill>
      </div>
    </div>
  );
}

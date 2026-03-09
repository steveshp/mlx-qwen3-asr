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
        브라우저 마이크로 녹음하거나 실시간 스트리밍으로 음성을 전사합니다.
      </p>
      <div className="status-row">
        <Pill variant={server.variant}>{server.label}</Pill>
        <Pill variant={recorderVariant}>{recorderStatus}</Pill>
      </div>
    </div>
  );
}

'use client';

import Pill from './Pill';
import LevelMeter from './LevelMeter';
import MetaCard from './MetaCard';
import type { PillVariant } from './Pill';

interface ControlPanelProps {
  language: string;
  onLanguageChange: (v: string) => void;
  recordLabel: string;
  recordDisabled: boolean;
  recordActive: boolean;
  onRecord: () => void;
  streamLabel: string;
  streamDisabled: boolean;
  streamActive: boolean;
  onStream: () => void;
  streamStatusVariant: PillVariant;
  streamStatusLabel: string;
  levelPct: number;
  clipDuration: string;
  detectedLanguage: string;
  inferenceTime: string;
  partialMode: string;
  onClear: () => void;
}

const LANGUAGES = [
  { value: 'auto', label: '자동감지' },
  { value: 'ko',   label: '한국어' },
  { value: 'en',   label: '영어' },
  { value: 'zh',   label: '중국어' },
  { value: 'ja',   label: '일본어' },
];

export default function ControlPanel({
  language,
  onLanguageChange,
  recordLabel,
  recordDisabled,
  recordActive,
  onRecord,
  streamLabel,
  streamDisabled,
  streamActive,
  onStream,
  streamStatusVariant,
  streamStatusLabel,
  levelPct,
  clipDuration,
  detectedLanguage,
  inferenceTime,
  partialMode,
  onClear,
}: ControlPanelProps) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>음성 입력</h2>
        <p>녹음 또는 스트리밍 모드를 선택하세요</p>
      </div>

      {/* Language selector */}
      <div className="field">
        <span>언어</span>
        <select
          value={language}
          onChange={(e) => onLanguageChange(e.target.value)}
        >
          {LANGUAGES.map((lang) => (
            <option key={lang.value} value={lang.value}>
              {lang.label}
            </option>
          ))}
        </select>
      </div>

      {/* Record + Clear buttons */}
      <div className="button-row">
        <button
          className={`primary-button${recordActive ? ' recording' : ''}`}
          disabled={recordDisabled}
          onClick={onRecord}
        >
          {recordLabel}
        </button>
        <button
          className="ghost-button"
          onClick={onClear}
        >
          지우기
        </button>
      </div>

      {/* Stream button + status */}
      <div className="button-row" style={{ marginTop: 10 }}>
        <button
          className={`primary-button secondary-accent${streamActive ? ' recording' : ''}`}
          disabled={streamDisabled}
          onClick={onStream}
        >
          {streamLabel}
        </button>
        <Pill variant={streamStatusVariant}>{streamStatusLabel}</Pill>
      </div>

      {/* Level meter */}
      <LevelMeter pct={levelPct} />

      {/* Meta cards */}
      <div className="meta-grid">
        <MetaCard label="길이" value={clipDuration} />
        <MetaCard label="언어" value={detectedLanguage} />
        <MetaCard label="추론" value={inferenceTime} />
        <MetaCard label="모드" value={partialMode} />
      </div>
    </div>
  );
}

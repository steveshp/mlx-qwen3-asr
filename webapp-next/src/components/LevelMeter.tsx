'use client';

interface LevelMeterProps {
  pct: number;
}

export default function LevelMeter({ pct }: LevelMeterProps) {
  return (
    <div className="meter-wrap">
      <div
        className="level-meter"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

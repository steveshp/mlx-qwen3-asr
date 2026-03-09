'use client';

interface TranscriptPanelProps {
  value: string;
  message: string;
}

export default function TranscriptPanel({ value, message }: TranscriptPanelProps) {
  return (
    <div className="panel transcript-panel">
      <h2>전사 결과</h2>
      <textarea
        className="transcript-box"
        readOnly
        value={value}
        placeholder="녹음을 마치면 전사 결과가 여기에 표시됩니다."
      />
      <p className="message-line">{message}</p>
    </div>
  );
}

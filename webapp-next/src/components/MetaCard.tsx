'use client';

interface MetaCardProps {
  label: string;
  value: string;
}

export default function MetaCard({ label, value }: MetaCardProps) {
  return (
    <div className="meta-card">
      <span className="meta-label">{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

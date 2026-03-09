'use client';

import type { ReactNode } from 'react';

export type PillVariant = 'muted' | 'ok' | 'live' | 'warn';

interface PillProps {
  variant: PillVariant;
  children: ReactNode;
}

export default function Pill({ variant, children }: PillProps) {
  return (
    <span className={`pill pill-${variant}`}>
      {children}
    </span>
  );
}

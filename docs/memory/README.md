# Memory Protocol

Agent-first memory system for stateless coding sessions.

## Why this exists

1. Session context is transient.
2. We need immutable history and compact retrieval.
3. Git history captures code deltas, not always reusable execution heuristics.

## Two Tracks

1. `events/` (append-only): immutable event stream with concrete implementation
   outcomes.
2. `operating-memory.md` (compacted): small, high-signal retrieval layer used
   during active execution.

## File Layout

1. `events/YYYY-MM.md`: monthly append-only event files.
2. `operating-memory.md`: current compacted memory and distilled learnings.

## Event Rules (append-only track)

1. Never edit old events for content changes; append corrective follow-ups.
2. Use event IDs: `MEM-YYYY-MM-DD-NNN`.
3. Include at minimum:
   - `Decision`
   - `Reuse next time`
   - `Evidence`

## Compaction Rules (operating track)

1. Every compacted item must cite at least one event ID.
2. Keep compacted memory short and actionable.
3. Promote repeated high-value patterns into `Distilled Learnings`.

## Promotion Rule

Promote to distilled learnings when either:

1. A pattern appears at least twice, or
2. It prevented a costly miss/regression.

## Agent Update Flow

1. Append an event in `events/YYYY-MM.md` for non-trivial work.
2. Update `operating-memory.md` only if active guidance changed.
3. If pattern repeats, add or update one distilled learning.

# Operating Memory

Primary memory entry point for agents.

This file is intentionally the front door: it contains protocol and current
compacted memory. Append-only history lives in `events/`.

## Protocol (Stable Rules)

### Two Tracks

1. `events/` is append-only and immutable.
2. `operating-memory.md` is compacted and mutable.

### Event Rules

1. Append events in `events/YYYY-MM.md` with IDs: `MEM-YYYY-MM-DD-NNN`.
2. Never rewrite prior event content; append follow-up corrections.
3. Minimum bar per event: `Decision`, `Reuse next time`, `Evidence`.

### Compaction Rules

1. Every compacted memory item must reference one or more event IDs.
2. Keep compacted guidance short, actionable, and current.
3. Promote repeated high-value patterns into `Distilled Learnings`.

### Promotion Rule

Promote to distilled learnings when:

1. A pattern appears at least twice, or
2. It prevented a costly miss/regression.

### Agent Update Flow

1. For non-trivial and meaningful work, append an event first.
2. Update compacted memory only if active guidance changed.
3. Keep references (`refs`) accurate.

## Current Operating Memory

## Active Defaults

1. For non-trivial and meaningful implementation work, append an immutable
   event first.
   - refs: `MEM-2026-02-16-002`, `MEM-2026-02-16-003`
2. Keep memory notes minimal but actionable:
   - include `Decision`, `Reuse next time`, `Evidence`.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
3. Keep memory guidance lightweight and flexible to preserve contribution
   velocity.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
4. Keep `CLAUDE.md` canonical and keep `AGENTS.md` as a thin compatibility
   delegator.
   - refs: `MEM-2026-02-16-004`

## Distilled Learnings

1. When changing process policy, update both `CLAUDE.md` and operating memory
   in the same commit to avoid drift.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`,
     `MEM-2026-02-16-003`
2. For multi-agent compatibility, prefer one canonical guide plus one thin
   delegator file.
   - refs: `MEM-2026-02-16-004`

## Open Risks

1. Memory updates are currently social-process enforced, not CI-enforced.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

## Revisit Triggers

1. If memory entries grow noisy or stale, tighten compaction and pruning rules.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

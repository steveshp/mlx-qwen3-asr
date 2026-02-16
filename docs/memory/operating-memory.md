# Operating Memory

Compacted retrieval layer for current agent execution.

## Active Defaults

1. For non-trivial implementation work, append an immutable event first.
   - refs: `MEM-2026-02-16-002`
2. Keep memory notes minimal but actionable:
   - include `Decision`, `Reuse next time`, `Evidence`.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`
3. Keep memory guidance lightweight and flexible to preserve contribution
   velocity.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

## Distilled Learnings

1. When changing process policy, update both `CLAUDE.md` and operating memory
   in the same commit to avoid drift.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

## Open Risks

1. Memory updates are currently social-process enforced, not CI-enforced.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

## Revisit Triggers

1. If memory entries grow noisy or stale, tighten compaction and pruning rules.
   - refs: `MEM-2026-02-16-001`, `MEM-2026-02-16-002`

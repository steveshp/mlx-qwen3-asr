# Memory Index

This folder stores agent memory in two tracks:

1. `operating-memory.md`:
   - single front door for protocol + compacted current guidance.
2. `events/`:
   - append-only event history with immutable memory records.

Quick start for agents:

1. Open `operating-memory.md`.
2. If work was non-trivial and meaningful, append to `events/YYYY-MM.md`.
3. Update compacted guidance only when active defaults changed.

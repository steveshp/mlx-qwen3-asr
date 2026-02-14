# Technical Decisions

Key technical decisions made for mlx-qwen3-asr, with rationale.

## Decision 1: Python + MLX

**Choice:** Python with Apple's MLX framework
**Alternatives considered:** Rust, Swift, C++

**Rationale:**
- MLX has no Rust bindings -- the framework is Python and C++ only
- 95% of runtime is spent in Metal/C++ kernels inside MLX -- Python overhead is negligible
- Swift port already exists (qwen3-asr-swift) -- no need to duplicate
- Python ecosystem makes it easy to integrate with HuggingFace, numpy, etc.
- Fastest path to a working implementation

## Decision 2: Standalone Package (not part of mlx-audio)

**Choice:** Independent `mlx-qwen3-asr` package
**Alternative:** Contribute to mlx-audio

**Rationale:**
- mlx-audio has critical bugs for Qwen3-ASR:
  - Uses standard nn.RoPE instead of interleaved MRoPE
  - Issue #459: long-audio truncation
  - Historical config drift risk across revisions
- mlx-audio depends on bleeding-edge packages: `transformers==5.0.0rc3`, `mlx-lm==0.30.5`
- Qwen3-ASR deserves dedicated focus -- it's SOTA and complex enough to warrant its own package
- Standalone allows us to optimize specifically for this model without compromise

## Decision 3: Use Official Forced Aligner as a Temporary Bridge

**Choice:** Enable timestamps via optional `qwen-asr` forced aligner backend
**Alternative:** Keep timestamps disabled until a native MLX aligner is finished

**Rationale:**
- Users get working timestamps now without waiting for a full native aligner
- Keeps core ASR path lean; extra dependency is only needed for timestamps
- Reuses the official reference implementation for alignment quality
- Keeps forward path open to replace backend with native MLX aligner

**Policy (important):**
- This is a transition state, not the end-state architecture.
- Project north star remains full native MLX for core + timestamps.
- `qwen-asr` should be removed from the timestamp path once native MLX aligner
  meets acceptance gates (timing quality + reliability + performance envelope).

## Decision 4: HuggingFace Tokenizer

**Choice:** Use `transformers.AutoTokenizer` (Qwen2TokenizerFast)
**Alternative:** Reimplement BPE tokenizer from scratch

**Rationale:**
- Qwen2TokenizerFast has 151936 tokens -- reimplementing is high effort, low value
- The tokenizer is well-tested and handles all edge cases
- Accept `transformers` as a dependency (needed for tokenizer only, not model inference)
- Tokenization is not a performance bottleneck

## Decision 5: ffmpeg for Audio Loading

**Choice:** Subprocess call to ffmpeg
**Alternative:** librosa, soundfile, torchaudio

**Rationale:**
- Same approach as mlx-whisper -- proven pattern
- Handles all audio formats (mp3, wav, flac, ogg, m4a, etc.)
- No additional Python dependencies (librosa pulls in many)
- ffmpeg is universally available (`brew install ffmpeg`)
- Consistent behavior across audio formats

## Decision 6: On-the-fly Weight Remapping

**Choice:** Remap HF weights at load time (strip `thinker.` prefix, transpose Conv2d)
**Alternative:** Require pre-converted weights

**Rationale:**
- Users can point directly at HuggingFace repo -- no separate conversion step
- Remapping is fast (< 1 second) compared to model download
- Reduces friction for new users
- Still support pre-converted local weights for advanced users

## Decision 7: Interleaved MRoPE (Custom Implementation)

**Choice:** Custom InterleavedMRoPE class instead of MLX's built-in nn.RoPE
**Alternative:** Use nn.RoPE with workarounds

**Rationale:**
- MLX's nn.RoPE doesn't support 3D interleaved frequency assignment
- The interleaving pattern (stride-3 across sections [24,20,20]) is specific to Qwen3
- Incorrect RoPE produces plausible but degraded transcription -- hard to debug
- This is the #1 bug in existing implementations (mlx-audio gets this wrong)
- Correctness is non-negotiable for the core position encoding

## Decision 8: Default Runtime Model = 0.6B

**Choice:** Default `transcribe()` and CLI to `Qwen/Qwen3-ASR-0.6B`
**Alternative:** Keep 1.7B as the default

**Rationale:**
- "One-line and it just works" is stronger with 0.6B on typical Mac hardware
- Lower memory footprint reduces first-run friction and OOM risk
- Better latency by default improves perceived product quality
- 1.7B remains a first-class opt-in for accuracy-focused workloads

## Decision 9: Native JA/KO Aligner Tokenization Mirrors Official Dependencies

**Choice:** For native MLX aligner path, require `nagisa` (JA) and `soynlp` + official
Korean dict (KO), and fail clearly when missing.
**Alternative:** Silent fallback to generic space/CJK tokenization.

**Rationale:**
- Timestamp quality gates require tokenizer behavior to match the official processor.
- Silent fallback hides quality degradation and weakens parity guarantees.
- Clear runtime errors are preferable to silent low-quality alignment in multilingual use.
- Vendoring the official Korean dictionary asset keeps behavior reproducible across machines.

## Decision 10: Prefer Direct `Qwen2Tokenizer` Loader over `AutoTokenizer`

**Choice:** In tokenizer loading, import and instantiate `Qwen2Tokenizer` directly when
available, with `AutoTokenizer` as fallback.
**Alternative:** Always use `AutoTokenizer.from_pretrained(...)`.

**Rationale:**
- `AutoTokenizer` dynamic import path pulls significantly more module graph at cold start.
- Direct `Qwen2Tokenizer` reduces process-level first-transcribe latency while preserving
  tokenization behavior and parity gates.
- Fallback path retains compatibility with older/variant transformer stacks.

## Decision 11: Resolve Local Model Snapshot Path Before Tokenizer Load

**Choice:** Pass resolved local model path (HF snapshot directory) to tokenizer loading
once the model is already resolved by `_ModelHolder`.
**Alternative:** Keep passing the original repo ID string to tokenizer loading.

**Rationale:**
- Repo-ID tokenizer loading may still perform Hub metadata checks in short-lived processes.
- Resolved local path removes that network overhead when weights/tokenizer files are already cached.
- This complements Decision 10 and further reduces first-transcribe latency without affecting quality.

## Decision 12: Streaming Uses Bounded Rolling Context Until True Incremental Decode Lands

**Choice:** Keep streaming as a rolling decode mode with a fixed max context window
(default 30s), explicitly marked experimental.
**Alternative:** Market current streaming as full incremental realtime ASR.

**Rationale:**
- Rolling context keeps per-chunk runtime bounded instead of growing with total session length.
- Prefix-rollback behavior remains useful for partial-output stability.
- This avoids over-claiming while preserving a usable API for live transcription workflows.
- A true production incremental mode still requires decoder cache lifecycle + chunk-level
  audio encoder state strategy; that remains on the roadmap.

## Decision 13: Add Explicit `Session` API While Keeping One-Liner Convenience

**Choice:** Introduce a first-class `Session` object that owns model/tokenizer state,
while preserving top-level `transcribe(...)` for simple usage.
**Alternative:** Keep only hidden process-global holders.

**Rationale:**
- Explicit state ownership is easier to reason about and test.
- Multiple model sessions can coexist in one process without implicit coupling.
- Keeps power-user workflows deterministic while preserving beginner ergonomics.

## Decision 14: Keep Streaming Experimental and Deprioritize Productionization

**Choice:** Keep streaming clearly labeled experimental and avoid major engineering
investment until core offline quality/speed gates are fully saturated.
**Alternative:** Spend near-term roadmap bandwidth on production-grade streaming.

**Rationale:**
- Qwen3-ASR's strongest practical value today is high-quality multilingual
  post-recording transcription.
- Current upstream streaming semantics are themselves constrained
  (vLLM-only and no timestamps), so production streaming is not a short path.
- Near-term engineering ROI is higher in correctness gates, native aligner quality,
  benchmark rigor, quantized packaging, and decode-path cleanliness.

## Decision 15: Adopt Upstream-Style ASR Output Repetition Cleanup

**Choice:** Apply repetition cleanup + edge-case parsing in `parse_asr_output(...)`,
aligned with official Qwen inference utility behavior.
**Alternative:** Keep minimal parser that only splits on `<asr_text>`.

**Rationale:**
- Repetition collapse directly reduces pathological decode tails in real-world audio.
- Handling `language None` and forced-language parse paths improves robustness.
- This is a low-risk, high-value inference-time quality safeguard.

#!/usr/bin/env python3
"""Compare MLX vs qwen_asr quality on a LibriSpeech eval artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent

if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from eval.metrics import compute_cer, compute_wer, normalize_text  # noqa: E402


def _read_audio(path: str) -> np.ndarray:
    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover - dependency message path
        raise RuntimeError(
            "soundfile is required for head-to-head evaluation. "
            "Install with: pip install 'mlx-qwen3-asr[eval]'"
        ) from exc

    audio, _sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32)
    return audio.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLX eval_librispeech artifact outputs against qwen_asr "
            "PyTorch reference on the same samples."
        )
    )
    parser.add_argument(
        "--mlx-json",
        required=True,
        help=(
            "Path to eval_librispeech JSON artifact "
            "(contains rows with audio_path/reference/hypothesis)."
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--language", default="English")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--md-output", default=None)
    args = parser.parse_args()

    try:
        import qwen_asr
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "qwen_asr + torch are required for head-to-head reference evaluation."
        ) from exc

    src = Path(args.mlx_json).expanduser().resolve()
    mlx_payload = json.loads(src.read_text(encoding="utf-8"))
    rows = mlx_payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows in MLX artifact: {src}")

    ref = qwen_asr.Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cpu",
        max_new_tokens=args.max_new_tokens,
    )

    refs: list[str] = []
    mlx_hyps: list[str] = []
    ref_hyps: list[str] = []
    lat_ref: list[float] = []
    out_rows: list[dict] = []

    for i, row in enumerate(rows, start=1):
        reference = str(row.get("reference", ""))
        mlx_hyp = str(row.get("hypothesis", ""))
        audio_path = str(row["audio_path"])

        audio = _read_audio(audio_path)
        t0 = time.perf_counter()
        raw = ref._infer_asr(  # noqa: SLF001
            contexts=[""],
            wavs=[audio],
            languages=[args.language],
        )[0]
        dt = time.perf_counter() - t0
        lat_ref.append(dt)

        _lang, parsed = qwen_asr.parse_asr_output(raw, user_language=args.language)
        ref_hyp = normalize_text(parsed)

        refs.append(reference)
        mlx_hyps.append(mlx_hyp)
        ref_hyps.append(ref_hyp)

        out_rows.append(
            {
                "index": i,
                "sample_id": row.get("sample_id"),
                "speaker_id": row.get("speaker_id"),
                "audio_path": audio_path,
                "reference": reference,
                "mlx_hypothesis": mlx_hyp,
                "pytorch_hypothesis": ref_hyp,
                "mlx_latency_sec": row.get("latency_sec"),
                "pytorch_latency_sec": dt,
            }
        )

    mlx_wer = compute_wer(refs, mlx_hyps)
    mlx_cer = compute_cer(refs, mlx_hyps)
    ref_wer = compute_wer(refs, ref_hyps)
    ref_cer = compute_cer(refs, ref_hyps)

    payload = {
        "suite": "quality-head2head-librispeech-v1",
        "source_mlx_artifact": str(src),
        "model": args.model,
        "language": args.language,
        "samples": len(rows),
        "systems": {
            "mlx": {
                "wer": mlx_wer,
                "cer": mlx_cer,
                "mean_latency_sec": mlx_payload.get("mean_latency_sec"),
                "rtf": mlx_payload.get("rtf"),
            },
            "pytorch_ref": {
                "wer": ref_wer,
                "cer": ref_cer,
                "mean_latency_sec": float(np.mean(lat_ref)) if lat_ref else 0.0,
            },
        },
        "delta": {
            "wer_mlx_minus_ref": mlx_wer - ref_wer,
            "cer_mlx_minus_ref": mlx_cer - ref_cer,
        },
        "rows": out_rows,
    }

    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_output:
        out_json = Path(args.json_output).expanduser().resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.md_output:
        out_md = Path(args.md_output).expanduser().resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        title = (
            "MLX vs PyTorch Quality Head-to-Head "
            f"({mlx_payload.get('subset', 'librispeech')}, n={len(rows)})"
        )
        mlx_latency = float(mlx_payload.get("mean_latency_sec", 0.0))
        ref_latency = float(np.mean(lat_ref))
        lines = [
            f"# {title}",
            "",
            f"- model: `{args.model}`",
            f"- samples: `{len(rows)}`",
            f"- MLX WER: `{mlx_wer:.4f}`",
            f"- PyTorch WER: `{ref_wer:.4f}`",
            f"- Delta WER (MLX-Ref): `{mlx_wer - ref_wer:+.4f}`",
            f"- MLX CER: `{mlx_cer:.4f}`",
            f"- PyTorch CER: `{ref_cer:.4f}`",
            f"- Delta CER (MLX-Ref): `{mlx_cer - ref_cer:+.4f}`",
            "",
            "| System | WER | CER | Mean latency (s) |",
            "|---|---:|---:|---:|",
            f"| MLX | {mlx_wer:.4f} | {mlx_cer:.4f} | {mlx_latency:.4f} |",
            f"| PyTorch ref | {ref_wer:.4f} | {ref_cer:.4f} | {ref_latency:.4f} |",
        ]
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

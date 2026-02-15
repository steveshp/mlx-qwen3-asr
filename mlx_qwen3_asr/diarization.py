"""Native diarization helpers.

This module provides a dependency-light baseline diarization path:
- windowed acoustic embeddings from raw waveform
- cosine clustering
- speaker-turn construction and transcript attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_SPEAKER_LABEL = "SPEAKER_00"


@dataclass(frozen=True)
class DiarizationConfig:
    """Configuration for runtime diarization behavior."""

    num_speakers: Optional[int] = None
    min_speakers: int = 1
    max_speakers: int = 8
    window_sec: float = 1.5
    hop_sec: float = 0.75


@dataclass(frozen=True)
class _WindowEmbedding:
    start: float
    end: float
    embedding: np.ndarray


def validate_diarization_config(
    *,
    num_speakers: Optional[int],
    min_speakers: int,
    max_speakers: int,
    window_sec: float,
    hop_sec: float,
) -> DiarizationConfig:
    """Validate diarization configuration values."""
    if num_speakers is not None and num_speakers < 1:
        raise ValueError("diarization_num_speakers must be >= 1.")
    if min_speakers < 1:
        raise ValueError("diarization_min_speakers must be >= 1.")
    if max_speakers < min_speakers:
        raise ValueError(
            "diarization_max_speakers must be >= diarization_min_speakers."
        )
    if window_sec <= 0.0:
        raise ValueError("diarization_window_sec must be > 0.")
    if hop_sec <= 0.0:
        raise ValueError("diarization_hop_sec must be > 0.")
    return DiarizationConfig(
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )


def infer_speaker_turns(
    audio: np.ndarray,
    *,
    sr: int,
    config: DiarizationConfig,
) -> list[dict]:
    """Infer speaker turns from audio using native windowed clustering."""
    if sr <= 0:
        raise ValueError("sr must be > 0.")
    if audio.size == 0:
        return []

    duration = float(len(audio) / sr)
    windows = _extract_window_embeddings(audio=audio, sr=sr, config=config)
    if not windows:
        return [
            {
                "speaker": DEFAULT_SPEAKER_LABEL,
                "start": 0.0,
                "end": duration,
            }
        ]

    emb = np.stack([w.embedding for w in windows], axis=0)
    labels = _cluster_embeddings(emb, config=config)
    if config.num_speakers is None:
        labels = _maybe_force_two_speaker_split(emb=emb, labels=labels, config=config)
        labels = _maybe_refine_two_cluster_auto_labels(emb=emb, labels=labels, config=config)
        if _should_smooth_auto_labels(labels):
            labels = _smooth_labels(labels, width=3)
    return _windows_to_turns(windows, labels=labels, duration=duration)


def diarize_word_segments(
    segments: list[dict],
    *,
    config: DiarizationConfig,
    speaker_turns: Optional[list[dict]] = None,
) -> tuple[list[dict], list[dict]]:
    """Assign speaker labels to word-level segments."""
    _ = config
    if not segments:
        return [], []

    turns = speaker_turns or []
    labeled: list[dict] = []
    for seg in segments:
        item = dict(seg)
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        item["speaker"] = _speaker_for_interval(start, end, turns)
        labeled.append(item)
    return labeled, _merge_speaker_segments(labeled)


def build_speaker_segments_from_turns(
    *,
    speaker_turns: list[dict],
    word_segments: Optional[list[dict]] = None,
    max_gap_sec: float = 0.2,
) -> list[dict]:
    """Build transcript speaker segments from diarization turns.

    Unlike ``_merge_speaker_segments``, this keeps empty-text turns so output
    time coverage tracks diarization output even when ASR words are sparse.
    """
    if not speaker_turns:
        return []

    turns = sorted(
        (
            {
                "speaker": str(t.get("speaker", DEFAULT_SPEAKER_LABEL)),
                "start": float(t.get("start", 0.0)),
                "end": float(t.get("end", t.get("start", 0.0))),
            }
            for t in speaker_turns
        ),
        key=lambda x: (x["start"], x["end"]),
    )
    words = sorted(
        (dict(w) for w in (word_segments or [])),
        key=lambda x: (float(x.get("start", 0.0)), float(x.get("end", 0.0))),
    )

    out: list[dict] = []
    wi = 0
    for turn in turns:
        start = max(0.0, float(turn["start"]))
        end = max(start, float(turn["end"]))
        speaker = str(turn["speaker"])
        while wi < len(words) and float(words[wi].get("end", 0.0)) <= start:
            wi += 1

        text_parts: list[str] = []
        wj = wi
        while wj < len(words) and float(words[wj].get("start", 0.0)) < end:
            w = words[wj]
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            overlap = max(0.0, min(end, we) - max(start, ws))
            if overlap > 0.0:
                token = str(w.get("text", "")).strip()
                if token:
                    text_parts.append(token)
            wj += 1

        out.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": " ".join(text_parts).strip(),
            }
        )

    merged: list[dict] = []
    for item in out:
        if not merged:
            merged.append(dict(item))
            continue
        prev = merged[-1]
        gap = float(item["start"]) - float(prev["end"])
        if prev["speaker"] == item["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(float(prev["end"]), float(item["end"]))
            prev_text = str(prev.get("text", "")).strip()
            next_text = str(item.get("text", "")).strip()
            if prev_text and next_text:
                prev["text"] = f"{prev_text} {next_text}".strip()
            elif next_text:
                prev["text"] = next_text
            else:
                prev["text"] = prev_text
        else:
            merged.append(dict(item))
    return merged


def diarize_chunk_items(
    chunks: list[dict],
    *,
    config: DiarizationConfig,
    speaker_turns: Optional[list[dict]] = None,
) -> list[dict]:
    """Fallback speaker segments derived from chunk-level transcript items."""
    _ = config
    if not chunks:
        return []
    turns = speaker_turns or []
    items: list[dict] = []
    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        if not text:
            continue
        start = float(chunk.get("start", 0.0))
        end = float(chunk.get("end", start))
        items.append(
            {
                "speaker": _speaker_for_interval(start, end, turns),
                "start": start,
                "end": max(end, start),
                "text": text,
            }
        )
    return _merge_speaker_segments(items)


def _extract_window_embeddings(
    *,
    audio: np.ndarray,
    sr: int,
    config: DiarizationConfig,
) -> list[_WindowEmbedding]:
    win = max(1, int(round(config.window_sec * sr)))
    hop = max(1, int(round(config.hop_sec * sr)))
    n = len(audio)
    if n <= 0:
        return []

    starts = list(range(0, max(1, n - win + 1), hop))
    if not starts:
        starts = [0]
    if starts[-1] + win < n:
        starts.append(max(0, n - win))

    windows_raw: list[tuple[int, int, float, np.ndarray]] = []
    for s in starts:
        e = min(n, s + win)
        seg = np.asarray(audio[s:e], dtype=np.float32)
        if seg.size == 0:
            continue
        energy = float(np.mean(seg * seg))
        windows_raw.append((s, e, energy, seg))

    if not windows_raw:
        return []

    energies = np.array([w[2] for w in windows_raw], dtype=np.float32)
    energy_floor = max(1e-8, float(np.median(energies) * 0.25))
    selected = [w for w in windows_raw if w[2] >= energy_floor]
    if not selected:
        selected = windows_raw

    windows: list[_WindowEmbedding] = []
    for s, e, _en, seg in selected:
        windows.append(
            _WindowEmbedding(
                start=float(s / sr),
                end=float(e / sr),
                embedding=_embed_window(seg, sr=sr),
            )
        )
    return windows


def _embed_window(segment: np.ndarray, *, sr: int) -> np.ndarray:
    x = np.asarray(segment, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return np.zeros((30,), dtype=np.float32)

    frame_len = max(1, int(0.025 * sr))
    frame_hop = max(1, int(0.010 * sr))
    if x.size < frame_len:
        x = np.pad(x, (0, frame_len - x.size))

    frames = []
    for i in range(0, x.size - frame_len + 1, frame_hop):
        frames.append(x[i : i + frame_len])
    if not frames:
        frames = [x[:frame_len]]
    f = np.stack(frames, axis=0)

    win = np.hanning(frame_len).astype(np.float32)
    spec = np.fft.rfft(f * win[None, :], axis=1)
    pwr = (np.abs(spec) ** 2).astype(np.float32) + 1e-8
    freqs = np.fft.rfftfreq(frame_len, d=1.0 / sr).astype(np.float32)

    total = np.sum(pwr, axis=1) + 1e-8
    centroid = np.sum(pwr * freqs[None, :], axis=1) / total
    spread = np.sqrt(
        np.sum(pwr * (freqs[None, :] - centroid[:, None]) ** 2, axis=1) / total
    )

    csum = np.cumsum(pwr, axis=1)
    roll_idx = np.argmax(csum >= (0.85 * total[:, None]), axis=1)
    rolloff = freqs[roll_idx]
    peak_idx = np.argmax(pwr, axis=1)
    peak_freq = freqs[peak_idx]

    zcr = np.mean(np.abs(np.diff(np.signbit(f), axis=1)), axis=1).astype(np.float32)

    band_edges = np.array(
        [0, 200, 400, 800, 1200, 1800, 2600, 3600, 5000, 7600, max(7600, sr / 2)],
        dtype=np.float32,
    )
    band_feats = []
    for b0, b1 in zip(band_edges[:-1], band_edges[1:]):
        idx = np.where((freqs >= b0) & (freqs < b1))[0]
        if idx.size == 0:
            band = np.zeros((pwr.shape[0],), dtype=np.float32)
        else:
            band = np.log1p(np.sum(pwr[:, idx], axis=1))
        band_feats.append(float(np.mean(band)))
    for b0, b1 in zip(band_edges[:-1], band_edges[1:]):
        idx = np.where((freqs >= b0) & (freqs < b1))[0]
        if idx.size == 0:
            band = np.zeros((pwr.shape[0],), dtype=np.float32)
        else:
            band = np.log1p(np.sum(pwr[:, idx], axis=1))
        band_feats.append(float(np.std(band)))

    feats = np.array(
        [
            float(np.mean(centroid)),
            float(np.std(centroid)),
            float(np.mean(spread)),
            float(np.std(spread)),
            float(np.mean(rolloff)),
            float(np.std(rolloff)),
            float(np.mean(zcr)),
            float(np.std(zcr)),
            float(np.mean(peak_freq)),
            float(np.std(peak_freq)),
            *band_feats,
        ],
        dtype=np.float32,
    )
    feats -= np.mean(feats)
    norm = float(np.linalg.norm(feats))
    if norm <= 1e-8:
        return feats
    return feats / norm


def _cluster_embeddings(emb: np.ndarray, *, config: DiarizationConfig) -> np.ndarray:
    n = emb.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.int32)
    if n == 1:
        return np.zeros((1,), dtype=np.int32)

    if config.num_speakers is not None:
        k = max(1, min(int(config.num_speakers), n))
        return _remap_labels_by_first_appearance(_kmeans_cosine(emb, k=k))

    threshold = _auto_distance_threshold(emb)
    labels = _online_threshold_cluster(
        emb=emb,
        threshold=threshold,
        max_clusters=max(1, min(config.max_speakers, n)),
    )
    k = len(set(int(x) for x in labels.tolist()))
    min_k = max(1, min(config.min_speakers, n))
    max_k = max(1, min(config.max_speakers, n))

    if k < min_k:
        labels = _kmeans_cosine(emb, k=min_k)
    elif k > max_k:
        labels = _kmeans_cosine(emb, k=max_k)
    return _remap_labels_by_first_appearance(labels)


def _auto_distance_threshold(emb: np.ndarray) -> float:
    n = emb.shape[0]
    if n < 2:
        return 0.22
    if n > 256:
        idx = np.linspace(0, n - 1, 256, dtype=np.int32)
        x = emb[idx]
    else:
        x = emb
    sim = np.clip(x @ x.T, -1.0, 1.0)
    dist = 1.0 - sim
    tri = dist[np.triu_indices(dist.shape[0], k=1)]
    if tri.size == 0:
        return 0.22
    p25 = float(np.percentile(tri, 25))
    return float(min(0.40, max(0.08, p25)))


def _online_threshold_cluster(
    *,
    emb: np.ndarray,
    threshold: float,
    max_clusters: int,
) -> np.ndarray:
    labels = np.zeros((emb.shape[0],), dtype=np.int32)
    centroids: list[np.ndarray] = []
    counts: list[int] = []

    for i, e in enumerate(emb):
        if not centroids:
            centroids.append(e.copy())
            counts.append(1)
            labels[i] = 0
            continue

        sims = np.array([float(np.dot(e, c)) for c in centroids], dtype=np.float32)
        dists = 1.0 - sims
        best = int(np.argmin(dists))
        if dists[best] <= threshold or len(centroids) >= max_clusters:
            labels[i] = best
            counts[best] += 1
            eta = 1.0 / float(counts[best])
            updated = (1.0 - eta) * centroids[best] + eta * e
            norm = float(np.linalg.norm(updated))
            if norm > 1e-8:
                updated = updated / norm
            centroids[best] = updated.astype(np.float32)
        else:
            new_idx = len(centroids)
            centroids.append(e.copy())
            counts.append(1)
            labels[i] = new_idx
    return labels


def _kmeans_cosine(emb: np.ndarray, *, k: int, iters: int = 20) -> np.ndarray:
    n = emb.shape[0]
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros((n,), dtype=np.int32)

    centroids = emb[np.linspace(0, n - 1, k, dtype=np.int32)].copy()
    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(max(1, iters)):
        sims = emb @ centroids.T
        new_labels = np.argmax(sims, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for ci in range(k):
            members = emb[labels == ci]
            if members.size == 0:
                continue
            c = np.mean(members, axis=0)
            norm = float(np.linalg.norm(c))
            if norm > 1e-8:
                c = c / norm
            centroids[ci] = c.astype(np.float32)
    return labels


def _smooth_labels(labels: np.ndarray, *, width: int) -> np.ndarray:
    if labels.size == 0 or width <= 1:
        return labels
    radius = max(1, width // 2)
    out = labels.copy()
    for i in range(labels.size):
        lo = max(0, i - radius)
        hi = min(labels.size, i + radius + 1)
        win = labels[lo:hi]
        vals, counts = np.unique(win, return_counts=True)
        max_count = int(np.max(counts))
        tied = vals[counts == max_count]
        current = int(labels[i])
        if current in tied.tolist():
            out[i] = current
        else:
            out[i] = int(tied[0])
    return out


def _should_smooth_auto_labels(labels: np.ndarray) -> bool:
    """Return whether smoothing is likely to help more than hurt.

    We skip smoothing when transitions are already sparse or when a cluster
    is tiny, because a 3-frame majority filter can erase valid short turns.
    """
    if labels.size < 6:
        return False
    vals, counts = np.unique(labels, return_counts=True)
    if vals.size <= 1:
        return False
    if int(np.min(counts)) < 2:
        return False
    transitions = int(np.sum(labels[1:] != labels[:-1]))
    if transitions <= 1:
        return False
    return True


def _maybe_force_two_speaker_split(
    *,
    emb: np.ndarray,
    labels: np.ndarray,
    config: DiarizationConfig,
) -> np.ndarray:
    """Conservative anti-collapse heuristic for auto speaker-count mode."""
    if emb.shape[0] < 4:
        return labels
    if config.max_speakers < 2:
        return labels
    unique = {int(x) for x in labels.tolist()}
    if len(unique) != 1:
        return labels

    labels2 = _kmeans_cosine(emb, k=2)
    counts = np.bincount(labels2, minlength=2)
    if int(np.min(counts)) < 2:
        return labels

    minority_ratio = float(np.min(counts)) / float(np.sum(counts))
    if minority_ratio < 0.18:
        return labels
    transitions = int(np.sum(labels2[1:] != labels2[:-1]))
    if transitions > 3:
        return labels
    if _first_last_centroid_distance(emb) < 0.04:
        return labels

    d1 = _mean_cosine_distance_to_centroids(emb, labels)
    d2 = _mean_cosine_distance_to_centroids(emb, labels2)
    abs_gain = d1 - d2
    if abs_gain >= 0.004:
        return _remap_labels_by_first_appearance(labels2)
    return labels


def _has_multispeaker_transition_evidence(emb: np.ndarray) -> bool:
    if emb.shape[0] < 3:
        return False
    step_sim = np.clip(np.sum(emb[1:] * emb[:-1], axis=1), -1.0, 1.0)
    step_dist = 1.0 - step_sim
    p90 = float(np.percentile(step_dist, 90))
    p50 = float(np.percentile(step_dist, 50))
    peak = float(np.max(step_dist))
    return bool(peak >= 0.30 and p90 >= 0.20 and p90 >= (p50 + 0.08))


def _first_last_centroid_distance(emb: np.ndarray) -> float:
    if emb.shape[0] < 2:
        return 0.0
    q = max(1, emb.shape[0] // 4)
    first = np.mean(emb[:q], axis=0)
    last = np.mean(emb[-q:], axis=0)
    n1 = float(np.linalg.norm(first))
    n2 = float(np.linalg.norm(last))
    if n1 > 1e-8:
        first = first / n1
    if n2 > 1e-8:
        last = last / n2
    return float(1.0 - np.clip(np.dot(first, last), -1.0, 1.0))


def _mean_cosine_distance_to_centroids(emb: np.ndarray, labels: np.ndarray) -> float:
    if emb.shape[0] == 0:
        return 0.0
    distances = np.zeros((emb.shape[0],), dtype=np.float32)
    for cluster_id in np.unique(labels):
        members = emb[labels == cluster_id]
        if members.size == 0:
            continue
        centroid = np.mean(members, axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm > 1e-8:
            centroid = centroid / norm
        idx = np.where(labels == cluster_id)[0]
        sims = np.clip(emb[idx] @ centroid, -1.0, 1.0)
        distances[idx] = 1.0 - sims
    return float(np.mean(distances))


def _maybe_refine_two_cluster_auto_labels(
    *,
    emb: np.ndarray,
    labels: np.ndarray,
    config: DiarizationConfig,
) -> np.ndarray:
    """Replace unstable 2-cluster auto labels with a cleaner k-means split.

    The online threshold pass can create a tiny middle cluster and flip back to
    the original speaker (A-B-A) even when a monotonic A-A-B-B pattern is more
    plausible. This refinement is only applied in auto mode with <=2 clusters.
    """
    if config.max_speakers < 2 or emb.shape[0] < 4:
        return labels

    unique = np.unique(labels)
    if unique.size != 2:
        return labels
    transitions = int(np.sum(labels[1:] != labels[:-1]))
    if transitions <= 1:
        return labels

    counts = np.bincount(labels.astype(np.int32), minlength=2)
    minority_ratio = float(np.min(counts)) / float(np.sum(counts))
    if minority_ratio >= 0.35 and transitions <= 2:
        return labels

    cand = _remap_labels_by_first_appearance(_kmeans_cosine(emb, k=2))
    cand_counts = np.bincount(cand.astype(np.int32), minlength=2)
    if int(np.min(cand_counts)) < 2:
        return labels

    base_d = _mean_cosine_distance_to_centroids(emb, labels)
    cand_d = _mean_cosine_distance_to_centroids(emb, cand)
    base_t = transitions
    cand_t = int(np.sum(cand[1:] != cand[:-1]))
    if cand_t <= base_t and cand_d <= (base_d + 1e-6):
        return cand
    return labels


def _windows_to_turns(
    windows: list[_WindowEmbedding],
    *,
    labels: np.ndarray,
    duration: float,
) -> list[dict]:
    if not windows:
        return []
    turns: list[dict] = []
    for w, label in zip(windows, labels):
        start = float(w.start)
        end = float(w.end)
        if not turns:
            turns.append({"label": int(label), "start": start, "end": end})
            continue
        prev = turns[-1]
        if prev["label"] == int(label):
            prev["end"] = max(float(prev["end"]), end)
            continue
        if start < float(prev["end"]):
            boundary = 0.5 * (start + float(prev["end"]))
            prev["end"] = boundary
            start = boundary
        turns.append({"label": int(label), "start": start, "end": end})

    label_order: dict[int, int] = {}
    ordered_labels = [int(t["label"]) for t in turns]
    for label in ordered_labels:
        if label not in label_order:
            label_order[label] = len(label_order)

    final: list[dict] = []
    for t in turns:
        start = float(max(0.0, t["start"]))
        end = float(min(duration, max(start, t["end"])))
        idx = label_order[int(t["label"])]
        final.append(
            {
                "speaker": f"SPEAKER_{idx:02d}",
                "start": start,
                "end": end,
            }
        )
    return _merge_speaker_turns(final)


def _merge_speaker_turns(turns: list[dict], *, max_gap_sec: float = 0.2) -> list[dict]:
    if not turns:
        return []
    merged: list[dict] = []
    for t in turns:
        if not merged:
            merged.append(dict(t))
            continue
        prev = merged[-1]
        if (
            prev["speaker"] == t["speaker"]
            and float(t["start"]) - float(prev["end"]) <= max_gap_sec
        ):
            prev["end"] = max(float(prev["end"]), float(t["end"]))
        else:
            merged.append(dict(t))
    return merged


def _speaker_for_interval(start: float, end: float, turns: list[dict]) -> str:
    if not turns:
        return DEFAULT_SPEAKER_LABEL
    start = float(start)
    end = float(max(end, start))
    best_speaker = str(turns[0].get("speaker", DEFAULT_SPEAKER_LABEL))
    best_overlap = -1.0
    for t in turns:
        ts = float(t.get("start", 0.0))
        te = float(t.get("end", ts))
        overlap = max(0.0, min(end, te) - max(start, ts))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = str(t.get("speaker", DEFAULT_SPEAKER_LABEL))
    if best_overlap > 0.0:
        return best_speaker

    mid = 0.5 * (start + end)
    best_dist = float("inf")
    for t in turns:
        ts = float(t.get("start", 0.0))
        te = float(t.get("end", ts))
        tmid = 0.5 * (ts + te)
        dist = abs(mid - tmid)
        if dist < best_dist:
            best_dist = dist
            best_speaker = str(t.get("speaker", DEFAULT_SPEAKER_LABEL))
    return best_speaker


def _remap_labels_by_first_appearance(labels: np.ndarray) -> np.ndarray:
    out = labels.copy().astype(np.int32)
    mapping: dict[int, int] = {}
    next_idx = 0
    for i, val in enumerate(out.tolist()):
        iv = int(val)
        if iv not in mapping:
            mapping[iv] = next_idx
            next_idx += 1
        out[i] = mapping[iv]
    return out


def _merge_speaker_segments(items: list[dict], *, max_gap_sec: float = 0.8) -> list[dict]:
    """Merge adjacent same-speaker items into longer contiguous turns."""
    if not items:
        return []

    merged: list[dict] = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        speaker = str(item.get("speaker", DEFAULT_SPEAKER_LABEL))
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        end = max(end, start)

        if not merged:
            merged.append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )
            continue

        prev = merged[-1]
        gap = start - float(prev["end"])
        if speaker == prev["speaker"] and gap <= max_gap_sec:
            prev["end"] = max(float(prev["end"]), end)
            prev["text"] = f"{prev['text']} {text}".strip()
        else:
            merged.append(
                {"speaker": speaker, "start": start, "end": end, "text": text}
            )
    return merged

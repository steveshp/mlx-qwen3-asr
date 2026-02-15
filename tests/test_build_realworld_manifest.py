from scripts.build_realworld_manifest import Candidate, _sample_speaker_round_robin


def _candidate(speaker: str, sample_id: str) -> Candidate:
    return Candidate(
        source="test",
        sample_id=sample_id,
        speaker_id=speaker,
        reference_text="hello world",
        duration_sec=2.0,
        audio_bytes=b"fake",
        subset="subset",
        language="English",
    )


def test_sample_speaker_round_robin_spreads_speakers() -> None:
    candidates = [
        _candidate("s1", "a1"),
        _candidate("s1", "a2"),
        _candidate("s2", "b1"),
        _candidate("s2", "b2"),
        _candidate("s3", "c1"),
        _candidate("s3", "c2"),
    ]

    picked = _sample_speaker_round_robin(candidates, samples=5, seed=7)

    assert len(picked) == 5
    # Round-robin should include all speakers before exhausting one speaker.
    assert {x.speaker_id for x in picked[:3]} == {"s1", "s2", "s3"}


def test_sample_speaker_round_robin_is_deterministic() -> None:
    candidates = [
        _candidate("s1", "a1"),
        _candidate("s1", "a2"),
        _candidate("s2", "b1"),
        _candidate("s2", "b2"),
        _candidate("s3", "c1"),
        _candidate("s3", "c2"),
    ]
    picked_a = _sample_speaker_round_robin(candidates, samples=6, seed=20260215)
    picked_b = _sample_speaker_round_robin(candidates, samples=6, seed=20260215)
    picked_c = _sample_speaker_round_robin(candidates, samples=6, seed=20260216)

    ids_a = [x.sample_id for x in picked_a]
    ids_b = [x.sample_id for x in picked_b]
    ids_c = [x.sample_id for x in picked_c]

    assert ids_a == ids_b
    assert ids_a != ids_c

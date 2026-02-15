from scripts.build_earnings22_longform_manifest import Candidate, _sample_family_round_robin


def _candidate(file_id: str, family: str) -> Candidate:
    return Candidate(
        file_id=file_id,
        language_family=family,
        country="X",
        duration_sec=900.0,
        transcription="hello",
        audio_bytes=b"abc",
        audio_name="a.mp3",
    )


def test_sample_family_round_robin_spreads_families() -> None:
    candidates = [
        _candidate("a1", "A"),
        _candidate("a2", "A"),
        _candidate("b1", "B"),
        _candidate("b2", "B"),
        _candidate("c1", "C"),
    ]
    selected = _sample_family_round_robin(candidates, samples=4, seed=3)
    assert len(selected) == 4
    # First cycle should include all families before repeats.
    assert {x.language_family for x in selected[:3]} == {"A", "B", "C"}


def test_sample_family_round_robin_is_deterministic() -> None:
    candidates = [
        _candidate("a1", "A"),
        _candidate("a2", "A"),
        _candidate("b1", "B"),
        _candidate("b2", "B"),
        _candidate("c1", "C"),
    ]
    a = _sample_family_round_robin(candidates, samples=5, seed=20260216)
    b = _sample_family_round_robin(candidates, samples=5, seed=20260216)
    c = _sample_family_round_robin(candidates, samples=5, seed=20260217)
    assert [x.file_id for x in a] == [x.file_id for x in b]
    assert [x.file_id for x in a] != [x.file_id for x in c]

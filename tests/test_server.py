"""Tests for the FastAPI ASR server web UI."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

import server


class DummyResult:
    text = "ni hao"
    language = "Chinese"


class DummySession:
    def transcribe(self, path: str, language: str | None = None) -> DummyResult:
        assert Path(path).exists()
        assert language in (None, "Chinese")
        return DummyResult()

    def init_streaming(
        self,
        *,
        language: str | None = None,
        sample_rate: int = 16000,
        chunk_size_sec: float = 2.0,
    ):
        assert language in (None, "Chinese")
        assert sample_rate == 16000
        return type(
            "DummyStreamingState",
            (),
            {"text": "", "stable_text": "", "language": language or "unknown", "chunk_size_sec": chunk_size_sec},
        )()

    def feed_audio(self, pcm, state):
        assert len(pcm) > 0
        state.text = "partial text"
        state.stable_text = "partial"
        if state.language == "unknown":
            state.language = "Chinese"
        return state

    def finish_streaming(self, state):
        state.text = "final text"
        state.stable_text = "final text"
        if state.language == "unknown":
            state.language = "Chinese"
        return state


def test_index_serves_webapp():
    client = TestClient(server.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "타미온 인공지능 ASR Server" in response.text
    assert "스트리밍 시작" in response.text
    assert "/static/app.js" in response.text


def test_health_returns_model_path():
    client = TestClient(server.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model"] == server.MODEL_PATH


def test_transcribe_accepts_audio_upload(monkeypatch):
    monkeypatch.setattr(server, "session", DummySession())
    client = TestClient(server.app)

    response = client.post(
        "/transcribe?language=Chinese",
        files={"file": ("mic-input.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json()["text"] == "ni hao"
    assert response.json()["language"] == "Chinese"


def test_streaming_websocket_emits_partial_and_final(monkeypatch):
    monkeypatch.setattr(server, "session", DummySession())
    client = TestClient(server.app)

    with client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json({"action": "start", "language": "Chinese", "chunk_size_ms": 1000})
        ready = websocket.receive_json()
        assert ready["event"] == "ready"
        assert ready["sample_rate"] == 16000

        websocket.send_bytes(b"\x00\x00\x00\x00" * 128)
        partial = websocket.receive_json()
        assert partial["event"] == "partial"
        assert partial["text"] == "partial text"
        assert partial["is_final"] is False

        websocket.send_json({"action": "stop"})
        final = websocket.receive_json()
        assert final["event"] == "final"
        assert final["text"] == "final text"
        assert final["is_final"] is True

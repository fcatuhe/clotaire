"""Tests for step 04 diarization helpers."""

from __future__ import annotations

from clotaire.step_04_diarize import _build_speakers, _build_step, _build_turns


class _FakeTurn:
    def __init__(self, start: float, end: float) -> None:
        self.start = start
        self.end = end


def test_build_turns_normalizes_pyannote_tracks() -> None:
    turns = _build_turns(
        [
            (_FakeTurn(0.267, 1.583), None, "SPEAKER_01"),
            (_FakeTurn(2.410, 6.342), None, "SPEAKER_00"),
        ]
    )

    assert turns == [
        {
            "id": "turn_0001",
            "start_ms": 267,
            "end_ms": 1583,
            "speaker": "SPEAKER_01",
            "duration_ms": 1316,
        },
        {
            "id": "turn_0002",
            "start_ms": 2410,
            "end_ms": 6342,
            "speaker": "SPEAKER_00",
            "duration_ms": 3932,
        },
    ]


def test_build_speakers_summarizes_turn_counts_and_duration() -> None:
    speakers = _build_speakers(
        [
            {"speaker": "SPEAKER_01", "duration_ms": 1316},
            {"speaker": "SPEAKER_00", "duration_ms": 3932},
            {"speaker": "SPEAKER_01", "duration_ms": 911},
        ]
    )

    assert speakers == [
        {"id": "SPEAKER_00", "num_turns": 1, "duration_s": 3.932},
        {"id": "SPEAKER_01", "num_turns": 2, "duration_s": 2.227},
    ]


def test_build_step_matches_schema() -> None:
    turns = [
        {
            "id": "turn_0001",
            "start_ms": 267,
            "end_ms": 1583,
            "speaker": "SPEAKER_01",
            "duration_ms": 1316,
        },
        {
            "id": "turn_0002",
            "start_ms": 2410,
            "end_ms": 6342,
            "speaker": "SPEAKER_00",
            "duration_ms": 3932,
        },
    ]
    turns_exclusive = [
        *turns,
        {
            "id": "turn_0003",
            "start_ms": 2500,
            "end_ms": 2600,
            "speaker": "SPEAKER_01",
            "duration_ms": 100,
        },
    ]

    step = _build_step(turns, turns_exclusive, wall_time_s=3.424)

    assert step["step"] == "04_diarize"
    assert "input" not in step
    assert step["model"]["name"] == "pyannote/speaker-diarization-community-1"
    assert step["config"] == {"min_speakers": None, "max_speakers": None}
    assert step["result"]["num_speakers"] == 2
    assert step["result"]["num_turns"] == 2
    assert step["result"]["num_turns_exclusive"] == 3
    assert step["result"]["turns"] == turns
    assert step["result"]["turns_exclusive"] == turns_exclusive
    assert step["timing"] == {"wall_s": 3.42}

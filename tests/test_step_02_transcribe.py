"""Tests for step 02 transcription helpers."""

from __future__ import annotations

import json

from clotaire.step_02_transcribe import (
    _build_lines,
    _build_transcription,
    _build_vad,
    _compact_voice_ranges_in_json,
    _ms_to_timestamp,
    _parse_language,
    _parse_timings,
    _parse_voice_ranges,
)


def test_parse_voice_ranges_and_timings_and_language() -> None:
    stderr = "\n".join(
        [
            "VAD segment 0: start = 0.123, end = 1.234",
            "VAD segment 1: start = 2.000, end = 2.500",
            "whisper_print_timings:     sample time = 12.345 ms",
            "whisper_print_timings:     encode time = 200.000 ms",
            "auto-detected language: fr (p = 0.9876)",
        ]
    )

    assert _parse_voice_ranges(stderr) == [
        {"start_ms": 123, "end_ms": 1234},
        {"start_ms": 2000, "end_ms": 2500},
    ]
    assert _parse_timings(stderr) == {"sample": 0.012, "encode": 0.2}
    assert _parse_language(stderr) == ("fr", 0.9876)


def test_build_transcription_filters_special_tokens() -> None:
    whisper_json = {
        "transcription": [
            {
                "text": " Bonjour",
                "offsets": {"from": 100, "to": 800},
                "tokens": [
                    {"text": "[_BEG_]", "offsets": {"from": 0, "to": 0}, "p": 1.0},
                    {"text": "Bonjour", "offsets": {"from": 100, "to": 800}, "p": 0.87654},
                ],
            }
        ]
    }

    segments = _build_transcription(whisper_json)

    assert segments == [
        {
            "start_ms": 100,
            "end_ms": 800,
            "text": "Bonjour",
            "tokens": [
                {"text": "Bonjour", "start_ms": 100, "end_ms": 800, "p": 0.8765}
            ],
        }
    ]


def test_build_vad_and_lines() -> None:
    voice_ranges = [{"start_ms": 100, "end_ms": 600}, {"start_ms": 1000, "end_ms": 1600}]
    vad = _build_vad(voice_ranges, audio_duration_ms=2000)

    assert vad["num_voice_ranges"] == 2
    assert vad["speech_duration_s"] == 1.1
    assert vad["audio_duration_s"] == 2.0
    assert vad["reduction_pct"] == 45.0

    lines = _build_lines([{"start_ms": 100, "end_ms": 1600, "text": "Bonjour"}])
    assert lines == ["[00:00:00.100 --> 00:00:01.600]   Bonjour"]
    assert _ms_to_timestamp(3723004) == "01:02:03.004"


def test_compact_voice_ranges_in_json() -> None:
    text = json.dumps(
        {
            "vad": {
                "voice_ranges": [
                    {"start_ms": 100, "end_ms": 600},
                    {"start_ms": 1000, "end_ms": 1600},
                ]
            }
        },
        indent=2,
    )

    compacted = _compact_voice_ranges_in_json(text)

    assert '{ "start_ms": 100, "end_ms": 600 }' in compacted
    assert '{ "start_ms": 1000, "end_ms": 1600 }' in compacted

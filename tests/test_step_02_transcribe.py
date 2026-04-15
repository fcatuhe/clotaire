"""Tests for step 02 transcription helpers."""

from __future__ import annotations

import json

from clotaire.step_02_transcribe import (
    _assign_voice_range_id,
    _build_items,
    _build_lines,
    _build_transcription,
    _build_vad,
    _build_voice_ranges,
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

    voice_ranges = _build_voice_ranges([{"start_ms": 0, "end_ms": 1000}])
    segments = _build_transcription(whisper_json, voice_ranges)

    assert segments == [
        {
            "id": "seg_0001",
            "voice_range_id": "vr_0001",
            "text": "Bonjour",
            "items": [
                {
                    "id": "seg_0001_item_0001",
                    "type": "word",
                    "text": "Bonjour",
                    "whisper": {
                        "start_ms": 100,
                        "end_ms": 800,
                        "probability": 0.8765,
                        "probability_min": 0.8765,
                        "tokens": [
                            {"text": "Bonjour", "start_ms": 100, "end_ms": 800, "p": 0.8765}
                        ],
                    },
                }
            ],
            "whisper": {
                "start_ms": 100,
                "end_ms": 800,
                "probability": 0.8765,
                "probability_min": 0.8765,
                "num_tokens": 1,
            },
        }
    ]


def test_build_items_groups_subtokens_into_words_and_punctuation() -> None:
    items = _build_items(
        [
            {"text": " J", "start_ms": 10, "end_ms": 20, "p": 0.2},
            {"text": "'", "start_ms": 20, "end_ms": 25, "p": 0.99},
            {"text": "ai", "start_ms": 25, "end_ms": 40, "p": 0.8},
            {"text": " ?", "start_ms": 40, "end_ms": 45, "p": 0.1},
        ],
        seg_index=1,
    )

    assert items == [
        {
            "id": "seg_0001_item_0001",
            "type": "word",
            "text": "J'ai",
            "whisper": {
                "start_ms": 10,
                "end_ms": 40,
                "probability": 0.6633,
                "probability_min": 0.2,
                "tokens": [
                    {"text": " J", "start_ms": 10, "end_ms": 20, "p": 0.2},
                    {"text": "'", "start_ms": 20, "end_ms": 25, "p": 0.99},
                    {"text": "ai", "start_ms": 25, "end_ms": 40, "p": 0.8},
                ],
            },
        },
        {
            "id": "seg_0001_item_0002",
            "type": "punctuation",
            "text": "?",
            "whisper": {
                "start_ms": 40,
                "end_ms": 45,
                "probability": 0.1,
                "probability_min": 0.1,
                "tokens": [
                    {"text": " ?", "start_ms": 40, "end_ms": 45, "p": 0.1},
                ],
            },
        },
    ]


def test_build_vad_and_lines() -> None:
    voice_ranges = _build_voice_ranges([
        {"start_ms": 100, "end_ms": 600},
        {"start_ms": 1000, "end_ms": 1600},
    ])
    vad = _build_vad(voice_ranges, audio_duration_ms=2000)

    assert vad["num_voice_ranges"] == 2
    assert vad["speech_duration_s"] == 1.1
    assert vad["audio_duration_s"] == 2.0
    assert vad["reduction_pct"] == 45.0

    lines = _build_lines([
        {"text": "Bonjour", "whisper": {"start_ms": 100, "end_ms": 1600}}
    ])
    assert lines == ["[00:00:00.100 --> 00:00:01.600]   Bonjour"]
    assert _ms_to_timestamp(3723004) == "01:02:03.004"


def test_assign_voice_range_id() -> None:
    voice_ranges = _build_voice_ranges([
        {"start_ms": 100, "end_ms": 600},
        {"start_ms": 1000, "end_ms": 1600},
    ])

    assert _assign_voice_range_id(100, voice_ranges) == "vr_0001"
    assert _assign_voice_range_id(1200, voice_ranges) == "vr_0002"
    assert _assign_voice_range_id(900, voice_ranges) is None


def test_compact_voice_ranges_in_json() -> None:
    text = json.dumps(
        {
            "vad": {
                "voice_ranges": [
                    {"id": "vr_0001", "start_ms": 100, "end_ms": 600},
                    {"id": "vr_0002", "start_ms": 1000, "end_ms": 1600},
                ]
            }
        },
        indent=2,
    )

    compacted = _compact_voice_ranges_in_json(text)

    assert '{ "id": "vr_0001", "start_ms": 100, "end_ms": 600 }' in compacted
    assert '{ "id": "vr_0002", "start_ms": 1000, "end_ms": 1600 }' in compacted

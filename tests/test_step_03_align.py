"""Tests for step 03 alignment helpers."""

from __future__ import annotations

from clotaire.step_03_align import (
    _anchor_segment_punctuation,
    _apply_fallback,
    _build_segment_wav2vec2,
    _is_punctuation_word,
    _normalize_for_alignment,
    _promote_alignment_timings,
)


def test_normalize_for_alignment_strips_punctuation_and_accents() -> None:
    assert _normalize_for_alignment("là ?") == "la"
    assert _normalize_for_alignment("Saint-Yves") == "saintyves"
    assert _normalize_for_alignment("J'ai") == "j'ai"


def test_is_punctuation_word() -> None:
    assert _is_punctuation_word("?") is True
    assert _is_punctuation_word("...") is True
    assert _is_punctuation_word("Aussi.") is False


def test_apply_fallback_copies_whisper_timing_into_wav2vec2() -> None:
    segment = {
        "id": "seg_0001",
        "text": "Bonjour ?",
        "whisper": {"start_ms": 0, "end_ms": 1000},
        "items": [
            {
                "id": "seg_0001_item_0001",
                "type": "word",
                "text": "Bonjour",
                "whisper": {
                    "start_ms": 100,
                    "end_ms": 300,
                    "probability": 0.9,
                    "probability_min": 0.9,
                    "tokens": [],
                },
            },
            {
                "id": "seg_0001_item_0002",
                "type": "punctuation",
                "text": "?",
                "whisper": {
                    "start_ms": 300,
                    "end_ms": 320,
                    "probability": 0.2,
                    "probability_min": 0.2,
                    "tokens": [],
                },
            },
        ],
    }

    _apply_fallback(segment, reason="test", voice_range_id="vr_0001")

    assert segment["items"][0]["wav2vec2"]["start_ms"] == 100
    assert segment["items"][0]["wav2vec2"]["end_ms"] == 300
    assert segment["items"][0]["wav2vec2"]["status"] == "fallback"
    assert segment["wav2vec2"]["status"] == "fallback"
    assert segment["wav2vec2"]["reason"] == "test"
    assert segment["wav2vec2"]["voice_range_id"] == "vr_0001"


def test_build_segment_wav2vec2_uses_lexical_item_bounds() -> None:
    segment = {
        "id": "seg_0001",
        "text": "Bonjour ?",
        "whisper": {"start_ms": 0, "end_ms": 1000},
        "items": [
            {
                "type": "word",
                "text": "Bonjour",
                "whisper": {"start_ms": 100, "end_ms": 300},
                "wav2vec2": {"start_ms": 120, "end_ms": 280, "status": "aligned"},
            },
            {
                "type": "punctuation",
                "text": "?",
                "whisper": {"start_ms": 300, "end_ms": 300},
                "wav2vec2": {"start_ms": 280, "end_ms": 280, "status": "anchored_to_previous_word"},
            },
        ],
    }

    result = _build_segment_wav2vec2(segment, status="aligned", voice_range_id="vr_0001")

    assert result["start_ms"] == 120
    assert result["end_ms"] == 280
    assert result["status"] == "aligned"
    assert result["voice_range_id"] == "vr_0001"


def test_promote_alignment_timings() -> None:
    segment = {
        "wav2vec2": {"start_ms": 120, "end_ms": 280, "status": "aligned"},
        "items": [
            {
                "type": "word",
                "text": "Bonjour",
                "wav2vec2": {"start_ms": 120, "end_ms": 280, "status": "aligned"},
            },
            {
                "type": "punctuation",
                "text": "!",
                "wav2vec2": {"start_ms": 280, "end_ms": 280, "status": "anchored_to_previous_word"},
            },
        ],
    }

    _promote_alignment_timings(segment)

    assert segment["start_ms"] == 120
    assert segment["end_ms"] == 280
    assert segment["items"][0]["start_ms"] == 120
    assert segment["items"][0]["end_ms"] == 280
    assert segment["items"][1]["start_ms"] == 280
    assert segment["items"][1]["end_ms"] == 280


def test_anchor_segment_punctuation() -> None:
    segment = {
        "items": [
            {
                "type": "word",
                "text": "Bonjour",
                "wav2vec2": {"start_ms": 100, "end_ms": 200, "status": "aligned"},
                "whisper": {"start_ms": 90, "end_ms": 210},
            },
            {
                "type": "punctuation",
                "text": "!",
                "whisper": {"start_ms": 210, "end_ms": 220},
            },
        ]
    }

    _anchor_segment_punctuation(segment)

    assert segment["items"][1]["wav2vec2"] == {
        "start_ms": 200,
        "end_ms": 200,
        "status": "anchored_to_previous_word",
    }

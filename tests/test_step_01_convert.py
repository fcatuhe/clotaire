"""Tests for step 01 conversion helpers."""

from __future__ import annotations

from pathlib import Path

from clotaire.step_01_convert import _build_step, _filter_and_order, _parse_ffprobe_json


def test_parse_ffprobe_json_converts_numbers_but_preserves_tags() -> None:
    data = _parse_ffprobe_json(
        """
        {
          "format": {
            "duration": "26.775542",
            "bit_rate": "193223",
            "tags": {
              "title": "260331_1031",
              "track": "01"
            }
          },
          "streams": [
            {
              "sample_rate": "16000",
              "channels": "1"
            }
          ]
        }
        """
    )

    assert data["format"]["duration"] == 26.775542
    assert data["format"]["bit_rate"] == 193223
    assert data["format"]["tags"]["title"] == "260331_1031"
    assert data["format"]["tags"]["track"] == "01"
    assert data["streams"][0]["sample_rate"] == 16000
    assert data["streams"][0]["channels"] == 1


def test_filter_and_order_keeps_only_known_keys() -> None:
    raw = {
        "format": {
            "format_name": "wav",
            "duration": 1.2,
            "size": 100,
            "ignored": "x",
        },
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": 16000,
                "channels": 1,
                "ignored": "x",
            }
        ],
    }

    filtered = _filter_and_order(raw)

    assert filtered == {
        "format": {
            "format_name": "wav",
            "duration": 1.2,
            "size": 100,
        },
        "streams": [
            {
                "codec_type": "audio",
                "sample_rate": 16000,
                "channels": 1,
            }
        ],
    }


def test_build_step_matches_current_schema(tmp_path: Path) -> None:
    media_path = tmp_path / "input.mp3"
    wav_path = tmp_path / "01_convert.input.wav"

    original_probe = {"format": {"format_name": "mp3"}, "streams": [{"codec_type": "audio"}]}
    converted_probe = {
        "format": {"format_name": "wav"},
        "streams": [{"codec_name": "pcm_s16le", "sample_rate": 16000, "channels": 1}],
    }

    step = _build_step(media_path, original_probe, wav_path, converted_probe, wall_time_s=0.42)

    assert step["step"] == "01_convert"
    assert "downstream_requirements" not in step
    assert step["original"]["file"] == "input.mp3"
    assert step["converted"]["file"] == "01_convert.input.wav"
    assert step["timing"]["wall_s"] == 0.42

"""Tests for step ⑦ — Formatting final."""

from __future__ import annotations

import json

from kloter.steps.format import format_output, to_markdown, write_files


class TestFormatOutput:
    """Tests for format_output function."""

    def test_basic_structure(self, sample_words, sample_diar_segments):
        """Result dict has all required top-level keys."""
        import numpy as np
        audio = np.zeros(32000, dtype=np.float32)  # 2 seconds

        result = format_output(sample_words, sample_diar_segments, [{"start": 0.0, "end": 1.8}], "test.wav", audio)

        assert "audio" in result
        assert "duration" in result
        assert "languages" in result
        assert "words" in result
        assert "segments" in result
        assert "diarization" in result
        assert "speech_segments" in result

    def test_duration_calculation(self, sample_words, sample_diar_segments):
        """Duration is calculated from audio array length."""
        import numpy as np
        audio = np.zeros(48000, dtype=np.float32)  # 3 seconds at 16kHz

        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)
        assert result["duration"] == 3.0

    def test_languages_computed(self, sample_words, sample_diar_segments):
        """Languages dict is computed from word durations."""
        import numpy as np
        audio = np.zeros(32000, dtype=np.float32)

        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)
        assert "fr" in result["languages"]
        assert result["languages"]["fr"] > 0

    def test_segments_built(self, sample_words, sample_diar_segments):
        """Segments are built from consecutive words with same speaker."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        assert len(result["segments"]) >= 2
        assert result["segments"][0]["speaker"] == "SPEAKER_00"
        assert result["segments"][0]["text"] == "Bonjour comment allez"
        assert result["segments"][1]["speaker"] == "SPEAKER_01"
        assert result["segments"][1]["text"] == "vous"


class TestToMarkdown:
    """Tests for to_markdown function."""

    def test_basic_markdown(self, sample_words, sample_diar_segments):
        """Markdown output contains key elements."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)
        md = to_markdown(result)

        assert "# Transcription" in md
        assert "Duration" in md
        assert "Languages" in md
        assert "SPEAKER_00" in md
        assert "SPEAKER_01" in md
        assert "kloter v" in md

    def test_empty_result(self):
        """Markdown handles empty result gracefully."""
        result = {
            "audio": "empty.wav",
            "duration": 0.0,
            "languages": {},
            "words": [],
            "segments": [],
            "diarization": [],
            "speech_segments": [],
        }
        md = to_markdown(result)
        assert "# Transcription" in md


class TestWriteFiles:
    """Tests for write_files function."""

    def test_write_both_formats(self, sample_words, sample_diar_segments, tmp_path):
        """Default writes both JSON and Markdown."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        paths = write_files(result, "test.wav", output_dir=str(tmp_path), fmt="all")
        assert len(paths) == 2
        assert paths[0].name == "test.transcription.json"
        assert paths[1].name == "test.transcription.md"

    def test_write_json_only(self, sample_words, sample_diar_segments, tmp_path):
        """fmt='json' writes only JSON."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        paths = write_files(result, "test.wav", output_dir=str(tmp_path), fmt="json")
        assert len(paths) == 1
        assert paths[0].suffixes == [".transcription", ".json"]

    def test_write_md_only(self, sample_words, sample_diar_segments, tmp_path):
        """fmt='md' writes only Markdown."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        paths = write_files(result, "test.wav", output_dir=str(tmp_path), fmt="md")
        assert len(paths) == 1
        assert paths[0].suffixes == [".transcription", ".md"]

    def test_json_is_valid(self, sample_words, sample_diar_segments, tmp_path):
        """Written JSON is valid and can be parsed back."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        write_files(result, "test.wav", output_dir=str(tmp_path), fmt="json")
        json_path = tmp_path / "test.transcription.json"
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
        assert parsed["audio"] == "test.wav"

    def test_creates_output_dir(self, sample_words, sample_diar_segments, tmp_path):
        """write_files creates the output directory if it doesn't exist."""
        import numpy as np

        audio = np.zeros(32000, dtype=np.float32)
        result = format_output(sample_words, sample_diar_segments, [], "test.wav", audio)

        nested_dir = tmp_path / "nested" / "dir"
        paths = write_files(result, "test.wav", output_dir=str(nested_dir), fmt="json")
        assert paths[0].exists()

"""Tests for step ⑥ — Speaker matching."""

from __future__ import annotations

from kloter.steps.match import match_speakers


class TestMatchSpeakers:
    """Tests for match_speakers function."""

    def test_basic_assignment(self, sample_words, sample_diar_segments):
        """Words get assigned to the speaker with the most overlap."""
        result = match_speakers(sample_words, sample_diar_segments)
        assert len(result) == len(sample_words)
        # First 3 words (0.0–1.5) overlap mostly with SPEAKER_00 (0.0–1.2)
        assert result[0]["speaker"] == "SPEAKER_00"
        # Word at 1.5–1.8 overlaps more with SPEAKER_01 (1.2–2.0)
        assert result[3]["speaker"] == "SPEAKER_01"

    def test_empty_words(self, sample_diar_segments):
        """Empty word list returns empty result."""
        result = match_speakers([], sample_diar_segments)
        assert result == []

    def test_empty_diarization(self, sample_words):
        """With no diarization, all speakers are UNKNOWN then propagated."""
        result = match_speakers(sample_words, [])
        # UNKNOWN propagates forward/backward, so all UNKNOWN
        assert all(w["speaker"] == "UNKNOWN" for w in result)

    def test_unknown_propagation(self):
        """Words in diarization gaps get speaker from nearest neighbor."""
        words = [
            {"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.9, "align_score": 0.8, "language": "en"},
            {"start": 0.5, "end": 1.0, "word": "world", "probability": 0.85, "align_score": 0.7, "language": "en"},
            {"start": 1.0, "end": 1.5, "word": "foo", "probability": 0.8, "align_score": 0.6, "language": "en"},
        ]
        diar = [
            {"start": 0.0, "end": 0.6, "speaker": "SPEAKER_00"},
            # Gap from 0.6 to 1.5 — no diarization
        ]
        result = match_speakers(words, diar)
        # All should be SPEAKER_00 after propagation
        assert all(w["speaker"] == "SPEAKER_00" for w in result)

    def test_coherence_short_segment(self):
        """Short segments (<2s) should have coherent speaker assignment."""
        words = [
            {"start": 0.0, "end": 0.3, "word": "je", "probability": 0.7, "align_score": 0.5, "language": "fr"},
            {"start": 0.3, "end": 0.6, "word": "sais", "probability": 0.8, "align_score": 0.6, "language": "fr"},
            {"start": 0.6, "end": 0.9, "word": "pas", "probability": 0.6, "align_score": 0.4, "language": "fr"},
        ]
        diar = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 0.3, "end": 0.5, "speaker": "SPEAKER_01"},  # brief overlap
        ]
        result = match_speakers(words, diar)
        # All should agree (majority vote in short segment)
        speakers = {w["speaker"] for w in result}
        assert len(speakers) == 1

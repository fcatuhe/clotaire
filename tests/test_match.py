"""Tests for step ⑥ — Speaker matching."""

from __future__ import annotations

from clotaire.steps.match import match_speakers, _apply_coherence, _best_speaker


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

    def test_coherence_short_segment_overlap(self):
        """Short segments (<2s): words covering >50% of segment get that speaker.

        The word 'sais' (0.3–0.6) covers 100% of the short SPEAKER_01 segment
        (0.3–0.5), so it gets SPEAKER_01 even though raw overlap favors
        SPEAKER_00 (0.0–1.0).
        """
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
        # "sais" covers 100% of the SPEAKER_01 interjection → SPEAKER_01
        assert result[1]["speaker"] == "SPEAKER_01"
        # Other words are outside the short SPEAKER_01 segment
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_coherence_interjection_aussi(self):
        """Brief interjection (Aussi.) sandwiched between same-speaker segments.

        This is the pattern from the community-1 misalignment: a 51ms
        SPEAKER_00 turn sandwiched between two SPEAKER_01 segments.
        The sandwich-gap override extends the interjection zone into
        the surrounding segments, correctly assigning 'Aussi.' to SPEAKER_00.
        """
        words = [
            {"start": 7.277, "end": 7.707, "word": "photo", "probability": 0.9, "align_score": 0.7, "language": "fr"},
            {"start": 8.526, "end": 8.666, "word": "Aussi.", "probability": 0.9, "align_score": 0.4, "language": "fr"},
            {"start": 9.079, "end": 9.160, "word": "Eh,", "probability": 0.7, "align_score": 0.5, "language": "fr"},
        ]
        diar = [
            {"start": 6.376, "end": 9.008, "speaker": "SPEAKER_01"},
            {"start": 9.008, "end": 9.059, "speaker": "SPEAKER_00"},  # 51ms interjection
            {"start": 9.059, "end": 9.97, "speaker": "SPEAKER_01"},
        ]
        result = match_speakers(words, diar)
        # "Aussi." falls in the sandwich-gap zone → SPEAKER_00
        assert result[1]["speaker"] == "SPEAKER_00"
        # "photo" is well before the zone → SPEAKER_01
        assert result[0]["speaker"] == "SPEAKER_01"
        # "Eh," also falls in the sandwich-gap zone → SPEAKER_00
        assert result[2]["speaker"] == "SPEAKER_00"

    def test_coherence_no_coverage(self):
        """Words that don't overlap with a short segment are unaffected."""
        words = [
            {"start": 0.0, "end": 0.3, "word": "je", "probability": 0.7, "align_score": 0.5, "language": "fr"},
            {"start": 0.3, "end": 0.6, "word": "sais", "probability": 0.8, "align_score": 0.6, "language": "fr"},
        ]
        diar = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 1.7, "speaker": "SPEAKER_01"},  # no overlap with any word
        ]
        result = match_speakers(words, diar)
        assert result[0]["speaker"] == "SPEAKER_00"
        assert result[1]["speaker"] == "SPEAKER_00"

    def test_coherence_highest_coverage_wins(self):
        """When multiple short segments claim a word with equal coverage, first wins."""
        words = [
            {"start": 0.0, "end": 1.5, "word": "longword", "probability": 0.8, "align_score": 0.6, "language": "fr"},
        ]
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 0.1, "end": 0.3, "speaker": "SPEAKER_01"},  # 100% covered by word
            {"start": 0.5, "end": 1.0, "speaker": "SPEAKER_02"},  # also 100% covered
        ]
        result = match_speakers(words, diar)
        # Both short segments have coverage 1.0. The first one processed
        # (SPEAKER_01) sets coverage=1.0. SPEAKER_02 also has coverage=1.0
        # but since 1.0 is NOT > 1.0, it doesn't override.
        assert result[0]["speaker"] == "SPEAKER_01"

    def test_no_temporary_metadata_leaked(self):
        """_coherence_coverage is cleaned up after coherence step."""
        words = [
            {"start": 0.0, "end": 0.5, "word": "test", "probability": 0.9, "align_score": 0.8, "language": "en"},
        ]
        diar = [
            {"start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
        ]
        result = match_speakers(words, diar)
        for w in result:
            assert "_coherence_coverage" not in w

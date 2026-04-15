"""Tests for step ⑤ — Alignment splitting at speaker changes."""

from __future__ import annotations

from kloter.steps.align import (
    _find_speaker_changes,
    _split_segment,
    _merge_short_subsegments,
    _split_at_speaker_changes,
)


class TestFindSpeakerChanges:
    """Tests for _find_speaker_changes."""

    def test_no_changes_single_segment(self):
        """Single diarization segment → no change points."""
        diar = [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}]
        assert _find_speaker_changes(diar) == []

    def test_same_speaker_no_change(self):
        """Consecutive segments with same speaker → no change points."""
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_00"},
        ]
        assert _find_speaker_changes(diar) == []

    def test_clean_speaker_change(self):
        """Non-overlapping speaker change produces a change point."""
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
        ]
        assert _find_speaker_changes(diar) == [2.0]

    def test_overlapping_no_change(self):
        """Overlapping segments with different speakers → no change point."""
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},  # overlap
        ]
        assert _find_speaker_changes(diar) == []

    def test_multiple_changes(self):
        """Multiple clean speaker changes produce multiple change points."""
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
            {"start": 4.0, "end": 6.0, "speaker": "SPEAKER_00"},
            {"start": 6.0, "end": 8.0, "speaker": "SPEAKER_01"},
        ]
        assert _find_speaker_changes(diar) == [2.0, 4.0, 6.0]

    def test_gap_between_segments(self):
        """Gap between segments still produces a change point."""
        diar = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 5.0, "speaker": "SPEAKER_01"},
        ]
        assert _find_speaker_changes(diar) == [3.0]

    def test_empty_diarization(self):
        """Empty diarization → no change points."""
        assert _find_speaker_changes([]) == []

    def test_real_sample_diarization(self):
        """Change points from the sample audio diarization."""
        diar = [
            {"start": 0.267, "end": 1.583, "speaker": "SPEAKER_01"},
            {"start": 1.398, "end": 1.432, "speaker": "SPEAKER_00"},  # overlap, skip
            {"start": 2.41, "end": 6.342, "speaker": "SPEAKER_00"},
            {"start": 6.376, "end": 9.008, "speaker": "SPEAKER_01"},
            {"start": 9.008, "end": 9.059, "speaker": "SPEAKER_00"},
            {"start": 9.059, "end": 9.97, "speaker": "SPEAKER_01"},
            {"start": 17.581, "end": 18.796, "speaker": "SPEAKER_01"},
            {"start": 20.045, "end": 20.433, "speaker": "SPEAKER_00"},
            {"start": 20.433, "end": 22.036, "speaker": "SPEAKER_01"},
            {"start": 22.036, "end": 22.998, "speaker": "SPEAKER_00"},
            {"start": 23.673, "end": 24.787, "speaker": "SPEAKER_01"},
            {"start": 23.723, "end": 24.905, "speaker": "SPEAKER_00"},  # overlap, skip
        ]
        changes = _find_speaker_changes(diar)
        # Clean transitions (overlaps skipped):
        # 2.41 (SPEAKER_01→SPEAKER_00 after overlap skipped)
        # 6.376 (SPEAKER_00→SPEAKER_01)
        # 9.008 (SPEAKER_01→SPEAKER_00)
        # 9.059 (SPEAKER_00→SPEAKER_01)
        # 20.045 (SPEAKER_01→SPEAKER_00)
        # 20.433 (SPEAKER_00→SPEAKER_01)
        # 22.036 (SPEAKER_01→SPEAKER_00)
        # 23.673 (SPEAKER_00→SPEAKER_01)
        expected = [2.41, 6.376, 9.008, 9.059, 20.045, 20.433, 22.036, 23.673]
        assert changes == expected


class TestSplitSegment:
    """Tests for _split_segment."""

    def test_no_change_points(self):
        """No relevant change points → segment unchanged."""
        seg = {
            "start": 0.0, "end": 5.0, "language": "fr",
            "words": [{"start": 0.0, "end": 0.5, "word": "Bonjour"}],
        }
        result = _split_segment(seg, [])
        assert len(result) == 1
        assert result[0] is seg

    def test_single_split(self):
        """One change point splits segment into two sub-segments."""
        seg = {
            "start": 0.0, "end": 5.0, "language": "fr",
            "words": [
                {"start": 0.5, "end": 1.0, "word": "Bonjour"},
                {"start": 3.0, "end": 3.5, "word": "monde"},
            ],
        }
        result = _split_segment(seg, [2.5])
        assert len(result) == 2
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 2.5
        assert len(result[0]["words"]) == 1
        assert result[0]["words"][0]["word"] == "Bonjour"
        assert result[1]["start"] == 2.5
        assert result[1]["end"] == 5.0
        assert len(result[1]["words"]) == 1
        assert result[1]["words"][0]["word"] == "monde"

    def test_midpoint_assignment(self):
        """Words are assigned to the sub-segment containing their midpoint."""
        seg = {
            "start": 0.0, "end": 4.0, "language": "fr",
            "words": [
                {"start": 0.0, "end": 1.5, "word": "left"},     # midpoint 0.75 → first
                {"start": 1.5, "end": 2.5, "word": "boundary"},  # midpoint 2.0 → first (< 2.5)
                {"start": 2.5, "end": 4.0, "word": "right"},     # midpoint 3.25 → second
            ],
        }
        result = _split_segment(seg, [2.5])
        assert len(result) == 2
        assert len(result[0]["words"]) == 2
        assert result[0]["words"][0]["word"] == "left"
        assert result[0]["words"][1]["word"] == "boundary"
        assert len(result[1]["words"]) == 1
        assert result[1]["words"][0]["word"] == "right"

    def test_change_point_outside_segment(self):
        """Change points outside the segment are ignored."""
        seg = {
            "start": 2.0, "end": 5.0, "language": "fr",
            "words": [{"start": 2.5, "end": 3.0, "word": "test"}],
        }
        result = _split_segment(seg, [1.0, 6.0])  # both outside
        assert len(result) == 1
        assert result[0] is seg

    def test_language_preserved(self):
        """Sub-segments inherit the parent segment's language."""
        seg = {
            "start": 0.0, "end": 4.0, "language": "fr",
            "words": [
                {"start": 0.0, "end": 1.0, "word": "gauche"},
                {"start": 3.0, "end": 4.0, "word": "droite"},
            ],
        }
        result = _split_segment(seg, [2.0])
        assert result[0]["language"] == "fr"
        assert result[1]["language"] == "fr"


class TestMergeShortSubsegments:
    """Tests for _merge_short_subsegments."""

    def test_no_merge_needed(self):
        """All sub-segments long enough → no merging."""
        sub_segs = [
            {"start": 0.0, "end": 2.0, "language": "fr", "words": [{"word": "a"}]},
            {"start": 2.0, "end": 4.0, "language": "fr", "words": [{"word": "b"}]},
        ]
        result = _merge_short_subsegments(sub_segs, min_duration=0.5)
        assert len(result) == 2

    def test_merge_short_with_previous(self):
        """Short sub-segment is merged into the previous one."""
        sub_segs = [
            {"start": 0.0, "end": 2.0, "language": "fr", "words": [{"word": "a"}]},
            {"start": 2.0, "end": 2.3, "language": "fr", "words": [{"word": "b"}]},  # 0.3s < 0.5s
            {"start": 2.3, "end": 5.0, "language": "fr", "words": [{"word": "c"}]},
        ]
        result = _merge_short_subsegments(sub_segs, min_duration=0.5)
        assert len(result) == 2
        assert result[0]["end"] == 2.3  # merged
        assert len(result[0]["words"]) == 2
        assert result[1]["start"] == 2.3

    def test_merge_short_first_segment(self):
        """First sub-segment too short → merged with next."""
        sub_segs = [
            {"start": 0.0, "end": 0.3, "language": "fr", "words": [{"word": "a"}]},  # 0.3s
            {"start": 0.3, "end": 3.0, "language": "fr", "words": [{"word": "b"}]},
        ]
        result = _merge_short_subsegments(sub_segs, min_duration=0.5)
        assert len(result) == 1
        assert result[0]["start"] == 0.0
        assert result[0]["end"] == 3.0
        assert len(result[0]["words"]) == 2

    def test_empty_input(self):
        """Empty list → empty result."""
        assert _merge_short_subsegments([], min_duration=0.5) == []


class TestSplitAtSpeakerChanges:
    """Integration tests for _split_at_speaker_changes."""

    def test_no_diarization(self):
        """No diarization segments → segments unchanged."""
        segments = [
            {"start": 0.0, "end": 5.0, "language": "fr", "words": [{"start": 0.0, "end": 0.5, "word": "test"}]},
        ]
        result = _split_at_speaker_changes(segments, [])
        assert len(result) == 1

    def test_split_long_segment(self):
        """Long whisper segment split at speaker change."""
        segments = [{
            "start": 0.0, "end": 10.0, "language": "fr",
            "words": [
                {"start": 0.5, "end": 1.0, "word": "bonjour"},
                {"start": 1.5, "end": 2.0, "word": "je"},
                {"start": 5.5, "end": 6.0, "word": "merci"},
                {"start": 7.0, "end": 7.5, "word": "oui"},
            ],
        }]
        diar = [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
        result = _split_at_speaker_changes(segments, diar)
        assert len(result) == 2
        # First sub-segment: words with midpoint < 5.0
        assert result[0]["end"] == 5.0
        assert any(w["word"] == "bonjour" for w in result[0]["words"])
        assert any(w["word"] == "je" for w in result[0]["words"])
        # Second sub-segment: words with midpoint >= 5.0
        assert result[1]["start"] == 5.0
        assert any(w["word"] == "merci" for w in result[1]["words"])
        assert any(w["word"] == "oui" for w in result[1]["words"])

    def test_interjection_creates_short_subsegment_merged(self):
        """Very short interjection sub-segment is merged with neighbour.

        Reproduces the 'Aussi.' pattern: a 51ms SPEAKER_00 turn surrounded
        by SPEAKER_01.  The resulting 51ms sub-segment is merged away.
        """
        segments = [{
            "start": 6.376, "end": 9.97, "language": "fr",
            "words": [
                {"start": 6.5, "end": 7.0, "word": "parle"},
                {"start": 8.397, "end": 9.247, "word": "Aussi."},
                {"start": 9.297, "end": 9.5, "word": "eh"},
            ],
        }]
        diar = [
            {"start": 6.376, "end": 9.008, "speaker": "SPEAKER_01"},
            {"start": 9.008, "end": 9.059, "speaker": "SPEAKER_00"},  # 51ms
            {"start": 9.059, "end": 9.97, "speaker": "SPEAKER_01"},
        ]
        result = _split_at_speaker_changes(segments, diar)
        # The 51ms sub-segment (9.008-9.059) should be merged with neighbour
        # so no sub-segment shorter than 0.5s remains
        for seg in result:
            duration = seg["end"] - seg["start"]
            assert duration >= 0.5, f"Sub-segment too short: {duration}s"


class TestAnchorPunctuation:
    """Tests for _anchor_punctuation."""

    def test_question_mark_anchored(self):
        """Standalone '?' snaps to preceding word's end."""
        from kloter.steps.align import _anchor_punctuation
        words = [
            {"start": 1.37, "end": 1.45, "word": "là", "align_score": 0.33},
            {"start": 2.633, "end": 2.653, "word": "?", "align_score": 0.004},
        ]
        result = _anchor_punctuation(words)
        assert result[1]["start"] == 1.45
        assert result[1]["end"] == 1.45
        assert result[1]["align_score"] is None

    def test_exclamation_anchored(self):
        """Standalone '!' snaps to preceding word's end."""
        from kloter.steps.align import _anchor_punctuation
        words = [
            {"start": 9.79, "end": 9.83, "word": "eh", "align_score": 0.13},
            {"start": 9.85, "end": 9.87, "word": "!", "align_score": 0.005},
        ]
        result = _anchor_punctuation(words)
        assert result[1]["start"] == 9.83
        assert result[1]["end"] == 9.83
        assert result[1]["align_score"] is None

    def test_word_with_punctuation_not_affected(self):
        """Words that contain letters (like 'Si.', 'Aussi.') are not moved."""
        from kloter.steps.align import _anchor_punctuation
        words = [
            {"start": 0.327, "end": 0.528, "word": "Si.", "align_score": 0.505},
            {"start": 9.108, "end": 9.529, "word": "Aussi.", "align_score": 0.376},
        ]
        result = _anchor_punctuation(words)
        assert result[0]["start"] == 0.327  # unchanged
        assert result[1]["start"] == 9.108  # unchanged

    def test_consecutive_punctuation(self):
        """Multiple punctuation in a row all anchor to last real word."""
        from kloter.steps.align import _anchor_punctuation
        words = [
            {"start": 1.0, "end": 1.5, "word": "quoi", "align_score": 0.5},
            {"start": 3.0, "end": 3.1, "word": ",", "align_score": 0.01},
            {"start": 4.0, "end": 4.1, "word": "?", "align_score": 0.01},
        ]
        result = _anchor_punctuation(words)
        assert result[1]["start"] == 1.5
        assert result[2]["start"] == 1.5

    def test_no_preceding_word(self):
        """Punctuation as first word stays where it is."""
        from kloter.steps.align import _anchor_punctuation
        words = [
            {"start": 0.0, "end": 0.1, "word": "...", "align_score": 0.1},
            {"start": 0.1, "end": 0.5, "word": "Bonjour", "align_score": 0.8},
        ]
        result = _anchor_punctuation(words)
        assert result[0]["start"] == 0.0  # no preceding word, unchanged

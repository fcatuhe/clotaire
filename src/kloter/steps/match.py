"""⑥ Speaker matching — assign speakers to words via temporal overlap."""

from __future__ import annotations

from typing import Any


def match_speakers(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign a speaker to each word based on temporal overlap with diarization.

    Each word gets the speaker of the diarization segment with the most overlap.
    Applies post-processing for coherence and sandwich-gap interjections.

    Args:
        words: List of word dicts with "start", "end" fields.
        diar_segments: List of {"start": float, "end": float, "speaker": str}.

    Returns:
        Same word list with "speaker" field added to each word.
    """
    # Clean diarization: drop very short noise (<50ms) that isn't sandwiched
    clean_diar = _clean_diarization(diar_segments)

    # Step 1: Basic overlap assignment
    for word in words:
        word["speaker"] = _best_speaker(word, clean_diar)

    # Step 2: Sandwich-gap interjection override
    # Short segments sandwiched between same-speaker segments represent
    # interjections that community-1 detects but mis-bounds.  The aligned
    # words can fall well outside the narrow diarization segment.
    words = _apply_sandwich_override(words, clean_diar)

    # Step 3: Intra-segment coherence (coverage-based for short segments)
    words = _apply_coherence(words, clean_diar)

    # Step 4: Propagation — fill gaps where no diarization overlaps
    words = _propagate_speakers(words)

    return words


def _best_speaker(
    word: dict[str, Any],
    diar_segments: list[dict[str, Any]],
) -> str:
    """Find the speaker with the most temporal overlap with a word."""
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for d in diar_segments:
        overlap = min(word["end"], d["end"]) - max(word["start"], d["start"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = d["speaker"]

    return best_speaker


def _clean_diarization(
    diar_segments: list[dict[str, Any]],
    min_duration: float = 0.05,
) -> list[dict[str, Any]]:
    """Remove very short diarization noise that isn't a sandwiched interjection.

    Segments shorter than *min_duration* that are NOT sandwiched between
    two same-speaker segments are dropped as diarization noise.
    Sandwiched short segments are kept — they represent real interjections.
    """
    sorted_segs = sorted(diar_segments, key=lambda d: d["start"])
    cleaned = []

    for i, d in enumerate(sorted_segs):
        dur = d["end"] - d["start"]
        if dur < min_duration:
            # Keep only if sandwiched between same-speaker segments
            prev_spk = sorted_segs[i - 1]["speaker"] if i > 0 else None
            nxt_spk = sorted_segs[i + 1]["speaker"] if i < len(sorted_segs) - 1 else None
            if (
                prev_spk
                and nxt_spk
                and prev_spk == nxt_spk
                and prev_spk != d["speaker"]
            ):
                cleaned.append(d)  # sandwiched interjection — keep
            # else: drop as noise
        else:
            cleaned.append(d)

    return cleaned


def _apply_sandwich_override(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Override speaker for words near sandwiched interjection gaps.

    Community-1 often detects short interjections but with imprecise boundaries.
    The aligned words can fall well outside the narrow diarization segment
    (before its start or after its end), because community-1 both over-extends
    the surrounding segments and under-bounds the interjection.

    When a short segment (<100ms) is sandwiched between two same-speaker
    segments, we extend the interjection's zone into the surrounding
    segments by a buffer proportional to the gap, with a minimum of 0.5s
    on each side.  Words currently assigned to the surrounding speaker
    whose midpoint falls in this zone get reassigned to the interjection
    speaker.
    """
    sorted_segs = sorted(diar_segments, key=lambda d: d["start"])

    for i, d in enumerate(sorted_segs):
        dur = d["end"] - d["start"]
        if dur >= 0.1 or i < 1 or i >= len(sorted_segs) - 1:
            continue

        prev = sorted_segs[i - 1]
        nxt = sorted_segs[i + 1]

        if prev["speaker"] != nxt["speaker"] or prev["speaker"] == d["speaker"]:
            continue

        # Sandwiched interjection found!
        # Extend zone into surrounding segments by at least 0.5s each side.
        # The buffer is proportional to the gap to handle very narrow gaps.
        gap = nxt["start"] - prev["end"]  # e.g. 9.059 - 9.008 = 0.051
        buffer = max(0.5, gap * 5)  # at least 0.5s, more for wider gaps
        zone_start = prev["end"] - buffer  # extend into preceding segment
        zone_end = nxt["start"] + buffer   # extend into following segment

        surrounding_speaker = prev["speaker"]
        interjection_speaker = d["speaker"]

        for w in words:
            if w["speaker"] != surrounding_speaker:
                continue  # only override surrounding-speaker words
            midpoint = (w["start"] + w["end"]) / 2
            if midpoint >= zone_start and midpoint <= zone_end:
                w["speaker"] = interjection_speaker

    return words


def _apply_coherence(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply intra-segment coherence for short diarization segments (<2s).

    Short segments (<2s) typically represent single-speaker turns (interjections,
    brief responses). The basic overlap matching tends to assign their words to
    the surrounding longer segment's speaker, because raw overlap duration
    always favors longer segments.

    This function fixes that by using *coverage* — the fraction of the short
    segment that a word covers. If a word covers >50% of a short segment,
    the word is assigned to that segment's speaker, overriding the raw-overlap
    result. When multiple short segments claim the same word, the one with
    the highest coverage wins.
    """
    for d_seg in diar_segments:
        seg_duration = d_seg["end"] - d_seg["start"]
        if seg_duration > 2.0:
            continue

        for w in words:
            overlap = min(w["end"], d_seg["end"]) - max(w["start"], d_seg["start"])
            if overlap <= 0:
                continue

            # How much of this short segment does the word cover?
            coverage = overlap / seg_duration
            if coverage > 0.5:
                # Strong signal: the word spans most of this short turn.
                prev_coverage = w.get("_coherence_coverage", 0.0)
                if coverage > prev_coverage:
                    w["speaker"] = d_seg["speaker"]
                    w["_coherence_coverage"] = coverage

    # Clean up temporary metadata
    for w in words:
        w.pop("_coherence_coverage", None)

    return words


def _propagate_speakers(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill UNKNOWN speakers by propagating from the nearest word with a speaker."""
    if not words:
        return words

    # Forward pass: propagate from left
    last_known = "UNKNOWN"
    for w in words:
        if w["speaker"] != "UNKNOWN":
            last_known = w["speaker"]
        else:
            w["speaker"] = last_known

    # Backward pass: propagate from right (for leading UNKNOWNs)
    last_known = "UNKNOWN"
    for w in reversed(words):
        if w["speaker"] != "UNKNOWN":
            last_known = w["speaker"]
        else:
            w["speaker"] = last_known

    return words

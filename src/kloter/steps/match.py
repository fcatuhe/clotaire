"""⑥ Speaker matching — assign speakers to words via temporal overlap."""

from __future__ import annotations

from typing import Any


def match_speakers(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign a speaker to each word based on temporal overlap with diarization.

    Each word gets the speaker of the diarization segment with the most overlap.
    Applies post-processing for coherence.

    Args:
        words: List of word dicts with "start", "end" fields.
        diar_segments: List of {"start": float, "end": float, "speaker": str}.

    Returns:
        Same word list with "speaker" field added to each word.
    """
    # Step 1: Basic overlap assignment
    for word in words:
        word["speaker"] = _best_speaker(word, diar_segments)

    # Step 2: Intra-segment coherence (majority vote for short segments)
    words = _apply_coherence(words, diar_segments)

    # Step 3: Propagation — fill gaps where no diarization overlaps
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


def _apply_coherence(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply intra-segment coherence: majority vote for short segments (<2s).

    In a short phrase (<2s), there's almost always only one speaker.
    Low-confidence words (align_score < 0.1) don't vote.
    """
    # Group words by diarization segment
    for d_seg in diar_segments:
        seg_duration = d_seg["end"] - d_seg["start"]
        if seg_duration > 2.0:
            continue

        # Find words in this segment
        seg_words = [
            w for w in words
            if w["start"] >= d_seg["start"]
            and w["end"] <= d_seg["end"]
        ]

        if not seg_words:
            continue

        # Majority vote (exclude low-confidence words)
        from collections import Counter
        voters = [
            w["speaker"] for w in seg_words
            if (w.get("align_score") or 0.0) >= 0.1 and w["speaker"] != "UNKNOWN"
        ]
        if not voters:
            continue

        majority = Counter(voters).most_common(1)[0][0]

        # Apply majority to all words in segment
        for w in seg_words:
            w["speaker"] = majority

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

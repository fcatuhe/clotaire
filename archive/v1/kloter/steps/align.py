"""⑤ Wav2vec2 alignment — per-segment language, cached models."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from kloter.models.loader import get_align_model, configure_threads


def detect_languages(segments: list[dict[str, Any]], top_n: int = 3) -> list[str]:
    """Return top-N languages by total speech duration.

    Args:
        segments: List of segment dicts with "language", "start", "end".
        top_n: Maximum number of languages to return.

    Returns:
        List of language codes, sorted by duration (descending).
    """
    duration_per_lang: dict[str, float] = defaultdict(float)
    for seg in segments:
        lang = seg["language"]
        dur = seg["end"] - seg["start"]
        duration_per_lang[lang] += dur
    sorted_langs = sorted(duration_per_lang.items(), key=lambda x: -x[1])
    return [lang for lang, _ in sorted_langs[:top_n]]


def align_words(
    segments: list[dict[str, Any]],
    audio: np.ndarray,
    max_languages: int = 3,
    diar_segments: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Align each segment with the wav2vec2 model of its detected language.

    Uses whisper's per-segment language detection to select the appropriate
    wav2vec2 alignment model. Models are cached to avoid reloading.
    Segments in minority languages (outside top-N) fall back to the majority language.

    When diarization segments are provided, whisper segments are split at
    speaker-change boundaries before alignment. Merged VAD segments can span
    multiple speaker turns; splitting them gives shorter sub-segments for more
    precise wav2vec2 alignment, reducing timing drift at speaker boundaries.

    Args:
        segments: Output from transcribe_segments (with "language" per segment).
        audio: Float32 numpy array, 16kHz mono (full audio).
        max_languages: Max number of wav2vec2 models to load (memory limit).
        diar_segments: Optional diarization segments for speaker-change splitting.

    Returns:
        List of word dicts with added "align_score" and "language" fields.
    """
    from whisperx.alignment import align as whisperx_align

    # Split whisper segments at diarization speaker-change boundaries
    # for more precise alignment (shorter segments = less timing drift).
    if diar_segments:
        segments = _split_at_speaker_changes(segments, diar_segments)

    configure_threads()
    supported_langs = detect_languages(segments, top_n=max_languages)
    majority_lang = supported_langs[0] if supported_langs else "en"

    all_aligned_words: list[dict[str, Any]] = []

    for seg in segments:
        lang = seg["language"]

        # Fallback to majority language if not in top-N
        language_fallback = False
        if lang not in supported_langs:
            lang = majority_lang
            language_fallback = True

        model, metadata = get_align_model(lang)

        # Build segment text for alignment
        seg_text = " ".join(w["word"] for w in seg["words"])
        if not seg_text.strip():
            # Skip empty segments
            for w in seg["words"]:
                w["align_score"] = None
                w["language"] = lang
                if language_fallback:
                    w["language_fallback"] = True
                all_aligned_words.append(w)
            continue

        # Prepare input for whisperx align
        wxs_segment = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg_text,
        }

        try:
            aligned = whisperx_align(
                [wxs_segment],
                model,
                metadata,
                audio,
                "cpu",
                return_char_alignments=False,
            )
            aligned_words = aligned.get("word_segments", [])
        except Exception:
            # If alignment fails, keep whisper timestamps as-is
            aligned_words = seg["words"]

        # Merge alignment results with existing word data
        for i, w in enumerate(seg["words"]):
            if i < len(aligned_words):
                aw = aligned_words[i]
                w["start"] = round(aw.get("start", w["start"]), 3)
                w["end"] = round(aw.get("end", w["end"]), 3)
                w["align_score"] = round(aw.get("score", 0.0), 3)
            else:
                w["align_score"] = None

            w["language"] = lang
            if language_fallback:
                w["language_fallback"] = True
                w["language_original"] = seg.get("language_original", seg.get("language"))
            elif seg.get("language_original"):
                # Even non-fallback segments may have had their language voted
                w["language_original"] = seg["language_original"]
            all_aligned_words.append(w)

    # Post-process: anchor punctuation-only words to the preceding word
    all_aligned_words = _anchor_punctuation(all_aligned_words)

    return all_aligned_words


def _anchor_punctuation(
    words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Anchor punctuation-only words to the preceding word's end time.

    Wav2vec2 cannot align punctuation (there is no acoustic signal), so it
    places it at random positions — often far from the word it belongs to.
    For example, a "?" after "là" might be placed 1.2s later, after the
    next speaker's first word.

    This function detects punctuation-only words (text contains no letters
    or digits) and snaps their start and end to the preceding word's end,
    giving them zero duration.  The align_score is set to None since the
    timing is synthetic.
    """
    import re

    _PUNCT_RE = re.compile(r'^[^\w]+$', re.UNICODE)  # no letters/digits/underscore

    for i, w in enumerate(words):
        if not _PUNCT_RE.match(w.get("word", "")):
            continue

        # Find the nearest preceding non-punctuation word
        prev_end = None
        for j in range(i - 1, -1, -1):
            if not _PUNCT_RE.match(words[j].get("word", "")):
                prev_end = words[j]["end"]
                break

        if prev_end is not None:
            w["start"] = prev_end
            w["end"] = prev_end
            w["align_score"] = None

    return words


def _split_at_speaker_changes(
    segments: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    min_subsegment: float = 0.5,
) -> list[dict[str, Any]]:
    """Split whisper segments at diarization speaker-change points.

    When VAD merges short-gap segments (min_duration_off=1.0), a single
    whisper segment can span multiple speaker turns.  Wav2vec2 alignment
    within such a long segment produces imprecise word timing, especially
    near speaker boundaries.

    Splitting at speaker-change points creates shorter sub-segments that
    give wav2vec2 tighter boundaries, reducing timing drift.

    Very short sub-segments (<min_subsegment) are merged with a neighbour
    so that wav2vec2 always has enough audio context.

    Args:
        segments: Whisper segments (with "words", "language", etc.).
        diar_segments: Diarization turns ("start", "end", "speaker").
        min_subsegment: Minimum duration (s) for a sub-segment after split.

    Returns:
        New list of segments, possibly more numerous than the input.
    """
    change_points = _find_speaker_changes(diar_segments)
    if not change_points:
        return segments

    result: list[dict[str, Any]] = []
    for seg in segments:
        sub_segs = _split_segment(seg, change_points)
        sub_segs = _merge_short_subsegments(sub_segs, min_duration=min_subsegment)
        result.extend(sub_segs)

    return result


def _find_speaker_changes(diar_segments: list[dict[str, Any]]) -> list[float]:
    """Return time points where the speaker changes (non-overlapping transitions).

    Walks the sorted diarization timeline and detects speaker transitions
    only at clean (non-overlapping) boundaries.  Overlapping segments
    (simultaneous speech) are skipped — the last_end is extended but
    the speaker label stays with the earlier segment, so we don't
    falsely detect a change inside an overlap.
    """
    if len(diar_segments) < 2:
        return []

    sorted_segs = sorted(diar_segments, key=lambda d: d["start"])
    changes: list[float] = []

    last_end = sorted_segs[0]["end"]
    last_speaker = sorted_segs[0]["speaker"]

    for seg in sorted_segs[1:]:
        if seg["start"] < last_end:
            # Overlapping — extend the horizon but keep the earlier speaker
            last_end = max(last_end, seg["end"])
            continue

        # Clean transition (gap or adjacency)
        if seg["speaker"] != last_speaker:
            changes.append(seg["start"])

        last_speaker = seg["speaker"]
        last_end = seg["end"]

    return sorted(set(changes))


def _split_segment(
    segment: dict[str, Any],
    change_points: list[float],
) -> list[dict[str, Any]]:
    """Split a single whisper segment at the given change points.

    Words are assigned to the sub-segment that contains their midpoint.
    """
    relevant = sorted(
        cp for cp in change_points
        if cp > segment["start"] and cp < segment["end"]
    )
    if not relevant:
        return [segment]

    boundaries = [segment["start"]] + relevant + [segment["end"]]
    groups: list[list[dict[str, Any]]] = [[] for _ in range(len(relevant) + 1)]

    for w in segment["words"]:
        midpoint = (w["start"] + w["end"]) / 2
        assigned = False
        for i, cp in enumerate(relevant):
            if midpoint < cp:
                groups[i].append(w)
                assigned = True
                break
        if not assigned:
            groups[-1].append(w)

    result: list[dict[str, Any]] = []
    for i, group in enumerate(groups):
        if not group:
            continue
        result.append({
            "start": boundaries[i],
            "end": boundaries[i + 1],
            "language": segment.get("language"),
            "language_prob": segment.get("language_prob"),
            "words": group,
        })

    return result


def _merge_short_subsegments(
    sub_segs: list[dict[str, Any]],
    min_duration: float = 0.5,
) -> list[dict[str, Any]]:
    """Merge consecutive sub-segments shorter than *min_duration*.

    A sub-segment that is too short for reliable wav2vec2 alignment is
    merged into the previous sub-segment (which has the same speaker on
    the other side of the brief interjection).
    """
    if not sub_segs:
        return sub_segs

    merged: list[dict[str, Any]] = [sub_segs[0].copy()]
    merged[0]["words"] = list(sub_segs[0]["words"])

    for seg in sub_segs[1:]:
        prev = merged[-1]
        if (seg["end"] - seg["start"]) < min_duration:
            # Merge into previous
            prev["end"] = seg["end"]
            prev["words"].extend(seg["words"])
        else:
            merged.append(seg.copy())
            merged[-1]["words"] = list(seg["words"])

    # If the very first sub-segment ended up too short, merge with next
    while len(merged) > 1 and (merged[0]["end"] - merged[0]["start"]) < min_duration:
        first = merged.pop(0)
        merged[0]["start"] = first["start"]
        merged[0]["words"] = first["words"] + merged[0]["words"]

    return merged

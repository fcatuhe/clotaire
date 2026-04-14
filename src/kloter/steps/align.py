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
) -> list[dict[str, Any]]:
    """Align each segment with the wav2vec2 model of its detected language.

    Uses whisper's per-segment language detection to select the appropriate
    wav2vec2 alignment model. Models are cached to avoid reloading.
    Segments in minority languages (outside top-N) fall back to the majority language.

    Args:
        segments: Output from transcribe_segments (with "language" per segment).
        audio: Float32 numpy array, 16kHz mono (full audio).
        max_languages: Max number of wav2vec2 models to load (memory limit).

    Returns:
        List of word dicts with added "align_score" and "language" fields.
    """
    from whisperx.alignment import align as whisperx_align

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
            aligned_words = aligned.get("words", [])
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
            all_aligned_words.append(w)

    return all_aligned_words

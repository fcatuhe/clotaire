"""③ Transcription (whisper.cpp) + ③bis merge tokens & language attach."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from typing import Any

import numpy as np

from kloter.models.loader import get_whisper_model_path, get_whisper_cli_path, configure_threads


def transcribe_segments(
    audio: np.ndarray,
    speech_segments: list[dict[str, float]],
    hf_token: str | None = None,
    whisper_threads: int = 8,
) -> list[dict[str, Any]]:
    """Transcribe each VAD segment using whisper.cpp with language detection.

    Each VAD segment is extracted to a temp WAV file, transcribed by
    whisper-cli with -l auto for language detection, and parsed back.

    Args:
        audio: Float32 numpy array, 16kHz mono (full audio).
        speech_segments: List of {"start": float, "end": float} from VAD.
        hf_token: HuggingFace token for model download.
        whisper_threads: Number of threads for whisper.cpp (default: 8).

    Returns:
        List of segment dicts, each with:
            - language: detected language code (e.g. "fr", "en")
            - language_prob: detection confidence
            - start, end: segment timing
            - words: list of word dicts with start, end, word, probability
    """
    configure_threads(whisper_threads=whisper_threads)
    model_path = get_whisper_model_path(hf_token=hf_token)
    cli_path = get_whisper_cli_path()

    results = []
    with tempfile.TemporaryDirectory(prefix="kloter_") as tmpdir:
        for i, seg in enumerate(speech_segments):
            start_sample = int(seg["start"] * 16000)
            end_sample = int(seg["end"] * 16000)
            segment_audio = audio[start_sample:end_sample]

            if len(segment_audio) < 1600:  # skip segments < 0.1s
                continue

            # Write segment to temp WAV file
            chunk_path = os.path.join(tmpdir, f"chunk_{i:04d}.wav")
            _write_wav(segment_audio, chunk_path)

            # Run whisper-cli on the chunk
            output_prefix = os.path.join(tmpdir, f"out_{i:04d}")
            cmd = [
                cli_path,
                "-m", model_path,
                "-l", "auto",           # auto-detect language
                "-t", str(whisper_threads),
                "-bo", "5",             # best-of
                "-bs", "5",             # beam size
                "-oj",                  # output JSON
                "-ojf",                 # output JSON full (with word timestamps)
                "-sow",                 # split on word
                "-of", output_prefix,
                chunk_path,
            ]

            subprocess.run(cmd, capture_output=True, check=True)

            # Parse the JSON output
            json_path = output_prefix + ".json"
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                whisper_output = json.load(f)

            # Extract language
            language = whisper_output.get("result", {}).get("language", "unknown")
            language_prob = None  # whisper.cpp doesn't output language confidence

            # Extract and merge tokens into words
            tokens = _extract_tokens(whisper_output)
            words = merge_tokens_to_words(tokens, offset=seg["start"])

            results.append({
                "language": language,
                "language_prob": language_prob,
                "start": seg["start"],
                "end": seg["end"],
                "words": words,
            })

    # Majority language vote: override minority languages on short segments
    results = _apply_language_vote(results)

    return results


def merge_tokens_to_words(
    tokens: list[dict[str, Any]],
    offset: float = 0.0,
) -> list[dict[str, Any]]:
    """Merge subword tokens into complete words.

    Whisper.cpp outputs subword tokens. Merge them by rule:
    - If a token does NOT start with a space, it's a continuation of the current word
    - If a token starts with a space, it's a new word

    Args:
        tokens: List of token dicts with "text", "start", "end", "probability".
        offset: Time offset to add (segment start time).

    Returns:
        List of word dicts with "start", "end", "word", "probability".
    """
    words: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for token in tokens:
        text = token.get("text", "")
        if not text or text.startswith("[_"):
            continue

        is_continuation = not text.startswith(" ")

        if current and is_continuation:
            # Continuation of current word
            current["end"] = round(token["end"] + offset, 3)
            current["word"] += text.strip()
            current["scores"].append(token.get("probability", 0.0))
        else:
            # New word — flush previous
            if current:
                current["probability"] = round(
                    sum(current["scores"]) / len(current["scores"]), 3
                )
                del current["scores"]
                words.append(current)

            current = {
                "start": round(token["start"] + offset, 3),
                "end": round(token["end"] + offset, 3),
                "word": text.strip(),
                "probability": None,
                "scores": [token.get("probability", 0.0)],
            }

    # Flush last word
    if current:
        current["probability"] = round(
            sum(current["scores"]) / len(current["scores"]), 3
        )
        del current["scores"]
        words.append(current)

    return words


def attach_language_to_words(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the detected language of each segment to its words.

    Args:
        segments: Output from transcribe_segments.

    Returns:
        Flat list of word dicts, each with an added "language" field.
    """
    all_words = []
    for seg in segments:
        lang = seg["language"]
        for w in seg["words"]:
            w["language"] = lang
            all_words.append(w)
    return all_words


def _write_wav(audio: np.ndarray, path: str) -> None:
    """Write a float32 numpy array as a 16kHz mono WAV file.

    Uses ffmpeg to ensure a proper WAV header.
    """
    pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", "16000",
            "-ac", "1",
            "-i", "pipe:0",
            path,
        ],
        input=pcm.tobytes(),
        capture_output=True,
        check=True,
    )


def _parse_timestamp(ts: str | dict | float) -> float:
    """Parse a timestamp from whisper.cpp JSON.

    Formats seen:
    - String: "00:00:01,230" (HH:MM:SS,ms)
    - String: "1.230" (seconds)
    - Dict: {"from": "00:00:00,000", "to": "00:00:00,130"}
    - Float: 1.23
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, dict):
        # Return the 'to' value by default
        return _parse_timestamp(ts.get("to", ts.get("from", 0)))
    if isinstance(ts, str):
        ts = ts.strip()
        # Try HH:MM:SS,ms format
        if ':' in ts:
            parts = ts.replace(',', '.').split(':')
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
        # Try plain seconds
        return float(ts)
    return 0.0


def _extract_tokens(whisper_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract token list from whisper.cpp JSON output.

    The JSON structure from -ojf is:
    {
      "systeminfo": {...},
      "model": {...},
      "params": {...},
      "result": {"language": "fr"},
      "transcription": [
        {
          "timestamps": {"from": "00:00:00,000", "to": "00:00:01,500"},
          "offsets": {"from": 0, "to": 24000},
          "text": "Bonjour comment allez vous",
          "tokens": [
            {"text": " Bon", "timestamps": {"from": "00:00:00,000", "to": "00:00:00,130"}, "offsets": {"from": 0, "to": 130}, "id": 4909, "p": 0.35},
            {"text": "jour", "timestamps": {"from": "00:00:00,130", "to": "00:00:00,260"}, "offsets": {"from": 130, "to": 260}, "id": 2630, "p": 0.95},
            ...
          ]
        }
      ]
    }
    """
    tokens = []
    for segment in whisper_output.get("transcription", []):
        for token in segment.get("tokens", []):
            text = token.get("text", "")
            if not text or text.startswith("[_"):
                continue

            timestamps = token.get("timestamps", {})
            start = _parse_timestamp(timestamps.get("from", 0))
            end = _parse_timestamp(timestamps.get("to", 0))
            prob = token.get("p", 0.0)

            tokens.append({
                "text": text,
                "start": start,
                "end": end,
                "probability": prob,
            })

    return tokens


def _apply_language_vote(
    segments: list[dict[str, Any]],
    min_segment_duration: float = 3.0,
) -> list[dict[str, Any]]:
    """Override minority languages on short segments via majority vote.

    Whisper.cpp can misidentify language on short segments (<3s) because
    there isn't enough audio context. This function:
    1. Finds the majority language (by total speech duration)
    2. For short segments with a minority language, overrides to the majority

    Long segments keep their detected language — they have enough context
    to be reliable, and might genuinely be code-switching.

    Args:
        segments: Output from transcribe_segments.
        min_segment_duration: Segments shorter than this (seconds) get
            their language overridden if it's a minority language.

    Returns:
        Same segments with language possibly overridden.
    """
    from collections import defaultdict

    if not segments:
        return segments

    # Compute language durations
    lang_duration: dict[str, float] = defaultdict(float)
    for seg in segments:
        lang_duration[seg["language"]] += seg["end"] - seg["start"]

    # Find majority language
    if not lang_duration:
        return segments
    majority_lang = max(lang_duration, key=lambda k: lang_duration[k])

    # Override minority languages on short segments
    for seg in segments:
        seg_duration = seg["end"] - seg["start"]
        if seg_duration < min_segment_duration and seg["language"] != majority_lang:
            seg["language_original"] = seg["language"]
            seg["language"] = majority_lang

    return segments

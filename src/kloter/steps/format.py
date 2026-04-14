"""⑦ Formatting final — JSON + Markdown output."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from kloter import __version__


def format_output(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    speech_segments: list[dict[str, Any]],
    audio_path: str,
    audio: Any,
) -> dict[str, Any]:
    """Build the full result dict (JSON-serializable).

    Args:
        words: List of word dicts with speaker, language, alignment scores.
        diar_segments: List of diarization segments.
        speech_segments: List of VAD speech segments.
        audio_path: Path to the original audio file.
        audio: Numpy array of the audio (for duration calculation).

    Returns:
        Complete result dict ready for JSON serialization.
    """
    # Compute per-language duration from words
    lang_duration: dict[str, float] = defaultdict(float)
    for w in words:
        lang = w.get("language", "unknown")
        lang_duration[lang] += w["end"] - w["start"]

    # Build segments (group consecutive words by speaker)
    segments = _build_segments(words)

    return {
        "audio": os.path.basename(audio_path),
        "duration": round(len(audio) / 16000, 1),
        "languages": {
            lang: round(dur, 1)
            for lang, dur in sorted(lang_duration.items(), key=lambda x: -x[1])
        },
        "words": words,
        "segments": segments,
        "diarization": diar_segments,
        "speech_segments": speech_segments,
    }


def to_markdown(result: dict[str, Any]) -> str:
    """Convert result dict to human-readable Markdown.

    Segments are in chronological order (not grouped by speaker),
    showing the conversation flow with alternating speakers.

    Format per line:
        [  0:00 -   0:02] SPEAKER_01:  text here

    Time is fixed-width: [MMM:SS - MMM:SS] padded to 999:99.

    Args:
        result: Output from format_output.

    Returns:
        Markdown string.
    """
    lines: list[str] = []

    # Header
    audio_name = result["audio"]
    duration = result["duration"]
    langs = ", ".join(f"{lang} ({dur}s)" for lang, dur in result["languages"].items())
    n_speakers = len({seg["speaker"] for seg in result["diarization"]})

    lines.append(f"# Transcription — {audio_name}")
    lines.append("")
    lines.append(f"**Duration**: {duration}s | **Languages**: {langs} | **Speakers**: {n_speakers}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Segments in chronological order
    for seg in result["segments"]:
        time_range = f"{_fmt_time_fw(seg['start'])} - {_fmt_time_fw(seg['end'])}"
        speaker = seg["speaker"]
        text = seg["text"]
        lines.append(f"{speaker} | {time_range}")
        lines.append(text)
        lines.append("")

    lines.append("---")

    return "\n".join(lines)


def write_files(
    result: dict[str, Any],
    audio_path: str,
    output_dir: str | None = None,
    fmt: str = "all",
) -> list[Path]:
    """Write output files to disk.

    Naming convention: {basename}.transcription.json / {basename}.transcription.md

    Args:
        result: Output from format_output.
        audio_path: Original audio file path (for basename and default directory).
        output_dir: Target directory (default: same as audio file).
        fmt: "json", "md", or "all".

    Returns:
        List of paths written.
    """
    basename = Path(audio_path).stem
    target_dir = Path(output_dir) if output_dir else Path(audio_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    if fmt in ("all", "json"):
        json_path = target_dir / f"{basename}.transcription.json"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(json_path)

    if fmt in ("all", "md"):
        md_path = target_dir / f"{basename}.transcription.md"
        md_path.write_text(to_markdown(result) + "\n", encoding="utf-8")
        written.append(md_path)

    return written


def _build_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive words by speaker into segments."""
    if not words:
        return []

    segments: list[dict[str, Any]] = []
    current: dict[str, Any] = {
        "start": words[0]["start"],
        "end": words[0]["end"],
        "speaker": words[0]["speaker"],
        "language": words[0].get("language"),
        "text": words[0]["word"],
    }

    for w in words[1:]:
        if w["speaker"] == current["speaker"] and w.get("language") == current.get("language"):
            current["end"] = w["end"]
            current["text"] += " " + w["word"]
        else:
            segments.append(current)
            current = {
                "start": w["start"],
                "end": w["end"],
                "speaker": w["speaker"],
                "language": w.get("language"),
                "text": w["word"],
            }

    segments.append(current)
    return segments


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS (compact)."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _fmt_time_fw(seconds: float) -> str:
    """Format seconds as M:SS.

    Examples:
        0.0   → '0:00'
        5.3   → '0:05'
        63.7  → '1:03'
        3600  → '60:00'
    """
    total_seconds = int(seconds)
    m, s = divmod(total_seconds, 60)
    return f"{m}:{s:02d}"

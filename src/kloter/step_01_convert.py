"""Step 01 — Audio conversion.

Extracts the audio stream from any media file (mp3, mp4, wav, ogg, etc.)
and converts it to 16kHz mono PCM — the format required by all downstream
tools in the pipeline:

  - whisper.cpp: needs a WAV file (pcm_s16le)
  - pyannote (VAD, diarization): needs float32 tensor, 16kHz mono, [-1,1]
  - wav2vec2 (alignment): needs float32 numpy array, 16kHz mono

The converted WAV is saved as a step artifact for whisper-cli to read
directly. The numpy array (float32, normalized) is kept in memory for
pyannote and wav2vec2.

Saves a step file with original and converted metadata in the same schema.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def load_audio(path: str | Path) -> np.ndarray:
    """Extract audio stream and convert to 16kHz mono float32 numpy array.

    ffmpeg: any format → pcm_s16le 16kHz mono (raw bytes, -ac 1 -ar 16000)
    numpy: int16 → float32 / 32768 to normalize to [-1, 1] for pyannote/wav2vec2.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio[22:]  # skip 44-byte WAV header


def save_wav(audio: np.ndarray, path: Path) -> Path:
    """Write the converted audio as a WAV file for whisper-cli."""
    pcm = (audio * 32768).clip(-32768, 32767).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
         "-i", "pipe:0", str(path)],
        input=pcm.tobytes(),
        capture_output=True,
        check=True,
    )
    return path


def probe_audio(path: str | Path) -> dict[str, Any]:
    """Extract relevant audio metadata via ffprobe.

    Keeps only fields useful for a transcription pipeline — format, duration,
    bitrate, size, tags, and stream specs. Numeric strings are converted to
    native types, except under "tags" which stays as strings.
    For multi-stream containers (mp4 etc.), all streams are kept with codec_type.
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    # Convert numeric strings to native types
    raw_fmt = _convert_values(probe.get("format", {}))
    raw_streams = [_convert_values(s) for s in probe.get("streams", [])]

    # ── Filter format ──
    _FMT_KEEP = {"format_name", "format_long_name", "duration", "bit_rate", "size", "tags"}
    fmt = {k: v for k, v in raw_fmt.items() if k in _FMT_KEEP}

    # ── Filter streams ──
    _STREAM_KEEP = {
        "codec_type", "codec_name", "codec_long_name", "sample_rate", "channels",
        "channel_layout", "sample_fmt", "bit_rate", "bits_per_sample", "duration",
    }
    streams = [{k: v for k, v in s.items() if k in _STREAM_KEEP} for s in raw_streams]

    return {"format": fmt, "streams": streams}


def build_step(
    audio_path: Path,
    audio: np.ndarray,
    probe: dict[str, Any],
    wav_path: Path,
) -> dict[str, Any]:
    """Build the step-01 output dict."""
    duration = round(len(audio) / 16000, 3)
    sample_rate = 16000
    bits_per_sample = 16  # ffmpeg outputs pcm_s16le
    num_samples = len(audio)

    return {
        "step": "01_convert",
        "description": "Audio conversion: any format → 16kHz mono PCM",
        "downstream_requirements": {
            "whisper_cpp": "WAV file, pcm_s16le",
            "pyannote_vad": "float32 tensor, 16kHz mono, [-1,1]",
            "pyannote_diarization": "float32 tensor, 16kHz mono, [-1,1]",
            "wav2vec2_alignment": "float32 numpy, 16kHz mono",
        },
        "original": {
            "file": audio_path.name,
            "path": str(audio_path.resolve()),
            "format": _ordered_format(probe["format"]),
            "streams": [_ordered_stream(s) for s in probe["streams"]],
        },
        "converted": {
            "file": wav_path.name,
            "path": str(wav_path.resolve()),
            "format": _ordered_format({
                "format_name": "wav",
                "format_long_name": "WAV / WAVE (Waveform Audio)",
                "duration": duration,
                "bit_rate": sample_rate * bits_per_sample,
                "size": wav_path.stat().st_size if wav_path.exists() else num_samples * (bits_per_sample // 8),
            }),
            "streams": [_ordered_stream({
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "codec_long_name": "PCM signed 16-bit little-endian",
                "sample_rate": sample_rate,
                "channels": 1,
                "channel_layout": "mono",
                "sample_fmt": "s16",
                "bits_per_sample": bits_per_sample,
                "bit_rate": sample_rate * bits_per_sample,
                "duration": duration,
            })],
        },
    }


# ── Internal helpers ──

# Canonical key order for human readability
_FORMAT_ORDER = ["format_name", "format_long_name", "duration", "bit_rate", "size", "tags"]
_STREAM_ORDER = [
    "codec_type", "codec_name", "codec_long_name",
    "duration",
    "sample_rate", "channels", "channel_layout",
    "sample_fmt", "bits_per_sample",
    "bit_rate",
]


def _ordered(d: dict, order: list[str]) -> dict:
    """Return dict with keys in *order*, then any remaining keys alphabetically."""
    ordered = {}
    for k in order:
        if k in d:
            ordered[k] = d[k]
    for k in sorted(d):
        if k not in ordered:
            ordered[k] = d[k]
    return ordered


def _ordered_format(fmt: dict) -> dict:
    return _ordered(fmt, _FORMAT_ORDER)


def _ordered_stream(stream: dict) -> dict:
    return _ordered(stream, _STREAM_ORDER)


# Keys whose entire subtree must stay as strings (tags can contain anything)
_STRING_KEYS = {"tags"}


def _convert_values(obj: Any, preserve_strings: bool = False, parent_key: str = "") -> Any:
    """Recursively convert numeric strings to int/float in a dict/list tree.

    Values under keys in _STRING_KEYS (e.g. "tags") and their descendants
    are never converted — they remain as strings.
    """
    if isinstance(obj, dict):
        new_preserve = preserve_strings or parent_key in _STRING_KEYS
        return {k: _convert_values(v, preserve_strings=new_preserve, parent_key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_values(v, preserve_strings=preserve_strings, parent_key=parent_key) for v in obj]
    if isinstance(obj, str):
        if preserve_strings:
            return obj
        try:
            return int(obj)
        except ValueError:
            pass
        try:
            return float(obj)
        except ValueError:
            pass
    return obj

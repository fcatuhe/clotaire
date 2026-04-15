"""Step 01 — Audio extraction and conversion.

Extracts the audio stream from any media file (mp3, mp4, wav, ogg, etc.)
and converts it to 16kHz mono PCM — the format required by all downstream
tools in the pipeline:

  - whisper.cpp: needs a WAV file (pcm_s16le)
  - pyannote (VAD, diarization): needs float32 tensor, 16kHz mono, [-1,1]
  - wav2vec2 (alignment): needs float32 numpy array, 16kHz mono

The converted WAV is saved as a step artifact for whisper-cli to read
directly. The numpy array (float32, normalized) is kept in memory for
pyannote and wav2vec2.

Both original and converted metadata come from ffprobe — same schema,
same code path, zero drift from ffmpeg naming.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from kloter.steps_io import StepWriter


# ── ffprobe schema ──────────────────────────────────────────────────────────

# Keys to keep from ffprobe output, in human-readable order.
# A key present here = keep it; its position = its order.
# These are the real ffprobe field names — no renaming, no invention.

_FORMAT_KEYS = [
    "format_name", "format_long_name",
    "duration",
    "bit_rate",
    "size",
    "tags",
]

_STREAM_KEYS = [
    "codec_type", "codec_name", "codec_long_name",
    "duration",
    "sample_rate", "channels", "channel_layout",
    "sample_fmt",
    "bits_per_sample",
    "bit_rate",
]


# ── Public API ──────────────────────────────────────────────────────────────

def execute(media_path: Path, writer: StepWriter) -> np.ndarray:
    """Run step 01 end to end.

    Probes the original media file, extracts and converts the audio stream
    to 16kHz mono PCM, saves the WAV as a step artifact, probes the
    converted file, writes the step JSON.

    Returns the audio array (float32, 16kHz mono) for downstream steps.
    """
    original_probe = _probe(media_path)
    audio = _load_audio(media_path)

    wav_path = writer.artifact_path(1, "convert", ".wav")
    _save_wav(audio, wav_path)

    converted_probe = _probe(wav_path)

    step_data = _build_step(media_path, original_probe, wav_path, converted_probe)
    writer.save(1, "convert", step_data)

    return audio


# ── Audio conversion ────────────────────────────────────────────────────────

# WAV header is 44 bytes = 22 int16 samples at 2 bytes each
_WAV_HEADER_SAMPLES = 22

# PCM int16 range: [-32768, 32767], normalized to [-1.0, 1.0]
_PCM_MAX = 32768.0
_PCM_CLIP_MAX = 32767


def _load_audio(path: str | Path) -> np.ndarray:
    """Extract audio stream and convert to 16kHz mono float32 numpy array.

    ffmpeg: any format → pcm_s16le 16kHz mono (raw bytes, -ac 1 -ar 16000)
    numpy: int16 → float32 / _PCM_MAX to normalize to [-1, 1] for pyannote/wav2vec2.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "wav", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / _PCM_MAX
    return audio[_WAV_HEADER_SAMPLES:]


def _save_wav(audio: np.ndarray, path: Path) -> None:
    """Write the converted audio as a WAV file for whisper-cli."""
    pcm = (audio * _PCM_MAX).clip(-_PCM_MAX, _PCM_CLIP_MAX).astype(np.int16)
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
         "-i", "pipe:0", str(path)],
        input=pcm.tobytes(),
        capture_output=True,
        check=True,
    )


# ── ffprobe ─────────────────────────────────────────────────────────────────

def _probe(path: str | Path) -> dict[str, Any]:
    """Probe a media file with ffprobe, returning filtered+ordered metadata.

    Runs ffprobe -show_format -show_streams, keeps only the keys listed in
    _FORMAT_KEYS and _STREAM_KEYS, orders them for readability, and converts
    numeric strings to native types (except under "tags" which stays as strings).
    """
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = _parse_ffprobe_json(result.stdout)
    return _filter_and_order(raw)


def _filter_and_order(raw: dict[str, Any]) -> dict[str, Any]:
    """Keep only relevant keys from raw ffprobe output, in readable order."""
    return {
        "format": _pick_keys(raw.get("format", {}), _FORMAT_KEYS),
        "streams": [_pick_keys(s, _STREAM_KEYS) for s in raw.get("streams", [])],
    }


def _pick_keys(d: dict, keys: list[str]) -> dict:
    """Return dict with only the keys listed, in that order."""
    return {k: d[k] for k in keys if k in d}


# ── Step output assembly ────────────────────────────────────────────────────

def _build_step(
    media_path: Path,
    original_probe: dict[str, Any],
    wav_path: Path,
    converted_probe: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the step-01 output dict from probed metadata."""
    return {
        "step": "01_convert",
        "description": "Audio extraction and conversion: any format → 16kHz mono PCM",
        "downstream_requirements": _build_downstream_requirements(),
        "original": _build_file_entry(media_path, original_probe),
        "converted": _build_file_entry(wav_path, converted_probe),
    }


def _build_downstream_requirements() -> dict[str, str]:
    """Document what each downstream tool requires from the conversion."""
    return {
        "whisper_cpp": "WAV file, pcm_s16le",
        "pyannote_vad": "float32 tensor, 16kHz mono, [-1,1]",
        "pyannote_diarization": "float32 tensor, 16kHz mono, [-1,1]",
        "wav2vec2_alignment": "float32 numpy, 16kHz mono",
    }


def _build_file_entry(path: Path, probe_data: dict[str, Any]) -> dict[str, Any]:
    """Build an original/converted entry from a file path and its probe data."""
    return {
        "file": path.name,
        "path": str(path.resolve()),
        "format": probe_data["format"],
        "streams": probe_data["streams"],
    }


# ── JSON parsing ────────────────────────────────────────────────────────────

# Keys whose entire subtree must stay as strings (tags can contain anything)
_STRING_KEYS = {"tags"}


def _parse_ffprobe_json(text: str) -> dict[str, Any]:
    """Parse ffprobe JSON, converting numeric strings to native types.

    Values under "tags" and its descendants are never converted — they stay
    as strings (e.g. "260331_1031" must not become an integer).
    """
    data = json.loads(text)
    return _convert_values(data)


def _convert_values(obj: Any, preserve_strings: bool = False, parent_key: str = "") -> Any:
    """Recursively convert numeric strings to int/float in a dict/list tree."""
    if isinstance(obj, dict):
        new_preserve = preserve_strings or parent_key in _STRING_KEYS
        return {k: _convert_values(v, preserve_strings=new_preserve, parent_key=k)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_values(v, preserve_strings=preserve_strings, parent_key=parent_key)
                for v in obj]
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

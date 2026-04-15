"""Step 02 — VAD (Voice Activity Detection).

Uses pyannote segmentation model to detect speech segments, then applies
a 0.3s padding buffer on each side to capture edge words cut off by the
model boundary. Padded segments that overlap are merged.

Reads audio from the WAV artifact produced by step 01.
Saves a rich step file with model info, config, raw model output,
and the padded+merged result.
"""

from __future__ import annotations

import os
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kloter.steps_io import StepWriter

# VAD defaults
_DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
_PAD_SECONDS = 0.3
_MIN_DURATION_ON = 0.1
_MIN_DURATION_OFF = 1.0

# WAV header size in bytes (standard 44-byte header)
_WAV_HEADER_BYTES = 44


# ── Public API ──────────────────────────────────────────────────────────────

def execute(wav_path: Path, writer: StepWriter) -> list[dict[str, float]]:
    """Run step 02 end to end.

    Loads audio from the step-01 WAV artifact, runs VAD, builds the step
    file, writes it.

    Returns the padded speech segments for downstream steps.
    """
    audio, sample_rate = _load_wav(wav_path)

    t0 = time.perf_counter()
    raw_segments, padded_segments, model_info = _detect_speech(audio, sample_rate)
    elapsed = time.perf_counter() - t0

    step_data = _build_step(
        raw_segments, padded_segments, model_info,
        audio, sample_rate, elapsed,
    )
    writer.save(2, "vad", step_data)

    return padded_segments


# ── Audio loading ───────────────────────────────────────────────────────────

def _load_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load audio from a WAV file as float32 numpy array.

    Reads the raw PCM data after the WAV header and normalizes to [-1, 1].
    No external audio library needed — just struct + numpy.

    Returns (audio, sample_rate) where audio is float32, 16kHz mono.
    """
    with open(wav_path, "rb") as f:
        # RIFF header
        _riff = f.read(4)    # "RIFF"
        _size = struct.unpack("<I", f.read(4))[0]
        _wave = f.read(4)    # "WAVE"

        # fmt chunk
        _fmt_id = f.read(4)  # "fmt "
        fmt_size = struct.unpack("<I", f.read(4))[0]
        fmt_data = f.read(fmt_size)
        # WAV fmt fields: audio_format(2), channels(2), sample_rate(4), byte_rate(4), block_align(2), bits_per_sample(2)
        sample_rate = struct.unpack("<I", fmt_data[4:8])[0]

        # Skip any extra chunks (LIST, fact, etc.) until we find "data"
        while True:
            chunk_id = f.read(4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"data":
                break
            f.read(chunk_size)

        pcm = f.read(chunk_size)

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sample_rate


def _audio_to_tensor(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    """Convert numpy audio to pyannote's expected input format."""
    return {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": sample_rate,
    }


# ── VAD detection ────────────────────────────────────────────────────────────

def _detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    hf_token: str | None = None,
    pad: float = _PAD_SECONDS,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    """Run VAD and return (raw_segments, padded_segments, model_info)."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required for pyannote model access")

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    segmentation_model = Model.from_pretrained(
        _DIARIZATION_REPO, subfolder="segmentation", token=hf_token,
    )

    vad = VoiceActivityDetection(segmentation=segmentation_model)
    vad_config = {"min_duration_on": _MIN_DURATION_ON, "min_duration_off": _MIN_DURATION_OFF}
    vad.instantiate(vad_config)

    t0 = time.perf_counter()
    audio_data = _audio_to_tensor(audio, sample_rate)
    vad_result = vad(audio_data)
    inference_time = time.perf_counter() - t0

    audio_duration = len(audio) / sample_rate
    raw_segments = _extract_segments(vad_result)
    padded_segments = _pad_and_merge(raw_segments, pad=pad, audio_duration=audio_duration)

    model_info = {
        "repo": _DIARIZATION_REPO,
        "subfolder": "segmentation",
        "vad_config": vad_config,
        "inference_time_s": round(inference_time, 2),
    }

    return raw_segments, padded_segments, model_info


def _extract_segments(vad_result: Any) -> list[dict[str, float]]:
    """Extract segment list from pyannote VAD result."""
    return [
        {"start": round(s.start, 3), "end": round(s.end, 3)}
        for s in vad_result.itersegments()
    ]


def _pad_and_merge(
    segments: list[dict[str, float]],
    pad: float = 0.3,
    audio_duration: float = 0.0,
) -> list[dict[str, float]]:
    """Pad VAD segments and merge overlapping results.

    Adds *pad* seconds before and after each segment, clamps to [0, audio_duration],
    then merges any segments that now overlap.
    """
    if not segments or pad <= 0:
        return segments

    padded = []
    for s in segments:
        new_start = max(0.0, s["start"] - pad)
        new_end = min(audio_duration, s["end"] + pad)
        padded.append({"start": round(new_start, 3), "end": round(new_end, 3)})

    merged = [padded[0].copy()]
    for s in padded[1:]:
        prev = merged[-1]
        if s["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], s["end"])
        else:
            merged.append(s.copy())

    return merged


# ── Step output assembly ────────────────────────────────────────────────────

def _build_step(
    raw_segments: list[dict[str, float]],
    padded_segments: list[dict[str, float]],
    model_info: dict[str, Any],
    audio: np.ndarray,
    sample_rate: int,
    wall_time_s: float,
) -> dict[str, Any]:
    """Assemble the step-02 output dict."""
    audio_duration = len(audio) / sample_rate

    return {
        "step": "02_vad",
        "description": "Voice Activity Detection using pyannote segmentation model",
        "input": _build_input_summary(audio, sample_rate),
        "config": _build_config(),
        "model": _build_model_info(model_info),
        "timing": _build_timing(model_info, wall_time_s),
        "original_output": _build_original_output(raw_segments, audio_duration),
        "processed_output": _build_processed_output(
            raw_segments, padded_segments, audio_duration,
            model_info["vad_config"]["min_duration_off"],
        ),
    }


def _build_input_summary(audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    return {
        "duration_s": round(len(audio) / sample_rate, 3),
        "num_samples": len(audio),
    }


def _build_config() -> dict[str, Any]:
    return {
        "min_duration_on": _MIN_DURATION_ON,
        "min_duration_off": _MIN_DURATION_OFF,
        "pad_seconds": _PAD_SECONDS,
    }


def _build_model_info(model_info: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": model_info["repo"],
        "subfolder": model_info["subfolder"],
        "vad_hyperparams": model_info["vad_config"],
    }


def _build_timing(model_info: dict[str, Any], wall_time_s: float) -> dict[str, Any]:
    return {
        "inference_s": model_info["inference_time_s"],
        "wall_s": round(wall_time_s, 2),
    }


def _build_original_output(
    segments: list[dict[str, float]],
    audio_duration: float,
) -> dict[str, Any]:
    total_speech = sum(s["end"] - s["start"] for s in segments)
    return {
        "num_segments": len(segments),
        "total_speech_s": round(total_speech, 3),
        "coverage_pct": round(total_speech / audio_duration * 100, 1) if audio_duration else 0,
        "segments": segments,
    }


def _build_processed_output(
    raw_segments: list[dict[str, float]],
    padded_segments: list[dict[str, float]],
    audio_duration: float,
    min_duration_off: float,
) -> dict[str, Any]:
    raw_speech = sum(s["end"] - s["start"] for s in raw_segments)
    padded_speech = sum(s["end"] - s["start"] for s in padded_segments)
    return {
        "description": f"Original segments padded with {_PAD_SECONDS}s buffer on each side, then merged where overlapping (min_duration_off={min_duration_off}s)",
        "num_segments": len(padded_segments),
        "total_speech_s": round(padded_speech, 3),
        "coverage_pct": round(padded_speech / audio_duration * 100, 1) if audio_duration else 0,
        "added_speech_s": round(padded_speech - raw_speech, 3),
        "segments": padded_segments,
    }

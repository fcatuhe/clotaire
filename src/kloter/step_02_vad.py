"""Step 02 — VAD (Voice Activity Detection).

Uses pyannote segmentation model to detect speech segments, then applies
a configurable padding buffer to capture edge words.

Saves a rich step file with model info, config, raw model output,
and the padded+merged result.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

# VAD defaults
_DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"
_PAD_SECONDS = 0.3
_MIN_DURATION_ON = 0.1
_MIN_DURATION_OFF = 1.0


def detect_speech(
    audio: np.ndarray,
    hf_token: str | None = None,
    pad: float = _PAD_SECONDS,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    """Run VAD and return (raw_segments, padded_segments, model_info).

    Args:
        audio: Float32 numpy array, 16kHz mono.
        hf_token: HuggingFace token for gated model access.
        pad: Seconds of padding to add before/after each segment.

    Returns:
        Tuple of:
            - raw_segments: list of {"start", "end"} from the model
            - padded_segments: after padding + merge
            - model_info: dict with model name, config used, etc.
    """
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN required for pyannote model access")

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    segmentation_model = Model.from_pretrained(
        _DIARIZATION_REPO, subfolder="segmentation", token=hf_token,
    )

    vad = VoiceActivityDetection(segmentation=segmentation_model)
    vad_config = {
        "min_duration_on": _MIN_DURATION_ON,
        "min_duration_off": _MIN_DURATION_OFF,
    }
    vad.instantiate(vad_config)

    t0 = time.perf_counter()
    audio_data = {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": 16000,
    }
    vad_result = vad(audio_data)
    elapsed = time.perf_counter() - t0

    raw_segments = [
        {"start": round(s.start, 3), "end": round(s.end, 3)}
        for s in vad_result.itersegments()
    ]

    audio_duration = len(audio) / 16000
    padded_segments = _pad_segments(raw_segments, pad=pad, audio_duration=audio_duration)

    model_info = {
        "model_repo": _DIARIZATION_REPO,
        "model_subfolder": "segmentation",
        "vad_config": vad_config,
        "pad_seconds": pad,
        "processing_time_s": round(elapsed, 2),
    }

    return raw_segments, padded_segments, model_info


def build_step(
    raw_segments: list[dict[str, float]],
    padded_segments: list[dict[str, float]],
    model_info: dict[str, Any],
    audio: np.ndarray,
) -> dict[str, Any]:
    """Build the step-02 output dict."""
    audio_duration = len(audio) / 16000

    # Compute speech coverage stats
    raw_speech = sum(s["end"] - s["start"] for s in raw_segments)
    padded_speech = sum(s["end"] - s["start"] for s in padded_segments)

    return {
        "step": "02_vad",
        "description": "Voice Activity Detection using pyannote segmentation model",
        "input": {
            "duration_s": round(audio_duration, 3),
            "num_samples": len(audio),
        },
        "config": {
            "min_duration_on": _MIN_DURATION_ON,
            "min_duration_off": _MIN_DURATION_OFF,
            "pad_seconds": _PAD_SECONDS,
        },
        "model": {
            "repo": model_info["model_repo"],
            "subfolder": model_info["model_subfolder"],
            "vad_hyperparams": model_info["vad_config"],
        },
        "processing_time_s": model_info["processing_time_s"],
        "original_output": {
            "num_segments": len(raw_segments),
            "total_speech_s": round(raw_speech, 3),
            "coverage_pct": round(raw_speech / audio_duration * 100, 1) if audio_duration else 0,
            "segments": raw_segments,
        },
        "processed_output": {
            "description": f"Original segments padded with {model_info['pad_seconds']}s buffer on each side, then merged where overlapping",
            "num_segments": len(padded_segments),
            "total_speech_s": round(padded_speech, 3),
            "coverage_pct": round(padded_speech / audio_duration * 100, 1) if audio_duration else 0,
            "added_speech_s": round(padded_speech - raw_speech, 3),
            "segments": padded_segments,
        },
    }


def _pad_segments(
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

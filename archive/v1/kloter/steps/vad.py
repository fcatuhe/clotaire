"""② VAD (Voice Activity Detection) — pyannote segmentation from diarization repo."""

from __future__ import annotations

import numpy as np
import torch

from kloter.models.loader import get_pyannote_segmentation, configure_threads


def detect_speech(
    audio: np.ndarray,
    hf_token: str | None = None,
    pad: float = 0.3,
) -> list[dict[str, float]]:
    """Detect speech segments in audio using pyannote segmentation model.

    The segmentation model is bundled inside the speaker-diarization-community-1 repo
    (subfolder='segmentation'). Only ONE gated repo accept is needed.

    VAD boundaries are imprecise at segment edges — short words like "Hein"
    can be cut off.  A small padding (*pad*) is added on each side of every
    detected segment to capture speech that the model boundary missed.
    Padding does not extend into a neighbouring VAD segment.

    Args:
        audio: Float32 numpy array, 16kHz mono.
        hf_token: HuggingFace token for gated model access.
        pad: Seconds of padding to add before/after each segment (default 0.3s).

    Returns:
        List of speech segments: [{"start": float, "end": float}, ...].
    """
    configure_threads()
    segmentation_model = get_pyannote_segmentation(hf_token=hf_token)

    from pyannote.audio.pipelines import VoiceActivityDetection

    vad = VoiceActivityDetection(segmentation=segmentation_model)
    vad.instantiate({
        "min_duration_on": 0.1,
        "min_duration_off": 1.0,
    })

    audio_data = {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": 16000,
    }
    vad_result = vad(audio_data)

    raw_segments = [
        {"start": round(s.start, 3), "end": round(s.end, 3)}
        for s in vad_result.itersegments()
    ]

    return _pad_segments(raw_segments, pad=pad, audio_duration=len(audio) / 16000)


def _pad_segments(
    segments: list[dict[str, float]],
    pad: float = 0.3,
    audio_duration: float = 0.0,
) -> list[dict[str, float]]:
    """Pad VAD segments and merge overlapping results.

    Adds *pad* seconds before and after each segment, then merges any
    segments that now overlap.  Padding stops at the audio boundary (0)
    and does not extend into a neighbouring segment's original space.
    """
    if not segments or pad <= 0:
        return segments

    # Pad each segment, clamped to [0, audio_duration]
    padded = []
    for s in segments:
        new_start = max(0.0, s["start"] - pad)
        new_end = min(audio_duration, s["end"] + pad)
        padded.append({"start": round(new_start, 3), "end": round(new_end, 3)})

    # Merge overlapping/adjacent segments
    merged = [padded[0].copy()]
    for s in padded[1:]:
        prev = merged[-1]
        if s["start"] <= prev["end"]:
            prev["end"] = max(prev["end"], s["end"])
        else:
            merged.append(s.copy())

    return merged

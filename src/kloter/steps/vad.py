"""② VAD (Voice Activity Detection) — pyannote segmentation from diarization repo."""

from __future__ import annotations

import numpy as np
import torch

from kloter.models.loader import get_pyannote_segmentation, configure_threads


def detect_speech(audio: np.ndarray, hf_token: str | None = None) -> list[dict[str, float]]:
    """Detect speech segments in audio using pyannote segmentation model.

    The segmentation model is bundled inside the speaker-diarization-community-1 repo
    (subfolder='segmentation'). Only ONE gated repo accept is needed.

    Args:
        audio: Float32 numpy array, 16kHz mono.
        hf_token: HuggingFace token for gated model access.

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

    return [
        {"start": round(s.start, 3), "end": round(s.end, 3)}
        for s in vad_result.itersegments()
    ]

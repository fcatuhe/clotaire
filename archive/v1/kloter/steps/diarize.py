"""④ Diarization — pyannote speaker diarization."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from kloter.models.loader import get_pyannote_diarization, configure_threads


def diarize(
    audio: np.ndarray,
    hf_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Perform speaker diarization on the full audio.

    Args:
        audio: Float32 numpy array, 16kHz mono (full audio).
        hf_token: HuggingFace token for gated models.
        min_speakers: Minimum number of speakers (optional).
        max_speakers: Maximum number of speakers (optional).

    Returns:
        List of diarization segments: [{"start": float, "end": float, "speaker": str}, ...].
    """
    configure_threads()
    pipeline = get_pyannote_diarization(hf_token=hf_token)

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    audio_data = {
        "waveform": torch.from_numpy(audio).unsqueeze(0),
        "sample_rate": 16000,
    }
    output = pipeline(audio_data, **kwargs)

    return [
        {
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": speaker,
        }
        for turn, _, speaker in output.speaker_diarization.itertracks(yield_label=True)
    ]

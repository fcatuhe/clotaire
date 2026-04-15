"""① Audio conversion — any format → WAV 16kHz mono float32."""

from __future__ import annotations

import subprocess

import numpy as np


def load_audio(path: str) -> np.ndarray:
    """Convert any audio file to WAV 16kHz mono float32 numpy array.

    Args:
        path: Path to audio file (mp3, ogg, m4a, flac, wav, webm…).

    Returns:
        Float32 numpy array, values in [-1, 1], sample rate 16kHz, mono.

    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If ffmpeg fails to convert.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", path,
            "-f", "wav",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-",  # output to stdout
        ],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    # Skip WAV header: 44 bytes = 22 int16 samples
    return audio[22:]

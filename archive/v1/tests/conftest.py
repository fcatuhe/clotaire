"""Shared test fixtures for kloter."""

from __future__ import annotations


import numpy as np
import pytest


@pytest.fixture
def short_silence_audio():
    """Generate 1 second of silence at 16kHz mono float32."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def short_speech_audio():
    """Generate 1 second of synthetic speech-like audio at 16kHz mono float32.

    Not real speech, but has amplitude variation for testing VAD-like behavior.
    """
    rng = np.random.default_rng(42)
    return (rng.standard_normal(16000) * 0.3).astype(np.float32)


@pytest.fixture
def tmp_audio_file(short_silence_audio, tmp_path):
    """Write a WAV file to a temp directory and return its path."""
    import subprocess

    audio = short_silence_audio
    # Convert float32 to int16 for WAV
    pcm = (audio * 32768).astype(np.int16)

    wav_path = tmp_path / "test_audio.wav"
    # Write raw PCM and use ffmpeg to make a proper WAV
    raw_path = tmp_path / "test_audio.raw"
    raw_path.write_bytes(pcm.tobytes())

    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", "16000",
            "-ac", "1",
            "-i", str(raw_path),
            str(wav_path),
        ],
        capture_output=True,
        check=True,
    )

    return str(wav_path)


@pytest.fixture
def sample_words():
    """Sample word dicts for testing match/format steps.
    
    In the real pipeline, match_speakers adds 'speaker' before format_output.
    So for format tests, speakers are included.
    """
    return [
        {"start": 0.0, "end": 0.5, "word": "Bonjour", "probability": 0.9, "align_score": 0.8, "language": "fr", "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 1.0, "word": "comment", "probability": 0.85, "align_score": 0.7, "language": "fr", "speaker": "SPEAKER_00"},
        {"start": 1.0, "end": 1.5, "word": "allez", "probability": 0.88, "align_score": 0.75, "language": "fr", "speaker": "SPEAKER_00"},
        {"start": 1.5, "end": 1.8, "word": "vous", "probability": 0.82, "align_score": 0.6, "language": "fr", "speaker": "SPEAKER_01"},
    ]


@pytest.fixture
def sample_diar_segments():
    """Sample diarization segments for testing match step."""
    return [
        {"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"},
        {"start": 1.2, "end": 2.0, "speaker": "SPEAKER_01"},
    ]

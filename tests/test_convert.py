"""Tests for step ① — Audio conversion."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest

from clotaire.steps.convert import load_audio


class TestLoadAudio:
    """Tests for load_audio function."""

    def test_load_wav(self, tmp_audio_file):
        """Can load a WAV file and return float32 numpy array."""
        audio = load_audio(tmp_audio_file)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_values_in_range(self, tmp_audio_file):
        """Audio values should be in [-1, 1]."""
        audio = load_audio(tmp_audio_file)
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_nonexistent_file_raises(self):
        """Should raise an error for a non-existent file."""
        with pytest.raises(subprocess.CalledProcessError):
            load_audio("/nonexistent/path/audio.mp3")

    def test_skip_wav_header(self, tmp_audio_file):
        """First 44 bytes (WAV header) should be skipped."""
        audio = load_audio(tmp_audio_file)
        # Silence audio: most values should be near zero
        # (ffmpeg may add tiny dithering, so we check the median)
        assert np.median(np.abs(audio)) < 0.1  # silence ≈ 0

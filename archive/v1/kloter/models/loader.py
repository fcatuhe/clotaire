"""Model loading, caching, and resource management.

Centralized model lifecycle:
- whisper.cpp ggml model: downloaded from HuggingFace, path cached
- Pyannote segmentation: bundled inside speaker-diarization-community-1,
  shared by VAD (step ②) and diarization (step ④)
- Pyannote diarization pipeline: loaded once
- Wav2vec2 alignment models: one per language, cached
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import torch

# ── Shared model cache ──
_cache: dict[str, Any] = {}

# HuggingFace repo that contains pyannote models
_DIARIZATION_REPO = "pyannote/speaker-diarization-community-1"

# whisper.cpp model config
_WHISPER_CPP_REPO = "ggerganov/whisper.cpp"
_WHISPER_CPP_MODEL = "ggml-large-v3.bin"  # f16, non-quantized, largest, latest


def _resolve_token(hf_token: str | None = None) -> str:
    """Resolve HF token from arg or env var."""
    hf_token = hf_token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HuggingFace token required for model downloads. "
            "Set HF_TOKEN env var or pass --hf-token."
        )
    return hf_token


def get_whisper_model_path(hf_token: str | None = None) -> str:
    """Download (if needed) and return path to whisper.cpp ggml model.

    Downloads from ggerganov/whisper.cpp on HuggingFace.
    Cached in ~/.cache/huggingface/hub/.

    Args:
        hf_token: HuggingFace token for download.

    Returns:
        Absolute path to the ggml model file.
    """
    key = "whisper:cpp_model_path"
    if key not in _cache:
        hf_token = _resolve_token(hf_token)
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            _WHISPER_CPP_REPO,
            _WHISPER_CPP_MODEL,
            token=hf_token,
        )
        _cache[key] = path
    return _cache[key]


def get_whisper_cli_path() -> str:
    """Return path to whisper-cli binary.

    Checks for whisper-cli on PATH.

    Returns:
        Path to whisper-cli binary.

    Raises:
        FileNotFoundError: If whisper-cli is not found on PATH.
    """
    key = "whisper:cli_path"
    if key not in _cache:
        cli_path = shutil.which("whisper-cli")
        if cli_path is None:
            raise FileNotFoundError(
                "whisper-cli not found on PATH. "
                "Install whisper.cpp: https://github.com/ggerganov/whisper.cpp"
            )
        _cache[key] = cli_path
    return _cache[key]


def get_pyannote_segmentation(hf_token: str | None = None):
    """Load and cache the segmentation model bundled inside the diarization repo.

    Uses subfolder='segmentation' from speaker-diarization-community-1.
    Only ONE gated repo accept is needed.

    Args:
        hf_token: HuggingFace token for gated model access.

    Returns:
        Pyannote segmentation model.
    """
    key = "pyannote:segmentation"
    if key not in _cache:
        hf_token = _resolve_token(hf_token)
        from pyannote.audio import Model

        _cache[key] = Model.from_pretrained(
            _DIARIZATION_REPO,
            subfolder="segmentation",
            token=hf_token,
        )
    return _cache[key]


def get_pyannote_diarization(hf_token: str | None = None):
    """Load and cache the full pyannote diarization pipeline.

    Args:
        hf_token: HuggingFace token for gated model access.

    Returns:
        Pyannote diarization pipeline.
    """
    key = "pyannote:diarization"
    if key not in _cache:
        hf_token = _resolve_token(hf_token)
        from pyannote.audio import Pipeline as PyannotePipeline

        _cache[key] = PyannotePipeline.from_pretrained(
            _DIARIZATION_REPO,
            token=hf_token,
        )
    return _cache[key]


def get_align_model(lang: str):
    """Load and cache wav2vec2 alignment model for a given language.

    Args:
        lang: Language code (e.g. "fr", "en", "de").

    Returns:
        Tuple of (model, metadata) for whisperx alignment.
    """
    key = f"align:{lang}"
    if key not in _cache:
        from whisperx.alignment import load_align_model

        model, metadata = load_align_model(lang, "cpu")
        _cache[key] = (model, metadata)
    return _cache[key]


def configure_threads(whisper_threads: int = 8, pyannote_threads: int = 4) -> None:
    """Set CPU thread counts for parallel execution.

    Args:
        whisper_threads: Threads for whisper.cpp (default: 8).
        pyannote_threads: Threads for pyannote/wav2vec2 (default: 4).
    """
    torch.set_num_threads(pyannote_threads)


def clear_cache() -> None:
    """Release all cached models (free memory)."""
    _cache.clear()

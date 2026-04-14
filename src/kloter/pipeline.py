"""Pipeline orchestrator — composes steps, manages parallelism."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from kloter.steps.convert import load_audio
from kloter.steps.vad import detect_speech
from kloter.steps.whisper import transcribe_segments
from kloter.steps.diarize import diarize
from kloter.steps.align import align_words
from kloter.steps.match import match_speakers
from kloter.steps.format import format_output


def run(
    audio_path: str,
    hf_token: str | None = None,
    max_speakers: int | None = None,
    min_speakers: int | None = None,
    max_languages: int = 3,
) -> dict[str, Any]:
    """Run the full pipeline. Returns the result dict (JSON-serializable)."""

    hf_token = hf_token or os.environ.get("HF_TOKEN")

    # ① Conversion
    audio = load_audio(audio_path)

    # ② VAD
    speech_segments = detect_speech(audio, hf_token=hf_token)

    # ③ + ④ in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_future = pool.submit(
            transcribe_segments, audio, speech_segments, hf_token=hf_token,
        )
        diar_future = pool.submit(
            diarize,
            audio,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        whisper_result = whisper_future.result()
        diar_segments = diar_future.result()

    # ③bis + ⑤ Wav2vec2 alignment (per-segment language, cached models)
    # align_words attaches language + align_score to each word (subsumes ③bis)
    aligned_words = align_words(whisper_result, audio, max_languages=max_languages)

    # ⑥ Speaker matching
    final_words = match_speakers(aligned_words, diar_segments)

    # ⑦ Format output
    result = format_output(final_words, diar_segments, speech_segments, audio_path, audio)

    return result

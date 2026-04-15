"""Pipeline orchestrator — composes steps, manages parallelism."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
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
    save_steps: str | None = None,
) -> dict[str, Any]:
    """Run the full pipeline. Returns the result dict (JSON-serializable).

    Args:
        audio_path: Path to the audio file.
        hf_token: HuggingFace token for gated models.
        max_speakers: Maximum number of speakers for diarization.
        min_speakers: Minimum number of speakers for diarization.
        max_languages: Maximum number of wav2vec2 models to load.
        save_steps: If set, directory where intermediate step outputs
            are saved as numbered JSON files (01_convert.json, 02_vad.json, etc.).
    """

    hf_token = hf_token or os.environ.get("HF_TOKEN")

    step_saver = _StepSaver(save_steps, Path(audio_path).stem)

    # ① Conversion
    audio = load_audio(audio_path)
    step_saver.save("01_convert", {
        "audio": os.path.basename(audio_path),
        "duration": round(len(audio) / 16000, 1),
        "sample_rate": 16000,
    })

    # ② VAD
    speech_segments = detect_speech(audio, hf_token=hf_token)
    step_saver.save("02_vad", speech_segments)

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

    step_saver.save("03_transcription", whisper_result)
    step_saver.save("04_diarization", diar_segments)

    # ③bis + ⑤ Wav2vec2 alignment (per-segment language, cached models)
    # align_words attaches language + align_score to each word (subsumes ③bis)
    # Diarization segments are passed so alignment can split at speaker changes,
    # reducing timing drift in merged VAD segments.
    aligned_words = align_words(
        whisper_result, audio,
        max_languages=max_languages,
        diar_segments=diar_segments,
    )
    step_saver.save("05_alignment", aligned_words)

    # ⑥ Speaker matching
    final_words = match_speakers(aligned_words, diar_segments)
    step_saver.save("06_matching", final_words)

    # ⑦ Format output
    result = format_output(final_words, diar_segments, speech_segments, audio_path, audio)
    step_saver.save("07_format", result)

    return result


class _StepSaver:
    """Save intermediate pipeline step outputs to numbered JSON files."""

    def __init__(self, output_dir: str | None, basename: str) -> None:
        self._dir = Path(output_dir) if output_dir else None
        self._basename = basename

    def save(self, step_name: str, data: Any) -> None:
        if self._dir is None:
            return
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{step_name}.{self._basename}.json"
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  step saved: {path}", file=__import__("sys").stderr)

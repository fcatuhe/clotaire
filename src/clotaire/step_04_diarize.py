"""Step 04 — Speaker diarization with pyannote.

Runs pyannote speaker diarization on the canonical WAV produced by step 01.
The diarization is performed on the full audio and outputs speaker turns only:
start, end, and speaker label.

Reads audio from the WAV artifact produced by step 01.
Saves the numbered step JSON and preserves the raw diarization output under
04_diarize.raw/.
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from clotaire.steps_io import StepWriter

_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"
_MIN_SPEAKERS: int | None = None
_MAX_SPEAKERS: int | None = None


def execute(wav_path: Path, writer: StepWriter) -> dict[str, Any]:
    """Run step 04 end to end."""
    t0 = time.perf_counter()

    stdout_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stdout_buffer):
        diarization_output = _run_diarization(
            wav_path,
            min_speakers=_MIN_SPEAKERS,
            max_speakers=_MAX_SPEAKERS,
        )

    wall_time_s = time.perf_counter() - t0
    annotation = _extract_annotation(diarization_output)
    annotation_exclusive = _extract_annotation_exclusive(diarization_output)
    turns = _build_turns(annotation.itertracks(yield_label=True))
    turns_exclusive = _build_turns(annotation_exclusive.itertracks(yield_label=True))
    step_data = _build_step(turns, turns_exclusive, wall_time_s)

    _save_raw_artifacts(writer, annotation, annotation_exclusive, stdout_buffer.getvalue())
    writer.save(4, "diarize", step_data)
    return step_data


@lru_cache(maxsize=1)
def _load_pipeline() -> Any:
    """Load the pyannote diarization pipeline once."""
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(_DIARIZATION_MODEL, token=_resolve_hf_token())


def _resolve_hf_token() -> str:
    """Resolve the Hugging Face token from the environment."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    raise ValueError(
        "Hugging Face token required for pyannote diarization. "
        "Set HF_TOKEN or HUGGINGFACE_TOKEN."
    )


def _run_diarization(
    wav_path: Path,
    min_speakers: int | None,
    max_speakers: int | None,
) -> Any:
    """Run pyannote diarization on the canonical WAV."""
    pipeline = _load_pipeline()
    waveform, sample_rate = _load_audio(wav_path)

    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    return pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)


def _load_audio(wav_path: Path) -> tuple[Any, int]:
    """Load mono audio from the canonical WAV file."""
    import numpy as np
    import torch

    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        pcm = wav_file.readframes(wav_file.getnframes())

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    waveform = torch.from_numpy(audio).unsqueeze(0)
    return waveform, sample_rate


def _extract_annotation(diarization_output: Any) -> Any:
    """Return the default non-exclusive pyannote Annotation when available."""
    if hasattr(diarization_output, "speaker_diarization"):
        return diarization_output.speaker_diarization
    return diarization_output


def _extract_annotation_exclusive(diarization_output: Any) -> Any:
    """Return the exclusive pyannote Annotation when available."""
    if hasattr(diarization_output, "exclusive_speaker_diarization"):
        return diarization_output.exclusive_speaker_diarization
    return _extract_annotation(diarization_output)


def _build_turns(tracks: Iterable[tuple[Any, Any, str]]) -> list[dict[str, Any]]:
    """Convert pyannote tracks into normalized diarization turns."""
    turns: list[dict[str, Any]] = []
    for index, (turn, _, speaker) in enumerate(tracks, start=1):
        start_ms = round(float(turn.start) * 1000)
        end_ms = round(float(turn.end) * 1000)
        turns.append(
            {
                "id": f"turn_{index:04d}",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": speaker,
                "duration_ms": max(0, end_ms - start_ms),
            }
        )
    return turns


def _build_speakers(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build per-speaker summary stats from diarization turns."""
    stats: dict[str, dict[str, int]] = {}
    for turn in turns:
        speaker = turn["speaker"]
        if speaker not in stats:
            stats[speaker] = {"num_turns": 0, "duration_ms": 0}
        stats[speaker]["num_turns"] += 1
        stats[speaker]["duration_ms"] += turn["duration_ms"]

    return [
        {
            "id": speaker,
            "num_turns": data["num_turns"],
            "duration_s": round(data["duration_ms"] / 1000, 3),
        }
        for speaker, data in sorted(stats.items())
    ]


def _build_step(
    turns: list[dict[str, Any]],
    turns_exclusive: list[dict[str, Any]],
    wall_time_s: float,
) -> dict[str, Any]:
    """Assemble the step-04 JSON output."""
    return {
        "step": "04_diarize",
        "description": "Pyannote speaker diarization on the step-01 WAV",
        "model": {
            "name": _DIARIZATION_MODEL,
            "type": "speaker-diarization",
        },
        "config": {
            "min_speakers": _MIN_SPEAKERS,
            "max_speakers": _MAX_SPEAKERS,
        },
        "result": {
            "num_speakers": len({turn["speaker"] for turn in turns}),
            "num_turns": len(turns),
            "num_turns_exclusive": len(turns_exclusive),
            "speakers": _build_speakers(turns),
            "turns": turns,
            "turns_exclusive": turns_exclusive,
        },
        "timing": {
            "wall_s": round(wall_time_s, 2),
        },
    }


def _save_raw_artifacts(
    writer: StepWriter,
    annotation: Any,
    annotation_exclusive: Any,
    pyannote_output: str,
) -> None:
    """Save raw diarization outputs and merged stdout/stderr."""
    raw_dir = writer.steps_dir / "04_diarize.raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with (raw_dir / "diarization.rttm").open("w", encoding="utf-8") as fp:
        annotation.write_rttm(fp)
    with (raw_dir / "diarization_exclusive.rttm").open("w", encoding="utf-8") as fp:
        annotation_exclusive.write_rttm(fp)
    (raw_dir / "pyannote_stdout.txt").write_text(pyannote_output, encoding="utf-8")

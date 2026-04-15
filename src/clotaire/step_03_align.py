"""Step 03 — wav2vec2 forced alignment on the canonical WAV.

Aligns the transcript produced by step 02 against the WAV produced by step 01.
Whisper remains the source of truth for the text; wav2vec2 only refines word
boundaries on the audio.

Alignment runs per segment, using torchaudio's multilingual MMS forced-alignment
bundle. Punctuation is preserved in the output but removed from the acoustic
alignment target. If alignment fails for a segment, the step falls back to the
word timings inherited from step 02.
"""

from __future__ import annotations

import json
import time
import unicodedata
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from clotaire.steps_io import StepWriter

_ALIGNER_BUNDLE = "MMS_FA"


def execute(
    wav_path: Path,
    transcription_step: dict[str, Any],
    writer: StepWriter,
) -> dict[str, Any]:
    """Run step 03 end to end.

    Reads the canonical WAV from step 01 and the segments/items produced by
    step 02, enriches them with wav2vec2 alignment, saves raw debug artifacts,
    and writes the numbered JSON step file.
    """
    t0 = time.perf_counter()
    waveform, sample_rate = _load_audio(wav_path)

    model_info = _model_info()
    alignment_debug: list[dict[str, Any]] = []

    transcription = transcription_step.get("transcription", {})
    voice_ranges = transcription.get("whisper", {}).get("voice_ranges", [])
    segments = transcription.get("segments", [])

    for voice_range in voice_ranges:
        range_segments = [seg for seg in segments if seg.get("voice_range_id") == voice_range["id"]]
        _align_voice_range(voice_range, range_segments, waveform, sample_rate)
        alignment_debug.append({
            "voice_range_id": voice_range["id"],
            "start_ms": voice_range["start_ms"],
            "end_ms": voice_range["end_ms"],
            "num_segments": len(range_segments),
            "segment_ids": [seg["id"] for seg in range_segments],
        })

    for segment in segments:
        if "wav2vec2" not in segment:
            _apply_fallback(segment, reason="segment_has_no_voice_range", voice_range_id=None)

    wall_time_s = time.perf_counter() - t0
    _save_raw_artifacts(writer, model_info, alignment_debug)

    step_data = _build_step(
        wav_path=wav_path,
        transcription_step=transcription_step,
        transcription=transcription,
        model_info=model_info,
        segments=segments,
        wall_time_s=wall_time_s,
    )
    path = writer.save(3, "align", step_data)
    from clotaire.step_02_transcribe import _compact_voice_ranges_in_json
    path.write_text(
        _compact_voice_ranges_in_json(path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return step_data


@lru_cache(maxsize=1)
def _load_aligner() -> dict[str, Any]:
    """Load the torchaudio MMS forced-alignment backend once."""
    import torch
    import torchaudio

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model()
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    return {
        "torch": torch,
        "bundle": bundle,
        "model": model,
        "tokenizer": tokenizer,
        "aligner": aligner,
        "sample_rate": bundle.sample_rate,
    }


def _model_info() -> dict[str, Any]:
    """Describe the alignment backend, or record why it is unavailable."""
    try:
        aligner = _load_aligner()
    except Exception as exc:  # pragma: no cover - exercised in execute fallback
        return {
            "name": _ALIGNER_BUNDLE,
            "type": "ctc-forced-alignment",
            "status": "unavailable",
            "error": str(exc),
        }

    return {
        "name": _ALIGNER_BUNDLE,
        "type": "ctc-forced-alignment",
        "status": "ready",
        "sample_rate": aligner["sample_rate"],
        "labels": list(aligner["bundle"].get_labels(star=None)),
    }


def _load_audio(wav_path: Path) -> tuple[Any, int]:
    """Load mono audio from the canonical WAV file.

    Step 01 guarantees PCM s16le WAV, so we can use the stdlib ``wave`` module
    and avoid optional torchaudio decoding backends.
    """
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


def _align_voice_range(
    voice_range: dict[str, Any],
    segments: list[dict[str, Any]],
    waveform: Any,
    sample_rate: int,
) -> None:
    """Align all segments assigned to one voice range in a single pass."""
    if not segments:
        return

    lexical_items: list[dict[str, Any]] = []
    for segment in segments:
        if not segment.get("items"):
            segment["wav2vec2"] = {
                "status": "skipped",
                "reason": "segment_has_no_items",
                "start_ms": segment["whisper"]["start_ms"],
                "end_ms": segment["whisper"]["end_ms"],
                "voice_range_id": voice_range["id"],
            }
            continue
        for item in segment.get("items", []):
            if item.get("type") != "word":
                continue
            normalized = _normalize_for_alignment(item["text"])
            if normalized:
                lexical_items.append({
                    "segment": segment,
                    "item": item,
                    "normalized": normalized,
                })

    if not lexical_items:
        for segment in segments:
            _apply_fallback(segment, reason="voice_range_has_no_alignable_items", voice_range_id=voice_range["id"])
        return

    try:
        _run_alignment(voice_range, lexical_items, waveform, sample_rate)
        for segment in segments:
            _anchor_segment_punctuation(segment)
            segment["wav2vec2"] = _build_segment_wav2vec2(
                segment,
                status="aligned",
                voice_range_id=voice_range["id"],
            )
    except Exception as exc:
        for segment in segments:
            _apply_fallback(segment, reason=str(exc), voice_range_id=voice_range["id"])


def _run_alignment(
    voice_range: dict[str, Any],
    lexical_items: list[dict[str, Any]],
    waveform: Any,
    sample_rate: int,
) -> None:
    """Run wav2vec2 forced alignment for one voice range."""
    aligner = _load_aligner()
    audio_slice, slice_start_ms, slice_end_ms = _slice_audio(
        waveform,
        sample_rate,
        voice_range["start_ms"],
        voice_range["end_ms"],
        pad_ms=0,
    )

    if sample_rate != aligner["sample_rate"]:
        torchaudio = __import__("torchaudio")
        audio_slice = torchaudio.functional.resample(audio_slice, sample_rate, aligner["sample_rate"])

    emission, _ = aligner["model"](audio_slice)
    spans_per_word = aligner["aligner"](
        emission[0],
        aligner["tokenizer"]([entry["normalized"] for entry in lexical_items]),
    )

    ms_per_frame = (slice_end_ms - slice_start_ms) / max(emission.shape[1], 1)
    for entry, spans in zip(lexical_items, spans_per_word):
        item = entry["item"]
        if not spans:
            item["wav2vec2"] = {
                "start_ms": item["whisper"]["start_ms"],
                "end_ms": item["whisper"]["end_ms"],
                "status": "fallback",
                "fallback_reason": "unaligned_word",
            }
            continue
        start_frame = spans[0].start
        end_frame = spans[-1].end
        item["wav2vec2"] = {
            "start_ms": slice_start_ms + round(start_frame * ms_per_frame),
            "end_ms": slice_start_ms + round(end_frame * ms_per_frame),
            "align_score": round(sum(span.score for span in spans) / len(spans), 4),
            "normalized_text": entry["normalized"],
            "status": "aligned",
        }


def _slice_audio(
    waveform: Any,
    sample_rate: int,
    start_ms: int,
    end_ms: int,
    pad_ms: int,
) -> tuple[Any, int, int]:
    """Extract a padded segment slice from the full waveform."""
    full_duration_ms = round(waveform.shape[1] * 1000 / sample_rate)
    slice_start_ms = max(0, start_ms - pad_ms)
    slice_end_ms = min(full_duration_ms, end_ms + pad_ms)
    start_sample = round(slice_start_ms * sample_rate / 1000)
    end_sample = round(slice_end_ms * sample_rate / 1000)
    return waveform[:, start_sample:end_sample], slice_start_ms, slice_end_ms


def _apply_fallback(segment: dict[str, Any], reason: str, voice_range_id: str | None) -> None:
    """Fill wav2vec2 fields from whisper timings when alignment fails."""
    for item in segment.get("items", []):
        item["wav2vec2"] = {
            "start_ms": item["whisper"]["start_ms"],
            "end_ms": item["whisper"]["end_ms"],
            "status": "fallback",
            "fallback_reason": reason,
        }
    segment["wav2vec2"] = _build_segment_wav2vec2(
        segment,
        status="fallback",
        reason=reason,
        voice_range_id=voice_range_id,
    )


def _anchor_segment_punctuation(segment: dict[str, Any]) -> None:
    """Anchor punctuation items to the previous lexical item's end."""
    last_lexical_end_ms: int | None = None
    for item in segment.get("items", []):
        if item.get("type") == "word":
            if "wav2vec2" not in item:
                item["wav2vec2"] = {
                    "start_ms": item["whisper"]["start_ms"],
                    "end_ms": item["whisper"]["end_ms"],
                    "status": "fallback",
                    "fallback_reason": "missing_word_alignment",
                }
            last_lexical_end_ms = item["wav2vec2"]["end_ms"]
            continue

        if last_lexical_end_ms is None:
            item["wav2vec2"] = {
                "start_ms": item["whisper"]["start_ms"],
                "end_ms": item["whisper"]["end_ms"],
                "status": "fallback",
                "fallback_reason": "no_previous_lexical_item",
            }
        else:
            item["wav2vec2"] = {
                "start_ms": last_lexical_end_ms,
                "end_ms": last_lexical_end_ms,
                "status": "anchored_to_previous_word",
            }


def _normalize_for_alignment(text: str) -> str:
    """Normalize display text into an alignable lexical form.

    Policy:
      - lowercase
      - strip accents/diacritics
      - keep letters, digits and apostrophes
      - drop punctuation, including sentence punctuation
      - collapse repeated whitespace
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    chars: list[str] = []
    for ch in text:
        if ch.isalnum() or ch == "'":
            chars.append(ch)
        else:
            chars.append(" ")
    normalized = "".join(chars)
    normalized = " ".join(normalized.split())
    normalized = normalized.replace(" ", "")
    return normalized.strip("'")


def _is_punctuation_word(text: str) -> bool:
    """Return True when a display item contains no letters or digits."""
    stripped = text.strip()
    return bool(stripped) and all(not ch.isalnum() for ch in stripped if not ch.isspace())


def _build_segment_wav2vec2(
    segment: dict[str, Any],
    status: str,
    reason: str | None = None,
    voice_range_id: str | None = None,
) -> dict[str, Any]:
    """Build the segment-level wav2vec2 summary block."""
    lexical_items = [item for item in segment.get("items", []) if item.get("type") == "word"]
    aligned_items = [item for item in lexical_items if item.get("wav2vec2", {}).get("status") == "aligned"]
    timing_items = aligned_items or lexical_items

    payload = {
        "status": status,
        "voice_range_id": voice_range_id,
        "start_ms": timing_items[0]["wav2vec2"]["start_ms"] if timing_items else segment["whisper"]["start_ms"],
        "end_ms": timing_items[-1]["wav2vec2"]["end_ms"] if timing_items else segment["whisper"]["end_ms"],
        "num_aligned_items": len(aligned_items),
        "num_fallback_items": sum(
            1 for item in lexical_items if item.get("wav2vec2", {}).get("status") == "fallback"
        ),
    }
    if reason:
        payload["reason"] = reason
    return payload


def _save_raw_artifacts(
    writer: StepWriter,
    model_info: dict[str, Any],
    alignment_debug: list[dict[str, Any]],
) -> None:
    """Save raw alignment debug artifacts to 03_align.raw/."""
    raw_dir = writer.steps_dir / "03_align.raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "model_info.json").write_text(
        json.dumps(model_info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "alignment_debug.json").write_text(
        json.dumps(alignment_debug, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _build_step(
    wav_path: Path,
    transcription_step: dict[str, Any],
    transcription: dict[str, Any],
    model_info: dict[str, Any],
    segments: list[dict[str, Any]],
    wall_time_s: float,
) -> dict[str, Any]:
    """Assemble the step-03 JSON output."""
    lexical_items = [
        item
        for segment in segments
        for item in segment.get("items", [])
        if item.get("type") == "word"
    ]
    aligned_items = [item for item in lexical_items if item.get("wav2vec2", {}).get("status") == "aligned"]
    fallback_items = [item for item in lexical_items if item.get("wav2vec2", {}).get("status") == "fallback"]

    return {
        "step": "03_align",
        "description": "wav2vec2 CTC forced alignment on the step-01 WAV using the step-02 transcript",
        "model": model_info,
        "config": {
            "alignment_unit": "one wav2vec2 pass per voice range",
            "item_probability": "item.whisper.probability = mean(source token p)",
            "alignment_text": "lowercase + accent stripping + punctuation removed, apostrophes kept",
            "punctuation_policy": "preserve in items, exclude from acoustic target, anchor to previous lexical item",
        },
        "input": {
            "wav_path": str(wav_path.resolve()),
            "transcription_step": transcription_step.get("step", "02_transcribe"),
            "language": transcription.get("whisper", {}).get("language", "unknown"),
            "num_segments": len(transcription.get("segments", [])),
        },
        "result": {
            "num_segments": len(segments),
            "num_items": sum(len(segment.get("items", [])) for segment in segments),
            "num_aligned_items": len(aligned_items),
            "num_fallback_items": len(fallback_items),
            "num_fallback_segments": sum(
                1 for segment in segments if segment.get("wav2vec2", {}).get("status") == "fallback"
            ),
        },
        "transcription": {
            **transcription,
            "segments": segments,
        },
        "timing": {
            "wall_s": round(wall_time_s, 2),
        },
    }

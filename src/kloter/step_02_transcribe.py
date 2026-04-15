"""Step 02 — Transcription with built-in VAD.

Runs whisper-cli with Silero VAD to transcribe the WAV produced by step 01.
The VAD skips silence internally, so we get both voice_ranges (where speech
was detected) and segments (transcribed text with word-level timestamps
and confidence scores).

Reads audio from the WAV artifact produced by step 01.
Saves a rich step file with model info, VAD config, voice ranges,
and full transcription with token-level detail.

Raw whisper-cli outputs (JSON + stderr) are preserved in a
<steps_dir>/02_transcribe.raw/ subfolder for debugging/replay.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from kloter.steps_io import StepWriter

# ── Defaults ────────────────────────────────────────────────────────────────

_WHISPER_MODEL = "ggml-large-v3.bin"
_VAD_MODEL = "ggml-silero-v6.2.0.bin"
_VAD_THRESHOLD = 0.5
_VAD_MIN_SILENCE_MS = 1000
_VAD_MIN_SPEECH_MS = 100
_VAD_SPEECH_PAD_MS = 100
_VAD_SAMPLES_OVERLAP = 0.0
_LANGUAGE = "auto"

# ── Regex patterns for stderr parsing ────────────────────────────────────────

# VAD segment: "VAD segment 0: start = 0.00, end = 10.15 (duration: 10.15)"
_RE_VAD_SEGMENT = re.compile(
    r"VAD segment \d+: start = ([\d.]+), end = ([\d.]+)"
)

# Timing: "whisper_print_timings:   encode time = 90453.16 ms /     2 runs"
_RE_TIMING = re.compile(
    r"whisper_print_timings:\s+(\w[\w ]*?)\s*=\s*([\d.]+)\s+ms"
)

# Language: "auto-detected language: fr (p = 0.971764)"
_RE_LANGUAGE = re.compile(
    r"auto-detected language: (\w+) \(p = ([\d.]+)\)"
)

# Model loading path: "loading model from '/path/to/ggml-large-v3.bin'"
_RE_MODEL_PATH = re.compile(
    r"loading model from '(.+)'"
)

# Model type: "type = 5 (large v3)"
_RE_MODEL_TYPE = re.compile(
    r"type\s*=\s*\d+\s+\((.+?)\)"
)

# ftype: "ftype = 1"
_RE_FTYPE = re.compile(
    r"ftype\s*=\s*(\d+)"
)

# qntvr: "qntvr = 0"
_RE_QNTVR = re.compile(
    r"qntvr\s*=\s*(\d+)"
)

# Audio duration: "(428408 samples, 26.8 sec)"
_RE_AUDIO_DURATION = re.compile(
    r"(\d+\.\d+)\s+sec\)"
)

# ── ftype / qntvr lookup tables ──────────────────────────────────────────────

_FTYPE_NAMES = {
    0: "Q4_0",
    1: "F16",
    2: "Q4_1",
    3: "Q4_2",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
}

_QNTVR_NAMES = {
    0: "none",
    1: "quantized",
}

_CONTAINER_NAMES = {
    ".bin": "ggml",
    ".gguf": "gguf",
}


# ── Public API ──────────────────────────────────────────────────────────────

def execute(wav_path: Path, writer: StepWriter) -> dict[str, Any]:
    """Run step 02 end to end.

    Runs whisper-cli with VAD on the step-01 WAV artifact, parses the
    JSON output + stderr metadata, builds and saves the step file.
    Preserves raw whisper outputs in 02_transcribe.raw/ subfolder.

    Returns the step data dict for downstream steps.
    """
    t0 = time.perf_counter()
    whisper_json, stderr_text = _run_whisper(wav_path)
    elapsed = time.perf_counter() - t0

    # Save raw artifacts for debugging / replay
    _save_raw_artifacts(writer, whisper_json, stderr_text)

    voice_ranges = _parse_voice_ranges(stderr_text)
    timings = _parse_timings(stderr_text)
    audio_duration_ms = _parse_audio_duration(stderr_text)
    lang_info = _parse_language(stderr_text)

    step_data = _build_step(
        whisper_json, stderr_text, voice_ranges, timings,
        audio_duration_ms, lang_info, elapsed,
    )
    writer.save(2, "transcribe", step_data)

    return step_data


# ── Whisper invocation ─────────────────────────────────────────────────────

def _run_whisper(wav_path: Path) -> tuple[dict[str, Any], str]:
    """Run whisper-cli with VAD and full JSON output.

    Returns (parsed_json, stderr_text).
    """
    cmd = [
        "whisper-cli",
        "-m", str(_resolve_model(_WHISPER_MODEL)),
        "-vm", str(_resolve_model(_VAD_MODEL)),
        "--vad",
        "--vad-threshold", str(_VAD_THRESHOLD),
        "--vad-min-silence-duration-ms", str(_VAD_MIN_SILENCE_MS),
        "--vad-min-speech-duration-ms", str(_VAD_MIN_SPEECH_MS),
        "--vad-speech-pad-ms", str(_VAD_SPEECH_PAD_MS),
        "--vad-samples-overlap", str(_VAD_SAMPLES_OVERLAP),
        "-f", str(wav_path),
        "-l", _LANGUAGE,
        "-ojf",  # output json full
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True,
    )

    # whisper-cli writes JSON to a file next to the input: <wav>.json
    json_path = Path(str(wav_path) + ".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"whisper-cli JSON output not found at {json_path}"
        )

    whisper_data = json.loads(json_path.read_text(encoding="utf-8"))

    # Clean up the sidecar JSON — consumed into raw artifacts + step file
    json_path.unlink()

    return whisper_data, result.stderr


def _resolve_model(name: str) -> Path:
    """Resolve a model filename to a full path.

    Looks in ./models/ first, then in standard locations.
    """
    candidates = [
        Path("models") / name,
        Path.home() / ".local" / "share" / "voxtype" / "models" / name,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()

    # Fall back to HuggingFace cache for whisper models
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    for snap in sorted(hf_cache.glob("models--ggerganov--whisper.cpp/snapshots/*/")):
        p = snap / name
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(f"Model not found: {name}")


# ── Raw artifact storage ────────────────────────────────────────────────────

def _save_raw_artifacts(
    writer: StepWriter,
    whisper_json: dict[str, Any],
    stderr_text: str,
) -> None:
    """Save raw whisper-cli outputs to 02_transcribe.raw/ subfolder."""
    raw_dir = writer.steps_dir / "02_transcribe.raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "whisper_output.json").write_text(
        json.dumps(whisper_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (raw_dir / "whisper_stderr.txt").write_text(stderr_text, encoding="utf-8")


# ── Stderr parsing ─────────────────────────────────────────────────────────

def _parse_voice_ranges(stderr: str) -> list[dict[str, int]]:
    """Extract VAD voice ranges from whisper-cli stderr.

    Returns list of {"start_ms": int, "end_ms": int}.
    """
    ranges = []
    for line in stderr.splitlines():
        m = _RE_VAD_SEGMENT.search(line)
        if m:
            ranges.append({
                "start_ms": round(float(m.group(1)) * 1000),
                "end_ms": round(float(m.group(2)) * 1000),
            })
    return ranges


def _parse_timings(stderr: str) -> dict[str, float]:
    """Extract timing info from whisper-cli stderr.

    Returns dict of {name: seconds}.
    """
    timings: dict[str, float] = {}
    for line in stderr.splitlines():
        m = _RE_TIMING.search(line)
        if m:
            key = m.group(1).replace(" time", "").strip()
            timings[key] = round(float(m.group(2)) / 1000, 3)
    return timings


def _parse_language(stderr: str) -> tuple[str, float]:
    """Extract detected language and confidence from stderr."""
    m = _RE_LANGUAGE.search(stderr)
    if m:
        return m.group(1), round(float(m.group(2)), 4)
    return "unknown", 0.0


def _parse_audio_duration(stderr: str) -> int:
    """Extract audio duration in ms from stderr."""
    m = _RE_AUDIO_DURATION.search(stderr)
    if m:
        return round(float(m.group(1)) * 1000)
    return 0


def _parse_model_path(stderr: str) -> str:
    """Extract whisper model path from stderr."""
    m = _RE_MODEL_PATH.search(stderr)
    return m.group(1) if m else "unknown"


def _parse_whisper_version(stderr: str) -> str:
    """Extract whisper model version name (e.g. 'large v3') from stderr."""
    m = _RE_MODEL_TYPE.search(stderr)
    return m.group(1).strip() if m else "unknown"


def _parse_ftype(stderr: str) -> int:
    """Extract ftype from stderr (0=Q4_0, 1=F16, etc.)."""
    m = _RE_FTYPE.search(stderr)
    return int(m.group(1)) if m else -1


def _parse_qntvr(stderr: str) -> int:
    """Extract qntvr from stderr (0=none, 1=quantized)."""
    m = _RE_QNTVR.search(stderr)
    return int(m.group(1)) if m else -1


# ── Step output assembly ────────────────────────────────────────────────────

def _build_step(
    whisper_json: dict[str, Any],
    stderr_text: str,
    voice_ranges: list[dict[str, int]],
    timings: dict[str, float],
    audio_duration_ms: int,
    lang_info: tuple[str, float],
    wall_time_s: float,
) -> dict[str, Any]:
    """Assemble the step-02 output dict from whisper output + parsed metadata."""
    segments = _build_transcription(whisper_json)

    return {
        "step": "02_transcribe",
        "description": "Whisper.cpp transcription with built-in Silero VAD",
        "model": _build_model_info(whisper_json, stderr_text),
        "config": _build_config(),
        "vad": _build_vad(voice_ranges, audio_duration_ms),
        "result": {
            "language": lang_info[0],
            "language_confidence": lang_info[1],
            "num_segments": len(segments),
            "full_text": " ".join(s["text"].strip() for s in segments),
        },
        "lines": _build_lines(segments),
        "transcription": segments,
        "timing": _build_timing(timings, wall_time_s),
    }


def _build_model_info(whisper_json: dict[str, Any], stderr: str) -> dict[str, Any]:
    """Extract model metadata from whisper JSON + stderr."""
    model = whisper_json.get("model", {})

    # Parse whisper version from stderr: "type = 5 (large v3)"
    version = _parse_whisper_version(stderr)

    # Parse quantization from stderr
    ftype_id = _parse_ftype(stderr)
    qntvr_id = _parse_qntvr(stderr)

    # Determine container format from model path
    model_path = _parse_model_path(stderr)
    suffix = Path(model_path).suffix if model_path != "unknown" else ""
    container = _CONTAINER_NAMES.get(suffix, suffix.lstrip(".") or "unknown")

    return {
        "whisper": {
            "version": version,
            "multilingual": model.get("multilingual", False),
            "container": container,
            "ftype": _FTYPE_NAMES.get(ftype_id, f"unknown({ftype_id})"),
            "quantization": _QNTVR_NAMES.get(qntvr_id, f"unknown({qntvr_id})"),
        },
        "vad": {
            "type": "silero",
            "version": _VAD_MODEL.replace("ggml-silero-", "").replace(".bin", ""),
        },
    }


def _build_config() -> dict[str, Any]:
    """Build the config section from our VAD params."""
    return {
        "language": _LANGUAGE,
        "vad_threshold": _VAD_THRESHOLD,
        "vad_min_silence_ms": _VAD_MIN_SILENCE_MS,
        "vad_min_speech_ms": _VAD_MIN_SPEECH_MS,
        "vad_speech_pad_ms": _VAD_SPEECH_PAD_MS,
        "vad_samples_overlap": _VAD_SAMPLES_OVERLAP,
    }


def _build_vad(
    voice_ranges: list[dict[str, int]],
    audio_duration_ms: int,
) -> dict[str, Any]:
    """Build the VAD section from parsed voice ranges."""
    speech_ms = sum(r["end_ms"] - r["start_ms"] for r in voice_ranges)
    return {
        "num_voice_ranges": len(voice_ranges),
        "speech_duration_s": round(speech_ms / 1000, 3),
        "audio_duration_s": round(audio_duration_ms / 1000, 3),
        "reduction_pct": round(
            (1 - speech_ms / audio_duration_ms) * 100, 1
        ) if audio_duration_ms else 0,
        "voice_ranges": voice_ranges,
    }


def _build_lines(segments: list[dict[str, Any]]) -> list[str]:
    """Build the lines section — one readable line per segment.

    Format mirrors whisper-cli stdout:
      [00:00:00.000 --> 00:00:02.260]   Si, vous faites une photo de quoi là ?
    """
    lines = []
    for seg in segments:
        start = _ms_to_timestamp(seg["start_ms"])
        end = _ms_to_timestamp(seg["end_ms"])
        lines.append(f"[{start} --> {end}]   {seg['text']}")
    return lines


def _ms_to_timestamp(ms: int) -> str:
    """Convert milliseconds to HH:MM:SS.mmm timestamp."""
    total_s = ms / 1000
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _build_transcription(whisper_json: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert whisper JSON transcription to our segment schema.

    Each segment has: start_ms, end_ms, text, tokens[]
    Each token has: text, start_ms, end_ms, p (probability)
    Special tokens (like [_BEG_], [_TT_*]) are filtered out.
    """
    segments = []
    for entry in whisper_json.get("transcription", []):
        offsets = entry.get("offsets", {})
        tokens = []
        for tok in entry.get("tokens", []):
            text = tok.get("text", "")
            # Skip special tokens (e.g. [_BEG_], [_TT_113])
            if text.startswith("[_") and text.endswith("]"):
                continue
            tok_offsets = tok.get("offsets", {})
            tokens.append({
                "text": text,
                "start_ms": tok_offsets.get("from", 0),
                "end_ms": tok_offsets.get("to", 0),
                "p": round(tok.get("p", 0.0), 4),
            })
        segments.append({
            "start_ms": offsets.get("from", 0),
            "end_ms": offsets.get("to", 0),
            "text": entry.get("text", "").strip(),
            "tokens": tokens,
        })
    return segments


def _build_timing(timings: dict[str, float], wall_time_s: float) -> dict[str, Any]:
    """Build the timing section from parsed timings + wall clock."""
    return {
        **timings,
        "wall_s": round(wall_time_s, 2),
    }

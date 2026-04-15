"""Step 02 — Transcription with built-in VAD.

Runs whisper-cli with Silero VAD to transcribe the WAV produced by step 01.
The VAD skips silence internally, so we get both voice_ranges (where speech
was detected) and segments (transcribed text with word-level timestamps
and confidence scores).

Reads audio from the WAV artifact produced by step 01.
Saves a rich step file with model info, VAD config, voice ranges,
and full transcription with token-level detail.

Raw whisper-cli outputs are preserved under 02_transcribe.raw/.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from clotaire.steps_io import StepWriter

# ── Defaults ────────────────────────────────────────────────────────────────

_WHISPER_MODEL = "ggml-large-v3.bin"
_VAD_MODEL = "ggml-silero-v6.2.0.bin"
_VAD_THRESHOLD = 0.5
_VAD_MIN_SILENCE_MS = 1000
_VAD_MIN_SPEECH_MS = 100
_VAD_SPEECH_PAD_MS = 100
_VAD_SAMPLES_OVERLAP = 0.0
_LANGUAGE = "auto"

# ── Regexes ─────────────────────────────────────────────────────────────────

_RE_VAD_SEGMENT = re.compile(r"VAD segment \d+: start = ([\d.]+), end = ([\d.]+)")
_RE_TIMING = re.compile(r"whisper_print_timings:\s+(\w[\w ]*?)\s*=\s*([\d.]+)\s+ms")
_RE_LANGUAGE = re.compile(r"auto-detected language: (\w+) \(p = ([\d.]+)\)")
_RE_MODEL_TYPE = re.compile(r"type\s*=\s*\d+\s+\((.+?)\)")
_RE_VAD_TYPE = re.compile(r"whisper_vad_init_with_params: model type: (.+)")
_RE_VAD_VERSION = re.compile(r"whisper_vad_init_with_params: model version: (.+)")
_RE_FTYPE = re.compile(r"ftype\s*=\s*(\d+)")
_RE_QNTVR = re.compile(r"qntvr\s*=\s*(\d+)")
_RE_AUDIO_DURATION = re.compile(r"(\d+\.\d+)\s+sec\)")

# ── Lookups ─────────────────────────────────────────────────────────────────

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


# ── Public API ──────────────────────────────────────────────────────────────

def execute(wav_path: Path, writer: StepWriter) -> dict[str, Any]:
    """Run step 02 end to end.

    Runs whisper-cli on the step-01 WAV artifact, builds the step file,
    saves it, and preserves raw tool outputs.

    Returns the step data dict for downstream steps.
    """
    t0 = time.perf_counter()
    whisper_json, stderr_text = _run_whisper(wav_path)
    elapsed = time.perf_counter() - t0

    _save_raw_artifacts(writer, whisper_json, stderr_text)

    voice_ranges = _parse_voice_ranges(stderr_text)
    timings = _parse_timings(stderr_text)
    audio_duration_ms = _parse_audio_duration(stderr_text)
    lang_info = _parse_language(stderr_text)

    step_data = _build_step(
        whisper_json, stderr_text, voice_ranges, timings,
        audio_duration_ms, lang_info, elapsed,
    )
    path = writer.save(2, "transcribe", step_data)
    path.write_text(
        _compact_voice_ranges_in_json(path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )

    return step_data


# ── Whisper invocation ──────────────────────────────────────────────────────

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

    json_path = Path(str(wav_path) + ".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"whisper-cli JSON output not found at {json_path}"
        )

    whisper_data = json.loads(json_path.read_text(encoding="utf-8"))

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


# ── Stderr parsing ──────────────────────────────────────────────────────────

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


def _parse_vad_type(stderr: str) -> str:
    """Extract VAD model type from stderr."""
    m = _RE_VAD_TYPE.search(stderr)
    return m.group(1).strip() if m else "unknown"


def _parse_vad_version(stderr: str) -> str:
    """Extract VAD model version from stderr."""
    m = _RE_VAD_VERSION.search(stderr)
    return m.group(1).strip() if m else "unknown"


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
            "segments": _build_lines(segments),
        },
        "transcription": segments,
        "timing": _build_timing(timings, wall_time_s),
    }


def _build_model_info(whisper_json: dict[str, Any], stderr: str) -> dict[str, Any]:
    """Extract model metadata from whisper JSON + stderr."""
    model = whisper_json.get("model", {})

    ftype = _parse_ftype(stderr)
    qntvr = _parse_qntvr(stderr)

    return {
        "whisper": {
            "version": _parse_whisper_version(stderr),
            "multilingual": model.get("multilingual", False),
            "ftype": ftype,
            "ftype_name": _FTYPE_NAMES.get(ftype, f"unknown({ftype})"),
            "qntvr": qntvr,
            "qntvr_name": _QNTVR_NAMES.get(qntvr, f"unknown({qntvr})"),
        },
        "vad": {
            "type": _parse_vad_type(stderr),
            "version": _parse_vad_version(stderr),
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
    """Build readable lines mirroring whisper-cli stdout."""
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


def _compact_voice_ranges_in_json(text: str) -> str:
    """Render vad.voice_ranges entries on single lines in saved JSON."""
    lines = text.splitlines()
    out: list[str] = []
    in_voice_ranges = False
    i = 0

    while i < len(lines):
        line = lines[i]

        if '"voice_ranges": [' in line:
            in_voice_ranges = True
            out.append(line)
            i += 1
            continue

        if in_voice_ranges:
            stripped = line.strip()

            if stripped == ']':
                in_voice_ranges = False
                out.append(line)
                i += 1
                continue

            if stripped == '{' and i + 3 < len(lines):
                l1 = lines[i + 1].strip()
                l2 = lines[i + 2].strip()
                l3 = lines[i + 3].strip()
                m1 = re.match(r'"start_ms":\s*(\d+),', l1)
                m2 = re.match(r'"end_ms":\s*(\d+)', l2)
                if m1 and m2 and l3 in {'}', '},'}:
                    indent = line[: len(line) - len(line.lstrip())]
                    trailing = ',' if l3.endswith(',') else ''
                    out.append(
                        f'{indent}{{ "start_ms": {m1.group(1)}, "end_ms": {m2.group(1)} }}{trailing}'
                    )
                    i += 4
                    continue

        out.append(line)
        i += 1

    return "\n".join(out) + "\n"

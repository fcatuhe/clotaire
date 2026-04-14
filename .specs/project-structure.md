# Kloter — Project Structure

> Python packaging & layout conventions for the pipeline

## Directory Layout

```
kloter/                          # Project root (git repo)
├── pyproject.toml               # Package metadata, dependencies, CLI entry point
├── README.md                    # Quick start, usage, install
├── architecture.md              # Pipeline design doc
├── project-structure.md         # This file
├── .gitignore
├── .env.example                 # HF_TOKEN placeholder (gated models)
│
├── src/                         # ── Source code (src layout) ──
│   └── kloter/                  # Main package
│       ├── __init__.py          # Version, public API
│       ├── __main__.py          # `python -m kloter` entry point
│       ├── cli.py              # argparse CLI — thin wrapper only
│       ├── pipeline.py          # Orchestrator — wires steps, parallelism
│       ├── steps/              # One module per pipeline step
│       │   ├── __init__.py     # Re-exports step functions
│       │   ├── convert.py      # ① Audio conversion (ffmpeg)
│       │   ├── vad.py          # ② VAD (pyannote segmentation)
│       │   ├── whisper.py     # ③ Transcription (whisper large-v3) + ③bis language attach
│       │   ├── diarize.py     # ④ Diarization (pyannote)
│       │   ├── align.py       # ⑤ Wav2vec2 alignment (per-segment language)
│       │   ├── match.py       # ⑥ Speaker matching
│       │   └── format.py      # ⑦ JSON + Markdown output
│       └── models/             # Model loading, caching, shared resources
│           ├── __init__.py
│           └── loader.py       # Whisper, pyannote, wav2vec2 model management
│
├── tests/                      # ── Tests ──
│   ├── conftest.py             # Shared fixtures (audio samples, mock models)
│   ├── test_convert.py
│   ├── test_vad.py
│   ├── test_whisper.py
│   ├── test_align.py
│   ├── test_match.py
│   ├── test_format.py
│   ├── test_pipeline.py        # Integration test (with tiny model)
│   └── fixtures/               # Test audio files
│       ├── short_fr.wav
│       ├── short_en.wav
│       └── multilingual.wav
│
└── scripts/                    # ── Dev/utility scripts ──
    ├── download_models.py      # Pre-download all models
    └── benchmark.py            # Timing & memory profiling
```

---

## Why `src/` layout?

The **src layout** (`src/kloter/`) is the modern Python standard (PEP 621, setuptools, hatch). Benefits:

| Flat layout (`kloter/`) | Src layout (`src/kloter/`) |
|---|---|
| `import kloter` works before `pip install` — hides packaging bugs | Must `pip install -e .` to import — catches issues early |
| Accidental CWD imports | Forces proper installed imports |
| pytest finds package by CWD | pytest finds package by installation |

Setuptools, hatch, flit all recommend it.

---

## `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "kloter"
version = "0.1.0"
description = "Multilingual audio transcription with speaker diarization"
requires-python = ">=3.10"
license = {text = "MIT"}

dependencies = [
    "openai-whisper",
    "torch",
    "torchaudio",
    "pyannote.audio",
    "whisperx",
    "numpy",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "ruff",
]

[project.scripts]
kloter = "kloter.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`uv sync` reads `pyproject.toml` + resolves a `uv.lock` file. No extra config needed — `uv` is a drop-in replacement for pip/venv/pip-tools.

---

## Core Files

### `src/kloter/__init__.py`

```python
"""Kloter — Multilingual audio transcription with speaker diarization."""

__version__ = "0.1.0"

from kloter.pipeline import run  # noqa: F401 — public API
```

Public API is just `kloter.run()`. Everything else is internal.

---

### `src/kloter/__main__.py`

```python
"""Allow `python -m kloter` invocation."""

from kloter.cli import main

main()
```

---

### `src/kloter/cli.py`

Thin CLI wrapper. **No business logic** — only parses args, calls `pipeline.run()`, prints result.

```python
"""CLI entry point for kloter."""

import argparse
import json
import sys

from kloter.pipeline import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kloter",
        description="Multilingual audio transcription with speaker diarization",
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--format", default="all",
        choices=["json", "md", "all"],
        help="Output format: json, md, or all (default: all)",
    )
    parser.add_argument("--output-dir", default=None, help="Write files to this directory (default: same as audio)")
    parser.add_argument("--stdout", action="store_true", help="Output JSON on stdout instead of writing files")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for gated models (or set HF_TOKEN env var)")
    parser.add_argument("--max-speakers", type=int, default=None, help="Max speakers for diarization")
    parser.add_argument("--min-speakers", type=int, default=None, help="Min speakers for diarization")
    parser.add_argument("--max-languages", type=int, default=3, help="Max wav2vec2 models to load (memory limit)")
    args = parser.parse_args(argv)

    result = run(
        audio_path=args.audio,
        hf_token=args.hf_token,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        max_languages=args.max_languages,
    )

    if args.stdout:
        # JSON on stdout for piping / Ruby
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        # Write files (default behavior)
        from kloter.steps.format import write_files
        write_files(result, audio_path=args.audio, output_dir=args.output_dir, fmt=args.format)


if __name__ == "__main__":
    main()
```

**Callable three ways:**

```bash
# As installed CLI
kloter audio.mp3

# As Python module
python -m kloter audio.mp3

# From Ruby
result_json = `kloter /path/to/audio.mp3 --output-dir /tmp/out`
```

---

### `src/kloter/pipeline.py`

Orchestrator. Owns the control flow (sequential/parallel), not the step logic.

```python
"""Pipeline orchestrator — composes steps, manages parallelism."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from kloter.steps.convert import load_audio
from kloter.steps.vad import detect_speech
from kloter.steps.whisper import transcribe_segments, attach_language_to_words
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
    speech_segments = detect_speech(audio)

    # ③ + ④ in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_future = pool.submit(transcribe_segments, audio, speech_segments)
        diar_future = pool.submit(
            diarize, audio, hf_token=hf_token,
            min_speakers=min_speakers, max_speakers=max_speakers,
        )
        whisper_result = whisper_future.result()
        diar_segments = diar_future.result()

    # ③bis — attach language from whisper to each word
    words = attach_language_to_words(whisper_result)

    # ⑤ Wav2vec2 alignment (per-segment language, cached models)
    aligned_words = align_words(whisper_result, audio, max_languages=max_languages)

    # ⑥ Speaker matching
    final_words = match_speakers(aligned_words, diar_segments)

    # ⑦ Format output
    result = format_output(final_words, diar_segments, speech_segments, audio_path, audio)

    return result
```

---

### `src/kloter/steps/*.py` — Step modules

Each step follows the same pattern:

1. **Pure function** — takes data in, returns data out
2. **No side effects** — no global state, no CLI parsing, no stdout
3. **Typed signatures** — input/output types are clear
4. **Independent** — testable in isolation

```python
# Example: src/kloter/steps/convert.py

"""① Audio conversion — any format → WAV 16kHz mono float32."""

from __future__ import annotations

import subprocess

import numpy as np


def load_audio(path: str) -> np.ndarray:
    """Convert any audio file to WAV 16kHz mono float32 numpy array.
    
    Args:
        path: Path to audio file (mp3, ogg, m4a, flac, wav, webm…)
    
    Returns:
        Float32 numpy array, values in [-1, 1], sample rate 16kHz, mono.
    
    Raises:
        FileNotFoundError: If path does not exist.
        RuntimeError: If ffmpeg fails to convert.
    """
    result = subprocess.run(
        ["ffmpeg", "-i", path,
         "-f", "wav", "-acodec", "pcm_s16le",
         "-ac", "1", "-ar", "16000", "-"],
        capture_output=True,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio[22:]  # skip WAV header (44 bytes = 22 int16 samples)
```

---

### `src/kloter/models/loader.py`

Centralized model loading & caching. Shares the pyannote segmentation model between VAD and diarization.

```python
"""Model loading, caching, and resource management."""

from __future__ import annotations

import os
import torch
from typing import Any

# ── Shared model cache ──

_cache: dict[str, Any] = {}


def get_whisper_model(model_name: str = "large-v3"):
    """Load and cache whisper model."""
    key = f"whisper:{model_name}"
    if key not in _cache:
        import whisper
        _cache[key] = whisper.load_model(model_name)
    return _cache[key]


def get_pyannote_segmentation():
    """Load and cache pyannote segmentation model (shared by VAD + diarization)."""
    key = "pyannote:segmentation"
    if key not in _cache:
        from pyannote.audio import Model
        _cache[key] = Model.from_pretrained("pyannote/segmentation")
    return _cache[key]


def get_pyannote_diarization(hf_token: str | None = None):
    """Load and cache pyannote diarization pipeline."""
    key = "pyannote:diarization"
    if key not in _cache:
        hf_token = hf_token or os.environ.get("HF_TOKEN")
        from pyannote.audio import Pipeline as PyannotePipeline
        _cache[key] = PyannotePipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=hf_token,
        )
    return _cache[key]


def get_align_model(lang: str):
    """Load and cache wav2vec2 alignment model for a given language."""
    key = f"align:{lang}"
    if key not in _cache:
        from whisperx.alignment import load_align_model
        model, metadata = load_align_model(lang, "cpu")
        _cache[key] = (model, metadata)
    return _cache[key]


def configure_threads(whisper_threads: int = 8, pyannote_threads: int = 4):
    """Set CPU thread counts for parallel execution."""
    torch.set_num_threads(pyannote_threads)


def clear_cache():
    """Release all cached models (free memory)."""
    _cache.clear()
```

---

### `src/kloter/steps/format.py`

Two output formats: JSON (full data) and Markdown (human-readable). File writing is here too.

```python
"""⑦ Formatting final -- JSON + Markdown output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from kloter import __version__


def format_output(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
    speech_segments: list[dict[str, Any]],
    audio_path: str,
    audio: Any,  # numpy array -- avoid importing numpy at module level
) -> dict[str, Any]:
    """Build the full result dict (JSON-serializable)."""
    import numpy as np

    # Compute per-language duration from words
    from collections import defaultdict
    lang_duration = defaultdict(float)
    for w in words:
        lang_duration[w.get("language", "unknown")] += w["end"] - w["start"]

    # Build segments (group consecutive words by speaker)
    segments = _build_segments(words)

    return {
        "audio": os.path.basename(audio_path),
        "duration": round(len(audio) / 16000, 1),
        "languages": {lang: round(dur, 1) for lang, dur in sorted(lang_duration.items(), key=lambda x: -x[1])},
        "words": words,
        "segments": segments,
        "diarization": diar_segments,
        "speech_segments": speech_segments,
    }


def to_markdown(result: dict[str, Any]) -> str:
    """Convert result dict to human-readable Markdown."""
    lines = []

    # Header
    audio_name = result["audio"]
    duration = result["duration"]
    langs = ", ".join(f"{lang} ({dur}s)" for lang, dur in result["languages"].items())
    n_speakers = len({seg["speaker"] for seg in result["diarization"]})

    lines.append(f"# Transcription -- {audio_name}")
    lines.append("")
    lines.append(f"**Duration**: {duration}s | **Languages**: {langs} | **Speakers**: {n_speakers}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group segments by speaker
    by_speaker: dict[str, list[dict]] = {}
    for seg in result["segments"]:
        spk = seg["speaker"]
        by_speaker.setdefault(spk, []).append(seg)

    for speaker, segs in sorted(by_speaker.items()):
        start = _fmt_time(segs[0]["start"])
        end = _fmt_time(segs[-1]["end"])
        lines.append(f"## {speaker}")
        lines.append("")
        lines.append(f"**[{start}--{end}]**")
        lines.append("")
        for seg in segs:
            lines.append(seg["text"])
            lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_Generated by kloter v{__version__}_")

    return "\n".join(lines)


def write_files(
    result: dict[str, Any],
    audio_path: str,
    output_dir: str | None = None,
    fmt: str = "all",
) -> list[Path]:
    """Write output files. Returns list of paths written."""
    basename = Path(audio_path).stem
    target_dir = Path(output_dir) if output_dir else Path(audio_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)

    written = []

    if fmt in ("all", "json"):
        json_path = target_dir / f"{basename}.transcription.json"
        json_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(json_path)

    if fmt in ("all", "md"):
        md_path = target_dir / f"{basename}.transcription.md"
        md_path.write_text(to_markdown(result) + "\n", encoding="utf-8")
        written.append(md_path)

    return written


def _build_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group consecutive words by speaker into segments."""
    if not words:
        return []
    segments = []
    current = {"start": words[0]["start"], "speaker": words[0]["speaker"], "language": words[0].get("language"), "text": words[0]["word"]}
    for w in words[1:]:
        if w["speaker"] == current["speaker"] and w.get("language") == current.get("language"):
            current["end"] = w["end"]
            current["text"] += " " + w["word"]
        else:
            segments.append(current)
            current = {"start": w["start"], "end": w["end"], "speaker": w["speaker"], "language": w.get("language"), "text": w["word"]}
    segments.append(current)
    return segments


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"
```

---

### `src/kloter/steps/__init__.py`

```python
"""Pipeline step functions."""

from kloter.steps.convert import load_audio
from kloter.steps.vad import detect_speech
from kloter.steps.whisper import transcribe_segments, attach_language_to_words
from kloter.steps.diarize import diarize
from kloter.steps.align import align_words
from kloter.steps.match import match_speakers
from kloter.steps.format import format_output, write_files, to_markdown

__all__ = [
    "load_audio",
    "detect_speech",
    "transcribe_segments",
    "attach_language_to_words",
    "diarize",
    "align_words",
    "match_speakers",
    "format_output",
    "write_files",
    "to_markdown",
]
```

---

## Design Principles

### 1. Steps are stateless functions

```python
# ✅ Good — pure function, testable, no side effects
def load_audio(path: str) -> np.ndarray: ...

# ❌ Bad — class with hidden state, hard to test
class AudioConverter:
    def __init__(self, config): ...
    def convert(self, path): ...
```

### 2. Pipeline owns control flow, steps own logic

```python
# ✅ pipeline.py decides what runs in parallel
with ThreadPoolExecutor(max_workers=2) as pool:
    whisper_future = pool.submit(transcribe_segments, ...)
    diar_future = pool.submit(diarize, ...)

# ❌ steps should not launch their own threads or know about parallelism
```

### 3. Models loaded once, shared, cached

```python
# ✅ loader.py centralizes model lifecycle
model = get_whisper_model()       # cached after first load
seg  = get_pyannote_segmentation()  # shared by VAD + diarization

# ❌ each step loading its own model = duplicate RAM, wasted time
```

### 4. CLI is a thin wrapper

```python
# ✅ cli.py — parse args, call pipeline, print result
def main():
    args = parse_args()
    result = run(audio_path=args.audio, ...)
    json.dump(result, sys.stdout)

# ❌ cli.py should not contain step logic, model loading, or threading
```

### 5. Env vars for secrets, args for behavior

```python
# ✅ HF token from env var (12-factor app)
hf_token = hf_token or os.environ.get("HF_TOKEN")

# ❌ Hardcoded tokens, config files for secrets
```

### 6. Type hints everywhere

```python
# ✅ Clear contracts
def match_speakers(
    words: list[dict[str, Any]],
    diar_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]: ...
```

---

## Calling from Ruby (Sidekiq)

```ruby
# app/jobs/transcribe_job.rb
class TranscribeJob < ApplicationJob
  def perform(audio_path, output_dir)
    # Option 1: JSON on stdout, no files written
    result_json = `kloter #{audio_path} --stdout 2>/dev/null`
    result = JSON.parse(result_json)

    # Option 2: Write files, read JSON back
    `kloter #{audio_path} --output-dir #{output_dir} 2>/dev/null`
    basename = File.basename(audio_path, ".*")
    result = JSON.parse(File.read("#{output_dir}/#{basename}.transcription.json"))
  end
end
```

Or without installing the package:

```ruby
result_json = `python3 -m kloter #{audio_path} --stdout`
```

Or programmatically from any Python context:

```python
from kloter import run
result = run("audio.mp3")
# result is a dict — write files yourself with kloter.steps.format.write_files()
```

---

## Development with `uv`

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager (written in Rust). Already installed on this machine.

```bash
# Create venv + install project in editable mode with dev deps
uv sync --extra dev

# Run the CLI (uv ensures the right venv)
uv run kloter audio.mp3                        # → audio.transcription.json + audio.transcription.md
uv run kloter audio.mp3 --format json          # → audio.transcription.json only
uv run kloter audio.mp3 --format md            # → audio.transcription.md only
uv run kloter audio.mp3 --stdout               # → JSON on stdout (for piping / Ruby)
uv run kloter audio.mp3 --output-dir /tmp/out  # → /tmp/out/audio.transcription.*

# Or activate the venv manually and run directly
source .venv/bin/activate
kloter audio.mp3

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Add a new dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>

# Run any command in the project environment
uv run python -m kloter audio.mp3
```

### Why `uv` over `pip`?

| | `pip` | `uv` |
|---|---|---|
| Speed | Slow (network + install) | 10-100× faster |
| Lock file | No (pip freeze) | `uv.lock` — deterministic |
| Venv management | Manual | Auto (`.venv/`) |
| Python version | System | Can pin in `pyproject.toml` |
| `uv sync` | No equivalent | Installs project + deps from lock | |

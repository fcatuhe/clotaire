# AGENTS.md — Project guide for AI agents

## Project overview

**clotaire** is a step-by-step audio/video transcription CLI. Each step produces a numbered JSON artifact under `<media_dir>/<media_stem>/steps/`. Steps are independent, re-runnable, and self-documenting.

## Running

```bash
# Run the full pipeline on a media file
uv run clotaire audio_samples/260331_1031.mp3 --trace

# Run tests
uv run pytest

# Lint
uv run ruff check src/
```

The project uses `uv` for dependency management. All commands go through `uv run`.

## Project structure

```
src/clotaire/
  __init__.py          — version
  __main__.py          — python -m clotaire entry
  cli.py               — argparse CLI (--trace mode)
  run.py               — orchestrator: runs steps sequentially
  steps_io.py          — StepWriter: saves numbered step JSON + artifact paths
  step_01_convert.py   — ffmpeg: any media → 16kHz mono WAV + ffprobe metadata
  step_02_transcribe.py — whisper-cli with Silero VAD: transcription + token detail
```

## Pipeline steps

| Step | Name | Tool | Input | Output |
|------|------|------|-------|--------|
| 01 | convert | ffmpeg + ffprobe | Original media | WAV on disk + probe metadata |
| 02 | transcribe | whisper-cli (Silero VAD) | WAV path | voice_ranges + segments with token-level timestamps + confidence |

### Step 01: convert
- One `ffmpeg` subprocess: any format → pcm_s16le 16kHz mono WAV
- One `ffprobe` subprocess: rich format/stream metadata
- The WAV is the canonical audio source for all downstream steps

### Step 02: transcribe
- Runs `whisper-cli` with built-in Silero VAD (`--vad`)
- VAD skips silence internally — no separate VAD step needed
- VAD config: threshold=0.5, min_silence=1000ms, min_speech=100ms, pad=100ms, overlap=0
- Outputs **voice_ranges** (VAD-detected speech regions) and **segments** (transcribed text with per-token offsets + probabilities)

## Key vocabulary

| Term | Meaning |
|------|---------|
| **voice_ranges** | VAD-detected continuous speech regions (broad: "someone was talking here") |
| **segments** | Whisper transcribed sentence-level units with text + tokens (granular: "this specific sentence") |

## Models

Whisper and VAD models are resolved from:
1. `./models/` (project-local)
2. `~/.local/share/voxtype/models/`
3. HuggingFace cache (`~/.cache/huggingface/hub/`)

Current models:
- `ggml-large-v3.bin` (whisper)
- `ggml-silero-v6.2.0.bin` (VAD)

## Dependencies

- **ffmpeg/ffprobe** — must be on PATH (step 01)
- **whisper-cli** — must be on PATH (step 02)
- **Python deps** — managed via uv: torch, torchaudio, pyannote.audio, numpy (for future steps)

## Testing

Tests live in `tests/`. Run with `uv run pytest`. Step files are integration-tested against the sample file in `audio_samples/`.

## Code style

- Ruff, line-length 100, target Python 3.10+
- Type hints everywhere
- Step modules follow pattern: `execute()` as public API, private helpers below

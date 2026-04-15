# Kloter

Multilingual audio transcription with speaker diarization.

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
kloter audio.mp3                        # → audio.transcription.json + audio.transcription.md
kloter audio.mp3 --format json          # → audio.transcription.json only
kloter audio.mp3 --format md            # → audio.transcription.md only
kloter audio.mp3 --stdout               # → JSON on stdout (for piping / Ruby)
kloter audio.mp3 --output-dir /tmp/out  # → /tmp/out/audio.transcription.*
```

## Requirements

- Python >= 3.10
- ffmpeg (system)
- HuggingFace token (for gated pyannote models): set `HF_TOKEN` env var or pass `--hf-token`

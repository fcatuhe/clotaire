# Clotaire

Step-by-step audio transcription CLI.

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
clotaire /path/to/audio.mp3 --trace
```

Saves numbered step files under `<audio_dir>/<audio_stem>/steps/`:

```
audio.mp3
audio/
  steps/
    01_convert.audio.json
    02_transcribe.audio.json
    03_align.audio.json
```

## Steps

| # | Name | Description |
|---|------|-------------|
| 01 | convert | Audio conversion: any format → 16kHz mono WAV |
| 02 | transcribe | whisper.cpp transcription with built-in Silero VAD |
| 03 | align | wav2vec2 forced alignment on the canonical WAV |

## Requirements

- Python >= 3.10
- ffmpeg / ffprobe (system)
- whisper-cli on PATH (whisper.cpp)

# Kloter

Step-by-step audio transcription CLI.

## Install

```bash
uv sync --extra dev
```

## Usage

```bash
kloter /path/to/audio.mp3 --trace
```

Saves numbered step files under `<audio_dir>/<audio_stem>/steps/`:

```
audio.mp3
audio/
  steps/
    01_convert.audio.json
    02_transcribe.audio.json
```

## Steps

| # | Name | Description |
|---|------|-------------|
| 01 | convert | Audio conversion: any format → 16kHz mono WAV |
| 02 | transcribe | whisper.cpp transcription with built-in Silero VAD |

## Requirements

- Python >= 3.10
- ffmpeg / ffprobe (system)
- whisper-cli on PATH (whisper.cpp)

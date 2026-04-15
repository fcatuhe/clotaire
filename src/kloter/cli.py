"""CLI entry point for kloter.

Usage:
    kloter /path/to/audio.mp3 --trace

Saves numbered step files under <audio_dir>/<audio_stem>/steps/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kloter.run import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kloter",
        description="Step-by-step audio transcription CLI",
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save numbered step files alongside the audio",
    )
    args = parser.parse_args(argv)

    if not args.trace:
        parser.error("Currently only --trace mode is supported")

    audio_path = Path(args.audio)
    if not audio_path.exists():
        parser.error(f"File not found: {audio_path}")

    run(audio_path)


if __name__ == "__main__":
    main()

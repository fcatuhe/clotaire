"""CLI entry point for kloter.

Usage:
    kloter /path/to/media.mp3 --trace

Saves numbered step files under <media_dir>/<media_stem>/steps/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kloter.run import run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kloter",
        description="Step-by-step media transcription CLI",
    )
    parser.add_argument("media", help="Path to audio or video file")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save numbered step files alongside the media",
    )
    args = parser.parse_args(argv)

    if not args.trace:
        parser.error("Currently only --trace mode is supported")

    media_path = Path(args.media)
    if not media_path.exists():
        parser.error(f"File not found: {media_path}")

    run(media_path)


if __name__ == "__main__":
    main()

"""CLI entry point for kloter."""

from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv

# Load .env from CWD before anything else
load_dotenv()

from kloter.pipeline import run  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="kloter",
        description="Multilingual audio transcription with speaker diarization",
    )
    parser.add_argument("audio", help="Path to audio file")
    parser.add_argument(
        "--format",
        default="all",
        choices=["json", "md", "all"],
        help="Output format: json, md, or all (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write files to this directory (default: same as audio)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Output JSON on stdout instead of writing files",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for gated models (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Max speakers for diarization",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Min speakers for diarization",
    )
    parser.add_argument(
        "--max-languages",
        type=int,
        default=3,
        help="Max wav2vec2 alignment models to load (memory limit)",
    )
    parser.add_argument(
        "--save-steps",
        default=None,
        help="Directory to save intermediate step outputs as numbered JSON files",
    )
    args = parser.parse_args(argv)

    result = run(
        audio_path=args.audio,
        hf_token=args.hf_token,
        max_speakers=args.max_speakers,
        min_speakers=args.min_speakers,
        max_languages=args.max_languages,
        save_steps=args.save_steps,
    )

    if args.stdout:
        json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
        sys.stdout.write("\n")
    else:
        from kloter.steps.format import write_files

        paths = write_files(
            result,
            audio_path=args.audio,
            output_dir=args.output_dir,
            fmt=args.format,
        )
        for p in paths:
            print(f"Wrote: {p}", file=sys.stderr)


if __name__ == "__main__":
    main()

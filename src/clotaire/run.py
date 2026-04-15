"""Orchestrator — runs steps sequentially, saves step files and artifacts."""

from __future__ import annotations

from pathlib import Path
from sys import stderr

from clotaire.steps_io import StepWriter
from clotaire.step_01_convert import execute as step_01
from clotaire.step_02_transcribe import execute as step_02


def run(media_path: Path) -> None:
    """Run all steps and save numbered step files."""
    writer = StepWriter(media_path)

    print("Step 01: convert …", flush=True)
    wav_path = step_01(media_path, writer)

    print("Step 02: transcribe …", flush=True)
    step_02(wav_path, writer)

    print("Done. Steps saved to:", writer.steps_dir, file=stderr)

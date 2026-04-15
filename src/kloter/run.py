"""Orchestrator — runs steps sequentially, saves step files and artifacts."""

from __future__ import annotations

from pathlib import Path

from kloter.steps_io import StepWriter
from kloter.step_01_convert import execute as step_01
from kloter.step_02_vad import execute as step_02


def run(audio_path: Path) -> None:
    """Run all steps and save numbered step files."""
    writer = StepWriter(audio_path)

    print("Step 01: convert …", flush=True)
    audio = step_01(audio_path, writer)

    print("Step 02: vad …", flush=True)
    step_02(audio, writer)

    print("Done. Steps saved to:", writer._steps_dir, file=__import__("sys").stderr)

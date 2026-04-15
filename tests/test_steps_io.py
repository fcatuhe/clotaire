"""Tests for step file writing."""

from __future__ import annotations

import json
from pathlib import Path

from clotaire.steps_io import StepWriter


def test_step_writer_builds_paths_and_saves_json(media_path: Path) -> None:
    writer = StepWriter(media_path)

    assert writer.steps_dir == media_path.parent / media_path.stem / "steps"
    assert writer.artifact_path(1, "convert", ".wav") == (
        writer.steps_dir / "01_convert.sample.wav"
    )

    out = writer.save(2, "transcribe", {"ok": True})

    assert out == writer.steps_dir / "02_transcribe.sample.json"
    assert json.loads(out.read_text(encoding="utf-8")) == {"ok": True}

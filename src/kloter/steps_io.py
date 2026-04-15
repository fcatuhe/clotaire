"""Step file writer — saves numbered JSON step outputs."""

from __future__ import annotations

import json
from pathlib import Path
from sys import stderr
from typing import Any


class StepWriter:
    """Write numbered step files under <media_dir>/<media_stem>/steps/.

    Each step file is named like:
        01_convert.<stem>.json
        02_vad.<stem>.json
    """

    def __init__(self, media_path: Path) -> None:
        self._stem = media_path.stem
        self._steps_dir = media_path.parent / self._stem / "steps"
        self._steps_dir.mkdir(parents=True, exist_ok=True)

    @property
    def steps_dir(self) -> Path:
        """Directory where step files are written."""
        return self._steps_dir

    def artifact_path(self, number: int, name: str, suffix: str) -> Path:
        """Return a path for a step artifact (e.g. converted WAV).

        Named like: 01_convert.<stem>.wav
        """
        return self._steps_dir / f"{number:02d}_{name}.{self._stem}{suffix}"

    def save(self, number: int, name: str, data: dict[str, Any]) -> Path:
        filename = f"{number:02d}_{name}.{self._stem}.json"
        path = self._steps_dir / filename
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  step saved: {path}", file=stderr)
        return path

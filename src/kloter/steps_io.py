"""Step file writer — saves numbered JSON step outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class StepWriter:
    """Write numbered step files under <audio_dir>/<audio_stem>/steps/.

    Each step file is named like:
        01_transcript.<stem>.json
        02_vad.<stem>.json
    """

    def __init__(self, audio_path: Path) -> None:
        self._stem = audio_path.stem
        self._steps_dir = audio_path.parent / self._stem / "steps"
        self._steps_dir.mkdir(parents=True, exist_ok=True)

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
        print(f"  step saved: {path}", file=sys.stderr)
        return path

"""Shared test fixtures for clotaire."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def media_path(tmp_path: Path) -> Path:
    """Return a sample media path for step/path tests."""
    path = tmp_path / "sample.mp3"
    path.write_bytes(b"fake-media")
    return path

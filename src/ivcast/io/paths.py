"""Filesystem path filters for generated research artifacts."""

from __future__ import annotations

from pathlib import Path


def sorted_artifact_files(root: Path, pattern: str) -> list[Path]:
    """Return deterministic file matches, excluding macOS AppleDouble sidecars."""

    return [
        path
        for path in sorted(root.glob(pattern))
        if path.is_file() and not path.name.startswith("._")
    ]


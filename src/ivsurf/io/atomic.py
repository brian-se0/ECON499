"""Atomic filesystem writes for crash-safe pipeline artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4


def _temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.{uuid4().hex}.tmp")


def cleanup_atomic_temp_files(output_path: Path) -> None:
    """Remove stale temp files left behind by interrupted atomic writes."""

    for stale_path in output_path.parent.glob(f"{output_path.name}.*.tmp"):
        if stale_path.is_file():
            stale_path.unlink()


def write_bytes_atomic(output_path: Path, payload: bytes) -> None:
    """Write bytes to a file atomically within the destination directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temp_files(output_path)
    temp_path = _temp_output_path(output_path)
    try:
        temp_path.write_bytes(payload)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_text_atomic(
    output_path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text to a file atomically within the destination directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temp_files(output_path)
    temp_path = _temp_output_path(output_path)
    try:
        temp_path.write_text(text, encoding=encoding)
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

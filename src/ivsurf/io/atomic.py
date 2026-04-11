"""Atomic filesystem writes for crash-safe pipeline artifacts."""

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

_REPLACE_RETRY_DELAYS_SECONDS: tuple[float, ...] = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
    2.00,
)


def _temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.{uuid4().hex}.tmp")


def cleanup_atomic_temp_files(output_path: Path) -> None:
    """Remove stale temp files left behind by interrupted atomic writes."""

    for stale_path in output_path.parent.glob(f"{output_path.name}.*.tmp"):
        if stale_path.is_file():
            stale_path.unlink()


def replace_path_atomic(temp_path: Path, output_path: Path) -> None:
    """Install a fully-written temp file at the destination path atomically."""

    for delay_seconds in (*_REPLACE_RETRY_DELAYS_SECONDS, -1.0):
        try:
            os.replace(temp_path, output_path)
            return
        except PermissionError:
            if delay_seconds < 0.0:
                raise
            # Windows file scanners can briefly hold a replaced target open.
            time.sleep(delay_seconds)


def write_bytes_atomic(output_path: Path, payload: bytes) -> None:
    """Write bytes to a file atomically within the destination directory."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temp_files(output_path)
    temp_path = _temp_output_path(output_path)
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        replace_path_atomic(temp_path, output_path)
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
        with temp_path.open("w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        replace_path_atomic(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

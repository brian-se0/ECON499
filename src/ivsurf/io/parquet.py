"""Shared parquet IO helpers with explicit defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl


def write_parquet_frame(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    statistics: bool = True,
) -> None:
    """Write a parquet artifact with the project's explicit defaults."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(output_path, compression=compression, statistics=statistics)


def read_parquet_files(paths: list[Path]) -> pl.DataFrame:
    """Read a list of parquet files into one concatenated frame."""

    if not paths:
        message = "read_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return pl.concat([pl.read_parquet(path) for path in paths])


def scan_parquet_files(paths: list[Path]) -> pl.LazyFrame:
    """Create one lazy parquet scan from explicit file paths."""

    if not paths:
        message = "scan_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return pl.scan_parquet([str(path) for path in paths])

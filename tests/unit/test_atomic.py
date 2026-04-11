from __future__ import annotations

import os
from pathlib import Path

import polars as pl
from pytest import MonkeyPatch

from ivsurf.io import atomic
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame

PathLikeArg = str | bytes | os.PathLike[str] | os.PathLike[bytes]


def test_write_bytes_atomic_retries_transient_replace_lock(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    output_path = tmp_path / "state.json"
    output_path.write_bytes(b"old")
    original_replace = os.replace
    replace_calls = {"count": 0}

    def flaky_replace(src: PathLikeArg, dst: PathLikeArg) -> None:
        replace_calls["count"] += 1
        if replace_calls["count"] < 3:
            raise PermissionError(5, "Access is denied")
        original_replace(src, dst)

    monkeypatch.setattr(atomic, "_REPLACE_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(atomic.time, "sleep", lambda _: None)
    monkeypatch.setattr(atomic.os, "replace", flaky_replace)

    write_bytes_atomic(output_path, b"new")

    assert output_path.read_bytes() == b"new"
    assert replace_calls["count"] == 3
    assert list(output_path.parent.glob("state.json.*.tmp")) == []


def test_write_parquet_frame_retries_transient_replace_lock(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    output_path = tmp_path / "frame.parquet"
    frame = pl.DataFrame({"value": [1.0, 2.0]})
    original_replace = os.replace
    replace_calls = {"count": 0}

    def flaky_replace(src: PathLikeArg, dst: PathLikeArg) -> None:
        replace_calls["count"] += 1
        if replace_calls["count"] < 2:
            raise PermissionError(5, "Access is denied")
        original_replace(src, dst)

    monkeypatch.setattr(atomic, "_REPLACE_RETRY_DELAYS_SECONDS", (0.0,))
    monkeypatch.setattr(atomic.time, "sleep", lambda _: None)
    monkeypatch.setattr(atomic.os, "replace", flaky_replace)

    write_parquet_frame(frame, output_path)

    assert pl.read_parquet(output_path).to_dicts() == frame.to_dicts()
    assert replace_calls["count"] == 2

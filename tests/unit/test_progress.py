from __future__ import annotations

import sys

from rich.progress import SpinnerColumn, TextColumn

from ivsurf.progress import _supports_braille_spinner, create_progress


class _DummyStdout:
    def __init__(self, encoding: str | None) -> None:
        self.encoding = encoding


def test_supports_braille_spinner_rejects_cp1252() -> None:
    assert _supports_braille_spinner("cp1252") is False


def test_supports_braille_spinner_accepts_utf8() -> None:
    assert _supports_braille_spinner("utf-8") is True


def test_create_progress_uses_ascii_safe_leading_column_when_stdout_is_cp1252(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "stdout", _DummyStdout("cp1252"))
    progress = create_progress()
    assert isinstance(progress.columns[0], TextColumn)


def test_create_progress_uses_spinner_when_stdout_is_utf8(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdout", _DummyStdout("utf-8"))
    progress = create_progress()
    assert isinstance(progress.columns[0], SpinnerColumn)

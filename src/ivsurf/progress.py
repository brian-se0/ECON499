"""Shared Rich progress-bar helpers for long-running pipeline stages."""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

BRAILLE_SPINNER_SAMPLE = "⠋"


def _supports_braille_spinner(encoding: str | None) -> bool:
    """Return whether the active stream encoding can emit Rich's braille spinner glyphs."""

    if encoding is None:
        return False
    try:
        BRAILLE_SPINNER_SAMPLE.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def create_progress() -> Progress:
    """Create a consistent progress-bar layout for long-running scripts."""

    leading_column = (
        SpinnerColumn()
        if _supports_braille_spinner(sys.stdout.encoding)
        else TextColumn("  ")
    )
    return Progress(
        leading_column,
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    )


def iter_with_progress[T](
    progress: Progress,
    items: Sequence[T],
    *,
    description: str,
) -> Iterator[T]:
    """Yield items while advancing a fixed-length task."""

    task_id = progress.add_task(description, total=len(items))
    for item in items:
        yield item
        progress.advance(task_id)

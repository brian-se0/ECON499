"""Shared Rich progress-bar helpers for long-running pipeline stages."""

from __future__ import annotations

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


def create_progress() -> Progress:
    """Create a consistent progress-bar layout for long-running scripts."""

    return Progress(
        SpinnerColumn(),
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

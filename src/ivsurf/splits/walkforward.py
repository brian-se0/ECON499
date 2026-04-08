"""Blocked walk-forward split generation."""

from __future__ import annotations

from datetime import date

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.manifests import WalkforwardSplit


def build_walkforward_splits(
    dates: list[date],
    config: WalkforwardConfig,
) -> list[WalkforwardSplit]:
    """Build explicit non-overlapping blocked time-series splits."""

    if len(dates) < (config.train_size + config.validation_size + config.test_size):
        message = "Not enough dates to build one walk-forward split."
        raise ValueError(message)

    splits: list[WalkforwardSplit] = []
    test_start = config.train_size + config.validation_size
    split_number = 0
    while (test_start + config.test_size) <= len(dates):
        validation_start = test_start - config.validation_size
        train_end = validation_start
        train_start = 0 if config.expanding_train else max(0, train_end - config.train_size)

        split = WalkforwardSplit(
            split_id=f"split_{split_number:04d}",
            train_dates=tuple(day.isoformat() for day in dates[train_start:train_end]),
            validation_dates=tuple(day.isoformat() for day in dates[validation_start:test_start]),
            test_dates=tuple(
                day.isoformat()
                for day in dates[test_start : test_start + config.test_size]
            ),
        )
        splits.append(split)
        split_number += 1
        test_start += config.step_size
    return splits

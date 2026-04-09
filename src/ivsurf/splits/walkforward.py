"""Blocked walk-forward split generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.manifests import WalkforwardSplit


@dataclass(frozen=True, slots=True)
class CleanEvaluationBoundary:
    """Boundary between HPO-used validation dates and clean evaluation test dates."""

    max_hpo_validation_date: date
    first_clean_test_split_id: str


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


def clean_evaluation_boundary(
    splits: list[WalkforwardSplit],
    *,
    tuning_splits_count: int,
) -> CleanEvaluationBoundary:
    """Return the first uncontaminated evaluation split after HPO validation windows."""

    if not splits:
        message = "At least one walk-forward split is required."
        raise ValueError(message)
    if tuning_splits_count > len(splits):
        message = (
            "tuning_splits_count cannot exceed the number of available splits: "
            f"{tuning_splits_count} > {len(splits)}."
        )
        raise ValueError(message)

    hpo_validation_dates = [
        date.fromisoformat(day)
        for split in splits[:tuning_splits_count]
        for day in split.validation_dates
    ]
    if not hpo_validation_dates:
        message = "HPO tuning splits must contain at least one validation date."
        raise ValueError(message)
    max_hpo_validation_date = max(hpo_validation_dates)

    for split in splits:
        if not split.test_dates:
            message = f"Walk-forward split {split.split_id} has no test dates."
            raise ValueError(message)
        test_start_date = date.fromisoformat(split.test_dates[0])
        if test_start_date > max_hpo_validation_date:
            return CleanEvaluationBoundary(
                max_hpo_validation_date=max_hpo_validation_date,
                first_clean_test_split_id=split.split_id,
            )

    message = (
        "No clean evaluation split remains after excluding all test windows that overlap the "
        "HPO-used validation sample."
    )
    raise ValueError(message)


def clean_evaluation_splits(
    splits: list[WalkforwardSplit],
    *,
    tuning_splits_count: int,
) -> tuple[CleanEvaluationBoundary, list[WalkforwardSplit]]:
    """Return the boundary metadata and the clean evaluation splits only."""

    boundary = clean_evaluation_boundary(splits, tuning_splits_count=tuning_splits_count)
    clean_splits = [
        split
        for split in splits
        if date.fromisoformat(split.test_dates[0]) > boundary.max_hpo_validation_date
    ]
    if not clean_splits:
        message = "Expected at least one clean evaluation split after HPO boundary filtering."
        raise ValueError(message)
    return boundary, clean_splits

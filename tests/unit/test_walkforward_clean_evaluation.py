from __future__ import annotations

from datetime import date, timedelta

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits


def _dates(count: int) -> list[date]:
    start = date(2021, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def test_official_walkforward_geometry_excludes_contaminated_early_test_splits() -> None:
    splits = build_walkforward_splits(
        dates=_dates(714),
        config=WalkforwardConfig(
            train_size=504,
            validation_size=126,
            test_size=21,
            step_size=21,
            expanding_train=True,
        ),
    )

    boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=3)
    clean_split_ids = [split.split_id for split in clean_splits]
    hpo_validation_dates = {
        day
        for split in splits[:3]
        for day in split.validation_dates
    }
    reported_test_dates = {
        day
        for split in clean_splits
        for day in split.test_dates
    }

    assert boundary.first_clean_test_split_id == "split_0002"
    assert "split_0000" not in clean_split_ids
    assert "split_0001" not in clean_split_ids
    assert clean_split_ids[0] == "split_0002"
    assert reported_test_dates.isdisjoint(hpo_validation_dates)

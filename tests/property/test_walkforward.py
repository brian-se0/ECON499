from __future__ import annotations

from datetime import date, timedelta

from hypothesis import assume, given
from hypothesis import strategies as st

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits


@given(st.integers(min_value=12, max_value=40))
def test_walkforward_splits_are_ordered_and_non_overlapping(length: int) -> None:
    dates = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(length)]
    splits = build_walkforward_splits(
        dates=dates,
        config=WalkforwardConfig(
            train_size=6,
            validation_size=3,
            test_size=2,
            step_size=2,
            expanding_train=True,
        ),
    )
    for split in splits:
        train_set = set(split.train_dates)
        validation_set = set(split.validation_dates)
        test_set = set(split.test_dates)
        assert train_set.isdisjoint(validation_set)
        assert train_set.isdisjoint(test_set)
        assert validation_set.isdisjoint(test_set)
        assert max(split.train_dates) < min(split.validation_dates)
        assert max(split.validation_dates) < min(split.test_dates)


@given(st.integers(min_value=14, max_value=60), st.integers(min_value=1, max_value=3))
def test_clean_evaluation_splits_exclude_hpo_validation_dates(
    length: int,
    tuning_splits_count: int,
) -> None:
    dates = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(length)]
    splits = build_walkforward_splits(
        dates=dates,
        config=WalkforwardConfig(
            train_size=6,
            validation_size=3,
            test_size=2,
            step_size=2,
            expanding_train=True,
        ),
    )
    assume(len(splits) > tuning_splits_count)

    boundary, clean_splits = clean_evaluation_splits(
        splits,
        tuning_splits_count=tuning_splits_count,
    )
    hpo_validation_dates = {
        day
        for split in splits[:tuning_splits_count]
        for day in split.validation_dates
    }
    evaluation_test_dates = {day for split in clean_splits for day in split.test_dates}

    assert clean_splits[0].split_id == boundary.first_clean_test_split_id
    assert evaluation_test_dates.isdisjoint(hpo_validation_dates)
    assert all(
        min(split.test_dates) > boundary.max_hpo_validation_date.isoformat()
        for split in clean_splits
    )

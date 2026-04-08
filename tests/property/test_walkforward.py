from __future__ import annotations

from datetime import date, timedelta

from hypothesis import given
from hypothesis import strategies as st

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.walkforward import build_walkforward_splits


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


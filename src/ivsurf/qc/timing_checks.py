"""Temporal-integrity validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import pairwise

from ivsurf.calendar import MarketCalendar
from ivsurf.exceptions import TemporalIntegrityError


def assert_session_has_decision_time(
    calendar: MarketCalendar,
    session_date: date,
    *,
    context: str,
) -> None:
    """Require a session to include a usable vendor decision snapshot."""

    if not calendar.session_has_decision_time(session_date):
        message = (
            f"{context} requires session {session_date.isoformat()} to contain a usable "
            "vendor decision snapshot."
        )
        raise TemporalIntegrityError(message)


def assert_next_decision_session_alignment(
    calendar: MarketCalendar,
    quote_date: date,
    target_date: date,
) -> None:
    """Require target_date to equal the next observed trading session."""

    expected_target = calendar.next_trading_session(quote_date)
    if expected_target != target_date:
        message = (
            "Feature/target alignment violated next-trading-session causality: "
            f"quote_date={quote_date.isoformat()} "
            f"expected_target_date={expected_target.isoformat()} "
            f"actual_target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)


def assert_strictly_increasing_unique_dates(
    observed_dates: Sequence[date],
    *,
    context: str,
) -> None:
    """Require an observed-date sequence to be unique and strictly increasing."""

    if len(set(observed_dates)) != len(observed_dates):
        message = f"{context} must contain unique dates."
        raise TemporalIntegrityError(message)
    for previous_date, current_date in pairwise(observed_dates):
        if current_date <= previous_date:
            message = (
                f"{context} must be strictly increasing: "
                f"{previous_date.isoformat()} then {current_date.isoformat()}."
            )
            raise TemporalIntegrityError(message)


def assert_next_observed_target_date(
    observed_dates: Sequence[date],
    *,
    position: int,
    quote_date: date,
    target_date: date,
) -> None:
    """Require target_date to equal the next observed gold-surface date."""

    if position < 0 or position >= len(observed_dates) - 1:
        message = (
            "Position for next-observed-date alignment must point to a quote date with a "
            f"subsequent observed target date, found position={position}."
        )
        raise TemporalIntegrityError(message)
    expected_quote_date = observed_dates[position]
    expected_target_date = observed_dates[position + 1]
    if quote_date != expected_quote_date:
        message = (
            "Feature/target alignment quote_date does not match the observed gold-surface "
            f"sequence: expected={expected_quote_date.isoformat()} "
            f"actual={quote_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)
    if target_date <= quote_date:
        message = (
            "Feature/target alignment requires target_date to be after quote_date: "
            f"quote_date={quote_date.isoformat()} "
            f"target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)
    if target_date != expected_target_date:
        message = (
            "Feature/target alignment violated next-observed-date causality: "
            f"quote_date={quote_date.isoformat()} "
            f"expected_target_date={expected_target_date.isoformat()} "
            f"actual_target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)

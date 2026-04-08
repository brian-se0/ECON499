"""Temporal-integrity validation helpers."""

from __future__ import annotations

from datetime import date

from ivsurf.calendar import MarketCalendar
from ivsurf.exceptions import TemporalIntegrityError


def assert_session_has_decision_time(
    calendar: MarketCalendar,
    session_date: date,
    *,
    context: str,
) -> None:
    """Require a session to include the configured 15:45 decision timestamp."""

    if not calendar.session_has_decision_time(session_date):
        message = (
            f"{context} requires session {session_date.isoformat()} to contain the "
            f"{calendar.decision_time.isoformat(timespec='minutes')} decision timestamp."
        )
        raise TemporalIntegrityError(message)


def assert_next_decision_session_alignment(
    calendar: MarketCalendar,
    quote_date: date,
    target_date: date,
) -> None:
    """Require target_date to equal the next session that actually has decision-time coverage."""

    expected_target = calendar.next_decision_session(quote_date)
    if expected_target != target_date:
        message = (
            "Feature/target alignment violated next-decision-session causality: "
            f"quote_date={quote_date.isoformat()} "
            f"expected_target_date={expected_target.isoformat()} "
            f"actual_target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)

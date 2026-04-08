"""Trading-calendar and maturity helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ivsurf.exceptions import TemporalIntegrityError


@dataclass(slots=True)
class MarketCalendar:
    """Explicit market calendar wrapper for session alignment and maturity timing."""

    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    decision_time: time = time(15, 45)
    am_settled_roots: tuple[str, ...] = ("SPX",)
    _calendar: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._calendar = xcals.get_calendar(self.calendar_name)

    def _to_session_label(self, session_date: date) -> pd.Timestamp:
        return pd.Timestamp(session_date)

    def is_session(self, session_date: date) -> bool:
        return bool(self._calendar.is_session(self._to_session_label(session_date)))

    def previous_session(self, session_date: date) -> date:
        label = self._to_session_label(session_date)
        if self.is_session(session_date):
            previous = self._calendar.previous_session(label)
        else:
            previous = self._calendar.date_to_session(label, direction="previous")
        return cast(date, previous.date())

    def next_session(self, session_date: date) -> date:
        label = self._to_session_label(session_date)
        next_value = self._calendar.next_session(label)
        return cast(date, next_value.date())

    def next_decision_session(self, session_date: date) -> date:
        """Return the next session that still contains the configured decision timestamp."""

        next_session = self.next_session(session_date)
        while not self.session_has_decision_time(next_session):
            next_session = self.next_session(next_session)
        return next_session

    def session_has_decision_time(self, session_date: date) -> bool:
        if not self.is_session(session_date):
            return False
        close_ts = self._calendar.session_close(self._to_session_label(session_date))
        close_local = close_ts.tz_convert(self.timezone)
        decision_dt = pd.Timestamp(
            datetime.combine(session_date, self.decision_time, tzinfo=ZoneInfo(self.timezone))
        )
        return bool(close_local >= decision_dt)

    def resolve_last_tradable_session(self, root: str, expiration: date) -> date:
        """Resolve the session on which a contract can last trade."""

        settlement_session = (
            expiration if self.is_session(expiration) else self.previous_session(expiration)
        )
        if root in self.am_settled_roots:
            return self.previous_session(settlement_session)
        return settlement_session

    def compute_tau_years(self, quote_date: date, expiration: date, root: str) -> float:
        """Compute ACT/365 time-to-maturity to last tradable session close."""

        if not self.session_has_decision_time(quote_date):
            message = f"Session {quote_date.isoformat()} does not contain the 15:45 decision time."
            raise TemporalIntegrityError(message)

        last_tradable_session = self.resolve_last_tradable_session(root=root, expiration=expiration)
        if quote_date > last_tradable_session:
            return 0.0

        decision_dt = datetime.combine(
            quote_date,
            self.decision_time,
            tzinfo=ZoneInfo(self.timezone),
        )
        expiry_close_ts = self._calendar.session_close(
            self._to_session_label(last_tradable_session)
        )
        expiry_close = expiry_close_ts.tz_convert(self.timezone).to_pydatetime()
        delta_seconds = (expiry_close - decision_dt).total_seconds()
        tau_years = max(delta_seconds, 0.0) / (365.0 * 24.0 * 60.0 * 60.0)
        return cast(float, tau_years)

    def next_trading_session(self, session_date: date) -> date:
        """Return the next trading session after the provided session date."""

        return self.next_session(session_date)

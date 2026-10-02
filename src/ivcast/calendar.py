"""Trading-calendar and maturity helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ivcast.exceptions import TemporalIntegrityError

SECONDS_PER_YEAR = 365.0 * 24.0 * 60.0 * 60.0


@dataclass(slots=True)
class MarketCalendar:
    """Explicit market calendar wrapper for session alignment and maturity timing.

    Two maturity instants are distinguished:

    - the economic settlement instant (`settlement_datetime`): the opening of the
      settlement session for AM-settled roots, its close for PM-settled roots. This is
      the maturity coordinate used for grid assignment.
    - the vendor implied-volatility reference instant (`vendor_iv_reference_datetime`):
      the maturity the vendor used when it solved the 15:45 implied volatility. The
      vendor treats only the `SPX` root as AM-settled (09:30 ET), and references
      SPX-root expirations dated before 2015 to the session preceding the recorded
      expiration date. Price-consistent total variance is `iv**2 * tau_vendor`.
    """

    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    decision_time: time = time(15, 45)
    decision_snapshot_minutes_before_close: int = 15
    am_settled_roots: tuple[str, ...] = ("SPX",)
    am_settlement_time: time = time(9, 30)
    vendor_am_iv_roots: tuple[str, ...] = ("SPX",)
    vendor_am_iv_previous_session_before: date = date(2015, 1, 1)
    _calendar: Any = field(init=False, repr=False, default=None)
    _calendar_start: date | None = field(init=False, repr=False, default=None)
    _calendar_end: date | None = field(init=False, repr=False, default=None)
    _calendar_padding_days: int = field(default=14, init=False, repr=False)

    def __post_init__(self) -> None:
        return None

    def _rebuild_calendar(self, start: date, end: date) -> None:
        self._calendar = xcals.get_calendar(self.calendar_name, start=start, end=end)
        self._calendar_start = cast(date, self._calendar.first_session.date())
        self._calendar_end = cast(date, self._calendar.last_session.date())

    def _ensure_calendar_bounds(self, *session_dates: date) -> None:
        if not session_dates:
            message = "_ensure_calendar_bounds requires at least one session date."
            raise ValueError(message)

        requested_start = min(session_dates) - timedelta(days=self._calendar_padding_days)
        requested_end = max(session_dates) + timedelta(days=self._calendar_padding_days)
        if self._calendar_start is None or self._calendar_end is None or self._calendar is None:
            self._rebuild_calendar(start=requested_start, end=requested_end)
            return
        if requested_start < self._calendar_start or requested_end > self._calendar_end:
            self._rebuild_calendar(
                start=min(requested_start, self._calendar_start),
                end=max(requested_end, self._calendar_end),
            )

    def _to_session_label(self, session_date: date) -> pd.Timestamp:
        return pd.Timestamp(session_date)

    def is_session(self, session_date: date) -> bool:
        self._ensure_calendar_bounds(session_date)
        return bool(self._calendar.is_session(self._to_session_label(session_date)))

    def previous_session(self, session_date: date) -> date:
        self._ensure_calendar_bounds(session_date)
        label = self._to_session_label(session_date)
        if self.is_session(session_date):
            previous = self._calendar.previous_session(label)
        else:
            previous = self._calendar.date_to_session(label, direction="previous")
        return cast(date, previous.date())

    def next_session(self, session_date: date) -> date:
        self._ensure_calendar_bounds(session_date)
        label = self._to_session_label(session_date)
        next_value = self._calendar.next_session(label)
        return cast(date, next_value.date())

    def next_decision_session(self, session_date: date) -> date:
        """Return the next observed trading session after the provided date."""

        return self.next_session(session_date)

    def _session_close_local(self, session_date: date) -> pd.Timestamp:
        self._ensure_calendar_bounds(session_date)
        if not self.is_session(session_date):
            message = f"{session_date.isoformat()} is not a trading session."
            raise TemporalIntegrityError(message)
        return self._calendar.session_close(self._to_session_label(session_date)).tz_convert(
            self.timezone
        )

    def effective_decision_datetime(self, session_date: date) -> datetime:
        """Return the effective vendor decision snapshot timestamp for one session."""

        close_local = self._session_close_local(session_date)
        configured_decision_dt = datetime.combine(
            session_date,
            self.decision_time,
            tzinfo=ZoneInfo(self.timezone),
        )
        close_buffer_dt = (
            close_local - timedelta(minutes=self.decision_snapshot_minutes_before_close)
        ).to_pydatetime()
        return min(configured_decision_dt, cast(datetime, close_buffer_dt))

    def session_has_decision_time(self, session_date: date) -> bool:
        """Return whether the session carries a usable vendor decision snapshot."""

        self._ensure_calendar_bounds(session_date)
        if not self.is_session(session_date):
            return False
        close_local = self._session_close_local(session_date)
        return bool(close_local.to_pydatetime() >= self.effective_decision_datetime(session_date))

    def settlement_session(self, expiration: date) -> date:
        """Return the session on which a contract settles.

        Recorded expirations that are not sessions (pre-2015 Saturday expirations and
        holiday-adjusted expirations) settle on the preceding session.
        """

        self._ensure_calendar_bounds(expiration)
        return expiration if self.is_session(expiration) else self.previous_session(expiration)

    def resolve_last_tradable_session(self, root: str, expiration: date) -> date:
        """Resolve the session on which a contract can last trade."""

        settlement_session = self.settlement_session(expiration)
        if root in self.am_settled_roots:
            return self.previous_session(settlement_session)
        return settlement_session

    def _am_datetime(self, session_date: date) -> datetime:
        return datetime.combine(
            session_date,
            self.am_settlement_time,
            tzinfo=ZoneInfo(self.timezone),
        )

    def settlement_datetime(self, root: str, expiration: date) -> datetime:
        """Return the economic settlement instant of one contract."""

        settlement_session = self.settlement_session(expiration)
        if root in self.am_settled_roots:
            return self._am_datetime(settlement_session)
        return cast(datetime, self._session_close_local(settlement_session).to_pydatetime())

    def vendor_iv_reference_datetime(self, root: str, expiration: date) -> datetime:
        """Return the maturity instant embedded in the vendor's implied volatility."""

        if root in self.vendor_am_iv_roots:
            if expiration < self.vendor_am_iv_previous_session_before:
                return self._am_datetime(self.previous_session(expiration))
            return self._am_datetime(self.settlement_session(expiration))
        return cast(
            datetime,
            self._session_close_local(self.settlement_session(expiration)).to_pydatetime(),
        )

    def _years_from_decision(self, quote_date: date, maturity_dt: datetime) -> float:
        if not self.session_has_decision_time(quote_date):
            message = (
                "Session "
                f"{quote_date.isoformat()} does not contain a usable decision snapshot."
            )
            raise TemporalIntegrityError(message)
        decision_dt = self.effective_decision_datetime(quote_date)
        delta_seconds = (maturity_dt - decision_dt).total_seconds()
        return max(delta_seconds, 0.0) / SECONDS_PER_YEAR

    def compute_tau_years(self, quote_date: date, expiration: date, root: str) -> float:
        """Compute ACT/365 time-to-maturity from the effective snapshot to settlement."""

        self._ensure_calendar_bounds(quote_date, expiration)
        return self._years_from_decision(
            quote_date,
            self.settlement_datetime(root=root, expiration=expiration),
        )

    def compute_vendor_iv_tau_years(self, quote_date: date, expiration: date, root: str) -> float:
        """Compute ACT/365 time from the effective snapshot to the vendor IV maturity."""

        self._ensure_calendar_bounds(quote_date, expiration)
        return self._years_from_decision(
            quote_date,
            self.vendor_iv_reference_datetime(root=root, expiration=expiration),
        )

    def next_trading_session(self, session_date: date) -> date:
        """Return the next trading session after the provided session date."""

        return self.next_session(session_date)

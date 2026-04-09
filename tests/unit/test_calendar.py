from __future__ import annotations

from datetime import date, time

from ivsurf.calendar import MarketCalendar


def test_early_close_day_uses_effective_vendor_snapshot_time() -> None:
    calendar = MarketCalendar()
    session_date = date(2019, 11, 29)

    assert calendar.session_has_decision_time(session_date) is True
    assert calendar.effective_decision_datetime(session_date).time() == time(12, 45)
    assert calendar.compute_tau_years(
        quote_date=session_date,
        expiration=date(2019, 12, 20),
        root="SPXW",
    ) > 0.0


def test_next_trading_session_skips_holiday() -> None:
    calendar = MarketCalendar()
    assert calendar.next_trading_session(date(2021, 4, 1)) == date(2021, 4, 5)


def test_next_decision_session_includes_early_close() -> None:
    calendar = MarketCalendar()
    assert calendar.next_decision_session(date(2019, 11, 27)) == date(2019, 11, 29)


def test_am_settled_tau_uses_previous_session_before_settlement() -> None:
    calendar = MarketCalendar(am_settled_roots=("SPX",))
    tau_years = calendar.compute_tau_years(
        quote_date=date(2021, 4, 15),
        expiration=date(2021, 4, 16),
        root="SPX",
    )
    assert 0.0 < tau_years < (1.0 / 365.0)


def test_calendar_supports_pre_2006_history() -> None:
    calendar = MarketCalendar()
    session_date = date(2004, 1, 2)

    assert calendar.is_session(session_date) is True
    assert calendar.session_has_decision_time(session_date) is True
    assert calendar.next_trading_session(session_date) == date(2004, 1, 5)

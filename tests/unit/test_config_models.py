from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest

from ivsurf.config import (
    HedgingConfig,
    RawDataConfig,
    ReportArtifactsConfig,
    StatsTestConfig,
    calendar_config_from_raw,
)


def test_calendar_config_from_raw_projects_shared_fields() -> None:
    raw_config = RawDataConfig(
        raw_options_dir=Path("D:/Options Data"),
        bronze_dir=Path("data/bronze"),
        silver_dir=Path("data/silver"),
        gold_dir=Path("data/gold"),
        manifests_dir=Path("data/manifests"),
        target_symbol="^SPX",
        calendar_name="XNYS",
        timezone="America/New_York",
        decision_time=time(15, 45),
        sample_start_date=date(2004, 1, 2),
        sample_end_date=date(2021, 4, 9),
        am_settled_roots=("SPX", "SPXW"),
    )

    calendar_config = calendar_config_from_raw(raw_config)

    assert calendar_config.calendar_name == raw_config.calendar_name
    assert calendar_config.timezone == raw_config.timezone
    assert calendar_config.decision_time == raw_config.decision_time
    assert calendar_config.am_settled_roots == raw_config.am_settled_roots


def test_raw_data_config_rejects_reversed_sample_window() -> None:
    with pytest.raises(ValueError, match="sample_end_date"):
        RawDataConfig(
            raw_options_dir=Path("D:/Options Data"),
            bronze_dir=Path("data/bronze"),
            silver_dir=Path("data/silver"),
            gold_dir=Path("data/gold"),
            manifests_dir=Path("data/manifests"),
            sample_start_date=date(2021, 4, 9),
            sample_end_date=date(2004, 1, 2),
        )


def test_report_artifacts_config_rejects_summary_column_metric_name() -> None:
    with pytest.raises(ValueError, match="base daily loss metric"):
        ReportArtifactsConfig(primary_loss_metric="mean_observed_wrmse_total_variance")


def test_report_artifacts_config_requires_primary_metric_in_official_metrics() -> None:
    with pytest.raises(ValueError, match="official_loss_metrics"):
        ReportArtifactsConfig(
            official_loss_metrics=("observed_qlike_total_variance",),
            primary_loss_metric="observed_mse_total_variance",
        )


def test_stats_test_config_rejects_duplicate_loss_metrics() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        StatsTestConfig(
            loss_metrics=(
                "observed_mse_total_variance",
                "observed_mse_total_variance",
            )
        )


def test_hedging_config_rejects_removed_spot_assumption_key() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        HedgingConfig.model_validate(
            {
                "risk_free_rate": 0.0,
                "level_notional": 1.0,
                "skew_notional": 1.0,
                "calendar_notional": 0.5,
                "skew_moneyness_abs": 0.1,
                "short_maturity_days": 30,
                "long_maturity_days": 90,
                "hedge_maturity_days": 30,
                "hedge_straddle_moneyness": 0.0,
                "hedge_spot_assumption": "no_change",
            }
        )

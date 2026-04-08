from __future__ import annotations

from datetime import date, time
from pathlib import Path

import pytest

from ivsurf.config import RawDataConfig, ReportArtifactsConfig, calendar_config_from_raw


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

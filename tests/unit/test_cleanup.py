from __future__ import annotations

from pathlib import Path

import pytest

from ivsurf.cleanup import build_cleanup_plan, cleanup_stage_names, execute_cleanup_plan
from ivsurf.config import RawDataConfig


def _raw_config(tmp_path: Path) -> RawDataConfig:
    return RawDataConfig(
        raw_options_dir=tmp_path / "raw_options",
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        manifests_dir=tmp_path / "data" / "manifests",
    )


def test_cleanup_stage_names_include_downstream_stages() -> None:
    assert cleanup_stage_names("stats") == ("stats", "hedging", "report")
    assert cleanup_stage_names("all") == (
        "ingest",
        "silver",
        "surfaces",
        "features",
        "hpo",
        "train",
        "stats",
        "hedging",
        "report",
    )


def test_build_cleanup_plan_for_features_includes_downstream_outputs(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)

    plan = build_cleanup_plan(
        raw_config=raw_config,
        selection="features",
        hpo_profile_name="hpo_30_trials",
        training_profile_name="train_30_epochs",
    )

    expected_paths = {
        raw_config.gold_dir / "daily_features.parquet",
        raw_config.manifests_dir / "walkforward_splits.json",
        raw_config.manifests_dir / "tuning" / "hpo_30_trials",
        raw_config.gold_dir / "forecasts" / "hpo_30_trials__train_30_epochs",
        raw_config.manifests_dir / "stats" / "hpo_30_trials__train_30_epochs",
        raw_config.manifests_dir / "hedging" / "hpo_30_trials__train_30_epochs",
        raw_config.manifests_dir / "report_artifacts" / "hpo_30_trials__train_30_epochs",
    }

    assert expected_paths.issubset(set(plan.paths))
    assert raw_config.raw_options_dir not in plan.paths
    assert plan.stage_names == ("features", "hpo", "train", "stats", "hedging", "report")


def test_build_cleanup_plan_rejects_configured_overlap_with_raw_options_dir(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw_options"
    raw_config = RawDataConfig(
        raw_options_dir=raw_root,
        bronze_dir=raw_root / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        manifests_dir=tmp_path / "data" / "manifests",
    )

    with pytest.raises(ValueError, match="protected raw options directory"):
        build_cleanup_plan(
            raw_config=raw_config,
            selection="ingest",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
        )


def test_execute_cleanup_plan_removes_derived_outputs_but_preserves_raw_options_dir(
    tmp_path: Path,
) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.raw_options_dir.mkdir(parents=True)
    protected_raw_file = raw_config.raw_options_dir / "UnderlyingOptionsEODCalcs_2004-01-02.zip"
    protected_raw_file.write_bytes(b"raw")

    for path in (
        raw_config.gold_dir / "daily_features.parquet",
        raw_config.manifests_dir / "walkforward_splits.json",
        raw_config.manifests_dir / "tuning" / "hpo_30_trials" / "ridge.json",
        raw_config.gold_dir / "forecasts" / "hpo_30_trials__train_30_epochs" / "ridge.parquet",
        raw_config.manifests_dir
        / "stats"
        / "hpo_30_trials__train_30_epochs"
        / "daily_loss_frame.parquet",
        raw_config.manifests_dir
        / "hedging"
        / "hpo_30_trials__train_30_epochs"
        / "hedging_results.parquet",
        raw_config.manifests_dir
        / "report_artifacts"
        / "hpo_30_trials__train_30_epochs"
        / "overview.md",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"derived")

    plan = build_cleanup_plan(
        raw_config=raw_config,
        selection="features",
        hpo_profile_name="hpo_30_trials",
        training_profile_name="train_30_epochs",
    )

    removed_paths = execute_cleanup_plan(plan, raw_config=raw_config)

    assert removed_paths
    assert protected_raw_file.exists()
    assert not (raw_config.gold_dir / "daily_features.parquet").exists()
    assert not (raw_config.gold_dir / "forecasts" / "hpo_30_trials__train_30_epochs").exists()
    assert not (raw_config.manifests_dir / "tuning" / "hpo_30_trials").exists()

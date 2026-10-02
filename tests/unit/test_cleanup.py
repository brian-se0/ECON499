from __future__ import annotations

from pathlib import Path

import pytest

from ivcast.cleanup import (
    CleanupPlan,
    build_cleanup_plan,
    cleanup_stage_names,
    execute_cleanup_plan,
)
from ivcast.config import RawDataConfig


def _raw_config(tmp_path: Path) -> RawDataConfig:
    return RawDataConfig(
        raw_options_dir=tmp_path / "raw_options",
        bronze_dir=tmp_path / "data" / "bronze",
        silver_dir=tmp_path / "data" / "silver",
        gold_dir=tmp_path / "data" / "gold",
        manifests_dir=tmp_path / "data" / "manifests",
    )


def _directory_symlink(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Creating directory symlinks is unavailable: {error}")


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


@pytest.mark.parametrize("overlap", ("same", "child", "ancestor"))
def test_cleanup_rejects_raw_overlap_through_parent_symlink(
    tmp_path: Path,
    overlap: str,
) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.raw_options_dir.mkdir()
    protected = raw_config.raw_options_dir / "source.zip"
    protected.write_bytes(b"raw")
    alias = tmp_path / "alias"
    _directory_symlink(alias, tmp_path)
    roots = {
        "same": alias / "raw_options",
        "child": alias / "raw_options" / "bronze",
        "ancestor": alias,
    }
    raw_config = raw_config.model_copy(update={"bronze_dir": roots[overlap]})

    with pytest.raises(ValueError, match="protected raw options directory"):
        build_cleanup_plan(
            raw_config=raw_config,
            selection="all",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
        )

    assert protected.read_bytes() == b"raw"


def test_cleanup_resolves_the_protected_raw_directory_itself(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.bronze_dir.mkdir(parents=True)
    protected = raw_config.bronze_dir / "source.zip"
    protected.write_bytes(b"raw")
    _directory_symlink(raw_config.raw_options_dir, raw_config.bronze_dir)

    with pytest.raises(ValueError, match="protected raw options directory"):
        build_cleanup_plan(
            raw_config=raw_config,
            selection="all",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
        )

    assert protected.read_bytes() == b"raw"


def test_cleanup_rejects_raw_overlap_through_case_alias(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.raw_options_dir.mkdir()
    case_alias = raw_config.raw_options_dir.with_name("RAW_OPTIONS")
    if not case_alias.exists():
        pytest.skip("The test volume uses case-sensitive directory names")
    protected = raw_config.raw_options_dir / "source.zip"
    protected.write_bytes(b"raw")
    raw_config = raw_config.model_copy(update={"bronze_dir": case_alias / "bronze"})

    with pytest.raises(ValueError, match="protected raw options directory"):
        build_cleanup_plan(
            raw_config=raw_config,
            selection="all",
            hpo_profile_name="hpo_30_trials",
            training_profile_name="train_30_epochs",
        )

    assert protected.read_bytes() == b"raw"


def test_cleanup_validates_all_targets_before_deleting_any_path(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.gold_dir.mkdir(parents=True)
    protected_dir = raw_config.raw_options_dir / "kept"
    protected_dir.mkdir(parents=True)
    protected = protected_dir / "source.zip"
    protected.write_bytes(b"raw")
    ordinary = raw_config.gold_dir / "daily_features.parquet"
    ordinary.write_bytes(b"derived")
    alias = raw_config.gold_dir / "redirected"
    _directory_symlink(alias, raw_config.raw_options_dir)
    plan = CleanupPlan(
        selection="features",
        stage_names=("features",),
        protected_raw_options_dir=raw_config.raw_options_dir,
        paths=(ordinary, alias / "kept"),
    )

    with pytest.raises(ValueError, match="protected raw options directory"):
        execute_cleanup_plan(plan, raw_config=raw_config)

    assert ordinary.read_bytes() == b"derived"
    assert protected.read_bytes() == b"raw"


def test_cleanup_revalidates_redirected_parent_after_planning(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)
    protected_dir = raw_config.raw_options_dir / "hpo_30_trials__train_30_epochs"
    protected_dir.mkdir(parents=True)
    protected = protected_dir / "source.zip"
    protected.write_bytes(b"raw")
    plan = build_cleanup_plan(
        raw_config=raw_config,
        selection="stats",
        hpo_profile_name="hpo_30_trials",
        training_profile_name="train_30_epochs",
    )
    raw_config.manifests_dir.mkdir(parents=True)
    _directory_symlink(raw_config.manifests_dir / "stats", raw_config.raw_options_dir)

    with pytest.raises(ValueError, match="protected raw options directory"):
        execute_cleanup_plan(plan, raw_config=raw_config)

    assert protected.read_bytes() == b"raw"


def test_cleanup_does_not_follow_links_inside_a_derived_directory(tmp_path: Path) -> None:
    raw_config = _raw_config(tmp_path)
    raw_config.raw_options_dir.mkdir()
    protected = raw_config.raw_options_dir / "source.zip"
    protected.write_bytes(b"raw")
    raw_config.bronze_dir.mkdir(parents=True)
    _directory_symlink(raw_config.bronze_dir / "raw_link", raw_config.raw_options_dir)
    plan = build_cleanup_plan(
        raw_config=raw_config,
        selection="ingest",
        hpo_profile_name="hpo_30_trials",
        training_profile_name="train_30_epochs",
    )

    execute_cleanup_plan(plan, raw_config=raw_config)

    assert not raw_config.bronze_dir.exists()
    assert protected.read_bytes() == b"raw"

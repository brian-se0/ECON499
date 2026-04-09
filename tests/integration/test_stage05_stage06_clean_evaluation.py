from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from itertools import pairwise
from pathlib import Path
from types import ModuleType

import polars as pl
from pytest import MonkeyPatch

from ivsurf.calendar import MarketCalendar
from ivsurf.config import WalkforwardConfig
from ivsurf.splits.manifests import serialize_splits
from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits
from ivsurf.training.tuning import load_tuning_result


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _calendar_dates(count: int) -> list[date]:
    start = date(2021, 1, 1)
    return [start + timedelta(days=offset) for offset in range(count)]


def _feature_frame(row_count: int) -> pl.DataFrame:
    dates = _calendar_dates(row_count + 1)
    rows = []
    for index in range(row_count):
        level = 0.0100 + (0.0001 * index)
        rows.append(
            {
                "quote_date": dates[index],
                "target_date": dates[index + 1],
                "feature_surface_mean_01_0000": level,
                "feature_surface_mean_01_0001": level + 0.0010,
                "target_total_variance_0000": level + 0.0002,
                "target_total_variance_0001": level + 0.0012,
                "target_observed_mask_0000": 1.0,
                "target_observed_mask_0001": 1.0,
                "target_vega_weight_0000": 1.0,
                "target_vega_weight_0001": 1.0,
            }
        )
    return pl.DataFrame(rows)


def _observed_quote_dates(count: int, *, skipped_session_indices: set[int]) -> list[date]:
    calendar = MarketCalendar()
    observed_dates: list[date] = []
    session_date = date(2021, 1, 4)
    session_index = 0
    while len(observed_dates) < count:
        if session_index not in skipped_session_indices:
            observed_dates.append(session_date)
        session_date = calendar.next_trading_session(session_date)
        session_index += 1
    return observed_dates


def _feature_frame_from_observed_dates(observed_dates: list[date]) -> pl.DataFrame:
    rows = []
    for index, (quote_date, target_date) in enumerate(pairwise(observed_dates)):
        level = 0.0100 + (0.0001 * index)
        rows.append(
            {
                "quote_date": quote_date,
                "target_date": target_date,
                "target_gap_sessions": 0,
                "feature_surface_mean_01_0000": level,
                "feature_surface_mean_01_0001": level + 0.0010,
                "target_total_variance_0000": level + 0.0002,
                "target_total_variance_0001": level + 0.0012,
                "target_observed_mask_0000": 1.0,
                "target_observed_mask_0001": 1.0,
                "target_vega_weight_0000": 1.0,
                "target_vega_weight_0001": 1.0,
            }
        )
    return pl.DataFrame(rows)


def test_stage05_and_stage06_emit_only_clean_evaluation_forecasts(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_dir = tmp_path / "data" / "gold"
    manifests_dir = tmp_path / "data" / "manifests"
    feature_frame = _feature_frame(row_count=714)
    gold_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    feature_frame.write_parquet(gold_dir / "daily_features.parquet")

    walkforward_config = WalkforwardConfig(
        train_size=504,
        validation_size=126,
        test_size=21,
        step_size=21,
        expanding_train=True,
    )
    splits = build_walkforward_splits(
        dates=feature_frame["quote_date"].to_list(),
        config=walkforward_config,
    )
    split_manifest_path = manifests_dir / "walkforward_splits.json"
    serialize_splits(splits, split_manifest_path)
    boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=3)
    expected_clean_quote_dates = {
        date.fromisoformat(day)
        for split in clean_splits
        for day in split.test_dates
    }

    raw_config_path = _write_text(
        tmp_path / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-01"\n'
            'sample_end_date: "2023-12-31"\n'
        ),
    )
    surface_config_path = _write_text(
        tmp_path / "configs" / "data" / "surface.yaml",
        (
            "moneyness_points: [0.0, 0.1]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    hpo_profile_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "hpo_test.yaml",
        (
            'profile_name: "hpo_test"\n'
            "n_trials: 1\n"
            "tuning_splits_count: 3\n"
            "seed: 7\n"
            'sampler: "tpe"\n'
            "pruner:\n"
            '  name: "median"\n'
            "  n_startup_trials: 0\n"
            "  n_warmup_steps: 0\n"
            "  interval_steps: 1\n"
        ),
    )
    training_profile_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "train_test.yaml",
        (
            'profile_name: "train_test"\n'
            "epochs: 1\n"
            "neural_early_stopping_patience: 1\n"
            "neural_early_stopping_min_delta: 0.0\n"
            "lightgbm_early_stopping_rounds: 1\n"
            "lightgbm_early_stopping_min_delta: 0.0\n"
            "lightgbm_first_metric_only: true\n"
        ),
    )

    stage05 = _load_script_module(
        repo_root / "scripts" / "05_tune_models.py",
        "stage05_clean_eval",
    )
    stage06 = _load_script_module(
        repo_root / "scripts" / "06_run_walkforward.py",
        "stage06_clean_eval",
    )
    monkeypatch.setattr(stage06, "TUNABLE_MODEL_NAMES", ("ridge",))

    stage05.main(
        model_name="ridge",
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
        neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
        hpo_profile_config_path=hpo_profile_config_path,
        training_profile_config_path=training_profile_config_path,
    )

    tuning_result = load_tuning_result(manifests_dir / "tuning" / "hpo_test" / "ridge.json")
    assert tuning_result.max_hpo_validation_date == boundary.max_hpo_validation_date
    assert tuning_result.first_clean_test_split_id == boundary.first_clean_test_split_id

    stage06.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        ridge_config_path=repo_root / "configs" / "models" / "ridge.yaml",
        elasticnet_config_path=repo_root / "configs" / "models" / "elasticnet.yaml",
        har_config_path=repo_root / "configs" / "models" / "har_factor.yaml",
        lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
        random_forest_config_path=repo_root / "configs" / "models" / "random_forest.yaml",
        neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
        hpo_profile_config_path=hpo_profile_config_path,
        training_profile_config_path=training_profile_config_path,
    )

    forecast_dir = gold_dir / "forecasts" / "hpo_test__train_test"
    ridge_forecast = pl.read_parquet(forecast_dir / "ridge.parquet")
    no_change_forecast = pl.read_parquet(forecast_dir / "no_change.parquet")

    for forecast_frame in (ridge_forecast, no_change_forecast):
        assert set(forecast_frame["quote_date"].to_list()) == expected_clean_quote_dates
        assert (
            min(forecast_frame["quote_date"].to_list()) > boundary.max_hpo_validation_date
        )


def test_stage05_and_stage06_preserve_target_dates_when_feature_rows_have_gaps(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_dir = tmp_path / "data" / "gold"
    manifests_dir = tmp_path / "data" / "manifests"
    observed_dates = _observed_quote_dates(21, skipped_session_indices={4, 5, 12})
    feature_frame = _feature_frame_from_observed_dates(observed_dates)
    gold_dir.mkdir(parents=True)
    manifests_dir.mkdir(parents=True)
    feature_frame.write_parquet(gold_dir / "daily_features.parquet")

    walkforward_config = WalkforwardConfig(
        train_size=6,
        validation_size=2,
        test_size=1,
        step_size=1,
        expanding_train=True,
    )
    splits = build_walkforward_splits(
        dates=feature_frame["quote_date"].to_list(),
        config=walkforward_config,
    )
    split_manifest_path = manifests_dir / "walkforward_splits.json"
    serialize_splits(splits, split_manifest_path)
    boundary, clean_splits = clean_evaluation_splits(splits, tuning_splits_count=2)
    expected_clean_quote_dates = {
        date.fromisoformat(day)
        for split in clean_splits
        for day in split.test_dates
    }
    expected_target_by_quote_date = {
        row["quote_date"]: row["target_date"]
        for row in feature_frame.iter_rows(named=True)
        if row["quote_date"] in expected_clean_quote_dates
    }

    raw_config_path = _write_text(
        tmp_path / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-03-31"\n'
        ),
    )
    surface_config_path = _write_text(
        tmp_path / "configs" / "data" / "surface.yaml",
        (
            "moneyness_points: [0.0, 0.1]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    hpo_profile_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "hpo_gap_test.yaml",
        (
            'profile_name: "hpo_gap_test"\n'
            "n_trials: 1\n"
            "tuning_splits_count: 2\n"
            "seed: 7\n"
            'sampler: "tpe"\n'
            "pruner:\n"
            '  name: "median"\n'
            "  n_startup_trials: 0\n"
            "  n_warmup_steps: 0\n"
            "  interval_steps: 1\n"
        ),
    )
    training_profile_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "train_gap_test.yaml",
        (
            'profile_name: "train_gap_test"\n'
            "epochs: 1\n"
            "neural_early_stopping_patience: 1\n"
            "neural_early_stopping_min_delta: 0.0\n"
            "lightgbm_early_stopping_rounds: 1\n"
            "lightgbm_early_stopping_min_delta: 0.0\n"
            "lightgbm_first_metric_only: true\n"
        ),
    )

    stage05 = _load_script_module(
        repo_root / "scripts" / "05_tune_models.py",
        "stage05_gap_eval",
    )
    stage06 = _load_script_module(
        repo_root / "scripts" / "06_run_walkforward.py",
        "stage06_gap_eval",
    )
    monkeypatch.setattr(stage06, "TUNABLE_MODEL_NAMES", ("ridge",))

    stage05.main(
        model_name="ridge",
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
        neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
        hpo_profile_config_path=hpo_profile_config_path,
        training_profile_config_path=training_profile_config_path,
    )
    stage06.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        ridge_config_path=repo_root / "configs" / "models" / "ridge.yaml",
        elasticnet_config_path=repo_root / "configs" / "models" / "elasticnet.yaml",
        har_config_path=repo_root / "configs" / "models" / "har_factor.yaml",
        lightgbm_config_path=repo_root / "configs" / "models" / "lightgbm.yaml",
        random_forest_config_path=repo_root / "configs" / "models" / "random_forest.yaml",
        neural_config_path=repo_root / "configs" / "models" / "neural_surface.yaml",
        hpo_profile_config_path=hpo_profile_config_path,
        training_profile_config_path=training_profile_config_path,
    )

    forecast_dir = gold_dir / "forecasts" / "hpo_gap_test__train_gap_test"
    ridge_forecast = pl.read_parquet(forecast_dir / "ridge.parquet")
    no_change_forecast = pl.read_parquet(forecast_dir / "no_change.parquet")

    for forecast_frame in (ridge_forecast, no_change_forecast):
        unique_pairs = (
            forecast_frame.select("quote_date", "target_date")
            .unique()
            .sort("quote_date")
            .iter_rows(named=True)
        )
        observed_pairs = {row["quote_date"]: row["target_date"] for row in unique_pairs}
        assert set(observed_pairs) == expected_clean_quote_dates
        assert observed_pairs == expected_target_by_quote_date
        assert any(
            (target_date - quote_date).days > 1
            for quote_date, target_date in observed_pairs.items()
        )
        assert min(observed_pairs) > boundary.max_hpo_validation_date

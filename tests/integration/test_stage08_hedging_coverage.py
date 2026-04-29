from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl
import pytest

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN
from ivsurf.features.availability import TARGET_DECISION_TIMESTAMP_COLUMN
from ivsurf.hedging.pnl import summarize_hedging_results
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)
from ivsurf.surfaces.interpolation import (
    COMPLETED_SURFACE_SCHEMA_VERSION,
    COMPLETION_STATUS_OBSERVED,
)
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES
from ivsurf.workflow import workflow_run_label

GRID = SurfaceGrid(maturity_days=(1, 30, 90), moneyness_points=(-0.1, 0.0, 0.1))
SURFACE_CONFIG_HASH = "surface-hash"
HPO_PROFILE = "hpo_test"
TRAINING_PROFILE = "train_test"
WORKFLOW_LABEL = workflow_run_label(HPO_PROFILE, TRAINING_PROFILE)
DATE_PAIRS = (
    (date(2021, 1, 4), date(2021, 1, 5)),
    (date(2021, 1, 5), date(2021, 1, 6)),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_text(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return path


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return path


def _surface_rows(quote_date: date, sigma_level: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate(GRID.maturity_days):
        maturity_years = maturity_days / 365.0
        for moneyness_index, moneyness_point in enumerate(GRID.moneyness_points):
            total_variance = sigma_level * maturity_years + 0.00001 * abs(moneyness_point)
            rows.append(
                {
                    "quote_date": quote_date,
                    DECISION_TIMESTAMP_COLUMN: f"{quote_date.isoformat()}T15:45:00-05:00",
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": total_variance,
                    "observed_iv": float((total_variance / maturity_years) ** 0.5),
                    "completed_total_variance": total_variance,
                    "completed_iv": float((total_variance / maturity_years) ** 0.5),
                    "observed_mask": True,
                    "completion_status": COMPLETION_STATUS_OBSERVED,
                    "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                    "surface_grid_hash": GRID.grid_hash,
                    "maturity_coordinate": MATURITY_COORDINATE,
                    "moneyness_coordinate": MONEYNESS_COORDINATE,
                    "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
                    "surface_config_hash": SURFACE_CONFIG_HASH,
                    "vega_sum": 1.0,
                }
            )
    return rows


def _forecast_rows(
    *,
    model_name: str,
    date_pairs: tuple[tuple[date, date], ...],
    scale: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for quote_date, target_date in date_pairs:
        for maturity_index, maturity_days in enumerate(GRID.maturity_days):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate(GRID.moneyness_points):
                total_variance = scale * 0.05 * maturity_years + 0.00001 * abs(moneyness_point)
                rows.append(
                    {
                        "model_name": model_name,
                        "quote_date": quote_date,
                        "target_date": target_date,
                        "split_id": "split_0000",
                        DECISION_TIMESTAMP_COLUMN: (
                            f"{quote_date.isoformat()}T15:45:00-05:00"
                        ),
                        TARGET_DECISION_TIMESTAMP_COLUMN: (
                            f"{target_date.isoformat()}T15:45:00-05:00"
                        ),
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "surface_grid_schema_version": SURFACE_GRID_SCHEMA_VERSION,
                        "surface_grid_hash": GRID.grid_hash,
                        "maturity_coordinate": MATURITY_COORDINATE,
                        "moneyness_coordinate": MONEYNESS_COORDINATE,
                        "target_surface_version": COMPLETED_SURFACE_SCHEMA_VERSION,
                        "surface_config_hash": SURFACE_CONFIG_HASH,
                        "model_config_hash": f"{model_name}-model-hash",
                        "training_run_id": f"{model_name}-training-run",
                        "predicted_total_variance": float(total_variance),
                    }
                )
    return rows


def _write_stage_configs(tmp_path: Path) -> dict[str, Path]:
    raw_config_path = _write_text(
        tmp_path / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{(tmp_path / 'data' / 'manifests').as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-06"\n'
        ),
    )
    surface_config_path = _write_text(
        tmp_path / "configs" / "data" / "surface.yaml",
        (
            "moneyness_points: [-0.1, 0.0, 0.1]\n"
            "maturity_days: [1, 30, 90]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    metrics_config_path = _write_text(
        tmp_path / "configs" / "eval" / "metrics.yaml",
        (
            "positive_floor: 1.0e-8\n"
            'primary_loss_metric: "observed_mse_total_variance"\n'
        ),
    )
    hedging_config_path = _write_text(
        tmp_path / "configs" / "eval" / "hedging.yaml",
        (
            "risk_free_rate: 0.0\n"
            "level_notional: 1.0\n"
            "skew_notional: 1.0\n"
            "calendar_notional: 0.5\n"
            "skew_moneyness_abs: 0.05\n"
            "short_maturity_days: 30\n"
            "long_maturity_days: 90\n"
            "hedge_maturity_days: 30\n"
            "hedge_straddle_moneyness: 0.0\n"
        ),
    )
    hpo_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "hpo_test.yaml",
        (
            f'profile_name: "{HPO_PROFILE}"\n'
            "n_trials: 1\n"
            "tuning_splits_count: 1\n"
            "seed: 7\n"
        ),
    )
    training_config_path = _write_text(
        tmp_path / "configs" / "workflow" / "train_test.yaml",
        (
            f'profile_name: "{TRAINING_PROFILE}"\n'
            "epochs: 1\n"
            "neural_early_stopping_patience: 1\n"
        ),
    )
    return {
        "raw": raw_config_path,
        "surface": surface_config_path,
        "metrics": metrics_config_path,
        "hedging": hedging_config_path,
        "hpo": hpo_config_path,
        "training": training_config_path,
    }


def _write_stage_inputs(
    tmp_path: Path,
    *,
    ridge_date_pairs: tuple[tuple[date, date], ...],
) -> dict[str, Path]:
    gold_year = tmp_path / "data" / "gold" / "year=2021"
    gold_year.mkdir(parents=True)
    for quote_date, sigma_level in (
        (date(2021, 1, 4), 0.040),
        (date(2021, 1, 5), 0.045),
        (date(2021, 1, 6), 0.050),
    ):
        pl.DataFrame(_surface_rows(quote_date, sigma_level)).write_parquet(
            gold_year / f"{quote_date.isoformat()}.parquet"
        )

    silver_year = tmp_path / "data" / "silver" / "year=2021"
    silver_year.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": quote_date,
                "active_underlying_price_1545": spot,
                "is_valid_observation": True,
            }
            for quote_date, spot in (
                (date(2021, 1, 4), 3700.0),
                (date(2021, 1, 5), 3712.0),
                (date(2021, 1, 6), 3724.0),
            )
        ]
    ).write_parquet(silver_year / "spots.parquet")

    forecast_dir = tmp_path / "data" / "gold" / "forecasts" / WORKFLOW_LABEL
    forecast_dir.mkdir(parents=True)
    pl.DataFrame(
        _forecast_rows(model_name="naive", date_pairs=DATE_PAIRS, scale=1.0)
    ).write_parquet(forecast_dir / "naive.parquet")
    pl.DataFrame(
        _forecast_rows(model_name="ridge", date_pairs=ridge_date_pairs, scale=0.97)
    ).write_parquet(forecast_dir / "ridge.parquet")

    manifests_dir = tmp_path / "data" / "manifests"
    _write_json(manifests_dir / "gold_surface_summary.json", [{"quote_date": "2021-01-04"}])
    _write_json(manifests_dir / "silver_build_summary.json", [{"quote_date": "2021-01-04"}])
    for model_name in TUNABLE_MODEL_NAMES:
        _write_json(
            manifests_dir / "tuning" / HPO_PROFILE / f"{model_name}.json",
            {
                "schema_version": "tuning_result_v2",
                "model_name": model_name,
                "hpo_profile_name": HPO_PROFILE,
                "training_profile_name": TRAINING_PROFILE,
                "primary_loss_metric": "observed_mse_total_variance",
                "best_value": 0.0,
                "best_params": {},
                "n_trials_requested": 1,
                "n_trials_completed": 1,
                "n_trials_pruned": 0,
                "tuning_splits_count": 1,
                "max_hpo_validation_date": "2021-01-01",
                "first_clean_test_split_id": "split_0000",
                "seed": 7,
                "sampler": "tpe",
                "pruner": "median",
            },
        )

    return {
        "forecast_dir": forecast_dir,
        "manifests_dir": manifests_dir,
        "hedging_dir": manifests_dir / "hedging" / WORKFLOW_LABEL,
        "stats_dir": manifests_dir / "stats" / WORKFLOW_LABEL,
        "report_dir": manifests_dir / "report_artifacts" / WORKFLOW_LABEL,
    }


def _write_minimal_stage09_stats(stats_dir: Path) -> None:
    stats_dir.mkdir(parents=True)
    pl.DataFrame({"model_name": ["naive"]}).write_parquet(
        stats_dir / "forecast_realization_panel.parquet"
    )
    pl.DataFrame({"model_name": ["naive"]}).write_parquet(stats_dir / "daily_loss_frame.parquet")
    pl.DataFrame({"model_name": ["naive"]}).write_parquet(stats_dir / "loss_summary.parquet")
    _write_json(stats_dir / "dm_results.json", [])
    _write_json(stats_dir / "spa_result.json", [])
    _write_json(stats_dir / "mcs_result.json", [])


def test_stage08_rejects_unequal_model_forecast_coverage_before_outputs(
    tmp_path: Path,
) -> None:
    config_paths = _write_stage_configs(tmp_path)
    paths = _write_stage_inputs(tmp_path, ridge_date_pairs=(DATE_PAIRS[0],))
    stage08 = _load_script_module(_repo_root() / "scripts" / "08_run_hedging_eval.py", "stage08")

    with pytest.raises(ValueError, match="identical coverage across models"):
        stage08.main(
            raw_config_path=config_paths["raw"],
            surface_config_path=config_paths["surface"],
            metrics_config_path=config_paths["metrics"],
            hedging_config_path=config_paths["hedging"],
            hpo_profile_config_path=config_paths["hpo"],
            training_profile_config_path=config_paths["training"],
        )

    assert not (paths["hedging_dir"] / "hedging_results.parquet").exists()
    assert not (paths["hedging_dir"] / "hedging_summary.parquet").exists()
    assert not (paths["hedging_dir"] / "by_model" / "naive.parquet").exists()
    assert not (paths["hedging_dir"] / "by_model" / "ridge.parquet").exists()


def test_stage08_accepts_equal_model_forecast_coverage(tmp_path: Path) -> None:
    config_paths = _write_stage_configs(tmp_path)
    paths = _write_stage_inputs(tmp_path, ridge_date_pairs=DATE_PAIRS)
    stage08 = _load_script_module(_repo_root() / "scripts" / "08_run_hedging_eval.py", "stage08")

    stage08.main(
        raw_config_path=config_paths["raw"],
        surface_config_path=config_paths["surface"],
        metrics_config_path=config_paths["metrics"],
        hedging_config_path=config_paths["hedging"],
        hpo_profile_config_path=config_paths["hpo"],
        training_profile_config_path=config_paths["training"],
    )

    summary = pl.read_parquet(paths["hedging_dir"] / "hedging_summary.parquet").sort("model_name")
    assert summary["model_name"].to_list() == ["naive", "ridge"]
    assert summary["n_trades"].to_list() == [2, 2]
    assert (paths["hedging_dir"] / "hedging_results.parquet").exists()


def test_stage09_rejects_hedging_results_with_unequal_forecast_coverage(
    tmp_path: Path,
) -> None:
    config_paths = _write_stage_configs(tmp_path)
    paths = _write_stage_inputs(tmp_path, ridge_date_pairs=DATE_PAIRS)
    forecast_frame = pl.concat(
        [
            pl.read_parquet(paths["forecast_dir"] / "naive.parquet"),
            pl.read_parquet(paths["forecast_dir"] / "ridge.parquet"),
        ]
    )
    _write_minimal_stage09_stats(paths["stats_dir"])

    hedging_dir = paths["hedging_dir"]
    hedging_dir.mkdir(parents=True)
    hedging_results = pl.DataFrame(
        [
            {
                "model_name": "naive",
                "quote_date": quote_date,
                "target_date": target_date,
                "abs_revaluation_error": 0.1,
                "squared_revaluation_error": 0.01,
                "abs_hedged_pnl": 0.2,
                "squared_hedged_pnl": 0.04,
                "hedged_pnl": 0.2,
            }
            for quote_date, target_date in DATE_PAIRS
        ]
        + [
            {
                "model_name": "ridge",
                "quote_date": DATE_PAIRS[0][0],
                "target_date": DATE_PAIRS[0][1],
                "abs_revaluation_error": 0.1,
                "squared_revaluation_error": 0.01,
                "abs_hedged_pnl": 0.2,
                "squared_hedged_pnl": 0.04,
                "hedged_pnl": 0.2,
            }
        ]
    ).sort(["model_name", "quote_date", "target_date"])
    hedging_results.write_parquet(hedging_dir / "hedging_results.parquet")
    summarize_hedging_results(hedging_results).write_parquet(
        hedging_dir / "hedging_summary.parquet"
    )
    assert forecast_frame.select("model_name", "quote_date", "target_date").unique().height == 4

    stage09 = _load_script_module(
        _repo_root() / "scripts" / "09_make_report_artifacts.py",
        "stage09",
    )

    with pytest.raises(ValueError, match="Hedging results must match"):
        stage09.main(
            raw_config_path=config_paths["raw"],
            surface_config_path=config_paths["surface"],
            metrics_config_path=config_paths["metrics"],
            hpo_profile_config_path=config_paths["hpo"],
            training_profile_config_path=config_paths["training"],
        )

    assert not (paths["report_dir"] / "tables" / "ranked_hedging_summary.csv").exists()


def test_stage09_rejects_stale_hedging_summary_metrics_with_equal_coverage(
    tmp_path: Path,
) -> None:
    config_paths = _write_stage_configs(tmp_path)
    paths = _write_stage_inputs(tmp_path, ridge_date_pairs=DATE_PAIRS)
    _write_minimal_stage09_stats(paths["stats_dir"])

    hedging_dir = paths["hedging_dir"]
    hedging_dir.mkdir(parents=True)
    hedging_results = pl.DataFrame(
        [
            {
                "model_name": model_name,
                "quote_date": quote_date,
                "target_date": target_date,
                "abs_revaluation_error": 0.1 if model_name == "naive" else 0.2,
                "squared_revaluation_error": 0.01 if model_name == "naive" else 0.04,
                "abs_hedged_pnl": 0.3 if model_name == "naive" else 0.4,
                "squared_hedged_pnl": 0.09 if model_name == "naive" else 0.16,
                "hedged_pnl": 0.3 if model_name == "naive" else -0.4,
            }
            for model_name in ("naive", "ridge")
            for quote_date, target_date in DATE_PAIRS
        ]
    ).sort(["model_name", "quote_date", "target_date"])
    hedging_results.write_parquet(hedging_dir / "hedging_results.parquet")
    stale_summary = summarize_hedging_results(hedging_results).with_columns(
        pl.when(pl.col("model_name") == "ridge")
        .then(pl.col("mean_abs_revaluation_error") + 1.0)
        .otherwise(pl.col("mean_abs_revaluation_error"))
        .alias("mean_abs_revaluation_error")
    )
    stale_summary.write_parquet(hedging_dir / "hedging_summary.parquet")

    stage09 = _load_script_module(
        _repo_root() / "scripts" / "09_make_report_artifacts.py",
        "stage09_stale_summary",
    )

    with pytest.raises(ValueError, match="recomputed hedging result aggregate"):
        stage09.main(
            raw_config_path=config_paths["raw"],
            surface_config_path=config_paths["surface"],
            metrics_config_path=config_paths["metrics"],
            hpo_profile_config_path=config_paths["hpo"],
            training_profile_config_path=config_paths["training"],
        )

    assert not (paths["report_dir"] / "tables" / "ranked_hedging_summary.csv").exists()

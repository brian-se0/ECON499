from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _surface_rows(quote_date: date, sigma: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        maturity_years = maturity_days / 365.0
        total_variance = (sigma * sigma) * maturity_years
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            rows.append(
                {
                    "quote_date": quote_date,
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "observed_total_variance": total_variance,
                    "observed_iv": sigma,
                    "completed_total_variance": total_variance,
                    "completed_iv": sigma,
                    "observed_mask": True,
                    "vega_sum": 1.0,
                }
            )
    return rows


def test_stage07_floors_negative_forecast_total_variance_before_iv_conversion(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_dir = tmp_path / "data" / "gold"
    gold_year = gold_dir / "year=2021"
    gold_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    workflow_label = "test_hpo__test_train"
    forecast_dir = gold_dir / "forecasts" / workflow_label
    forecast_dir.mkdir(parents=True)

    gold_summary = []
    for quote_date, sigma in ((date(2021, 1, 4), 0.20), (date(2021, 1, 5), 0.22)):
        rows = _surface_rows(quote_date, sigma)
        output_path = gold_year / f"{quote_date.isoformat()}.parquet"
        pl.DataFrame(rows).write_parquet(output_path)
        gold_summary.append(
            {
                "gold_path": str(output_path),
                "quote_date": quote_date.isoformat(),
                "observed_cells": len(rows),
            }
        )
    (manifests_dir / "gold_surface_summary.json").write_bytes(
        orjson.dumps(gold_summary, option=orjson.OPT_INDENT_2)
    )

    ridge_rows = []
    no_change_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            common = {
                "quote_date": date(2021, 1, 4),
                "target_date": date(2021, 1, 5),
                "maturity_index": maturity_index,
                "maturity_days": maturity_days,
                "moneyness_index": moneyness_index,
                "moneyness_point": moneyness_point,
            }
            ridge_rows.append(
                {
                    "model_name": "ridge",
                    **common,
                    "predicted_total_variance": -1.0e-4 if maturity_index == 0 else 0.01,
                }
            )
            no_change_rows.append(
                {
                    "model_name": "no_change",
                    **common,
                    "predicted_total_variance": 0.008,
                }
            )
    pl.DataFrame(ridge_rows).write_parquet(forecast_dir / "ridge.parquet")
    pl.DataFrame(no_change_rows).write_parquet(forecast_dir / "no_change.parquet")

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-05"\n'
        ),
    )
    metrics_config_path = _write_yaml(tmp_path / "metrics.yaml", "positive_floor: 1.0e-8\n")
    stats_config_path = _write_yaml(
        tmp_path / "stats.yaml",
        (
            "loss_metrics:\n"
            '  - "observed_mse_total_variance"\n'
            '  - "observed_qlike_total_variance"\n'
            'benchmark_model: "no_change"\n'
            'dm_alternative: "greater"\n'
            "dm_max_lag: 0\n"
            "spa_block_size: 2\n"
            "spa_bootstrap_reps: 10\n"
            "spa_alpha: 0.10\n"
            "mcs_block_size: 2\n"
            "mcs_bootstrap_reps: 10\n"
            "mcs_alpha: 0.10\n"
            "bootstrap_seed: 7\n"
            'full_grid_weighting: "uniform"\n'
        ),
    )
    hpo_profile_path = _write_yaml(
        tmp_path / "hpo.yaml",
        (
            'profile_name: "test_hpo"\n'
            "n_trials: 1\n"
            "tuning_splits_count: 1\n"
            "seed: 7\n"
        ),
    )
    training_profile_path = _write_yaml(
        tmp_path / "train.yaml",
        (
            'profile_name: "test_train"\n'
            "epochs: 1\n"
            "neural_early_stopping_patience: 1\n"
            "neural_early_stopping_min_delta: 0.0\n"
            "lightgbm_early_stopping_rounds: 1\n"
            "lightgbm_early_stopping_min_delta: 0.0\n"
            "lightgbm_first_metric_only: true\n"
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "07_run_stats.py",
        "stage07_negative_prediction_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        metrics_config_path=metrics_config_path,
        stats_config_path=stats_config_path,
        hpo_profile_config_path=hpo_profile_path,
        training_profile_config_path=training_profile_path,
    )

    stats_dir = manifests_dir / "stats" / workflow_label
    panel = pl.read_parquet(stats_dir / "forecast_realization_panel.parquet")
    daily_loss = pl.read_parquet(stats_dir / "daily_loss_frame.parquet")

    assert panel["predicted_iv"].is_finite().all()
    assert panel["predicted_iv_change"].is_finite().all()
    assert daily_loss["observed_mse_total_variance"].is_finite().all()

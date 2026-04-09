from __future__ import annotations

import os
from datetime import date, timedelta
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


def _write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _business_dates(start: date, count: int) -> list[date]:
    dates: list[date] = []
    current = start
    while len(dates) < count:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _bronze_rows(quote_date: date, spot: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expirations = (quote_date + timedelta(days=7), quote_date + timedelta(days=30))
    for expiration in expirations:
        for option_type in ("C", "P"):
            for moneyness_point in (-0.1, 0.0):
                strike = spot * (2.718281828459045 ** moneyness_point)
                maturity_shift = 0.01 if expiration == expirations[1] else 0.0
                iv = 0.18 + maturity_shift + (0.01 * abs(moneyness_point))
                rows.append(
                    {
                        "quote_date": quote_date,
                        "expiration": expiration,
                        "root": "SPXW",
                        "strike": float(strike),
                        "option_type": option_type,
                        "bid_1545": 1.0,
                        "ask_1545": 1.2,
                        "vega_1545": 1.0,
                        "active_underlying_price_1545": spot,
                        "implied_volatility_1545": iv,
                    }
                )
    return rows


def test_official_default_path_pipeline_runs_through_stage09_on_synthetic_data(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow_label = "hpo_30_trials__train_30_epochs"
    quote_dates = _business_dates(date(2021, 1, 4), count=10)

    bronze_year_dir = tmp_path / "data" / "bronze" / "year=2021"
    bronze_year_dir.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    bronze_summary_rows = []
    for quote_date, spot in zip(quote_dates, range(100, 110), strict=True):
        bronze_path = bronze_year_dir / f"{quote_date.isoformat()}.parquet"
        pl.DataFrame(_bronze_rows(quote_date, float(spot))).write_parquet(bronze_path)
        bronze_summary_rows.append(
            {
                "bronze_path": str(bronze_path),
                "quote_date": quote_date.isoformat(),
            }
        )
    (manifests_dir / "bronze_ingestion_summary.json").write_bytes(
        orjson.dumps(bronze_summary_rows, option=orjson.OPT_INDENT_2)
    )

    _write_text(
        tmp_path / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'target_symbol: "^SPX"\n'
            'calendar_name: "XNYS"\n'
            'timezone: "America/New_York"\n'
            'decision_time: "15:45:00"\n'
            f'sample_start_date: "{quote_dates[0].isoformat()}"\n'
            f'sample_end_date: "{quote_dates[-1].isoformat()}"\n'
            'am_settled_roots: ["SPX"]\n'
        ),
    )
    _write_text(
        tmp_path / "configs" / "data" / "cleaning.yaml",
        (
            'target_symbol: "^SPX"\n'
            'allowed_option_types: ["C", "P"]\n'
            "min_bid: 0.0\n"
            "min_ask: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_mid_price: 0.0\n"
            "max_abs_log_moneyness: 0.5\n"
            "min_tau_years: 0.0001\n"
            "max_tau_years: 2.5\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "data" / "surface.yaml",
        (
            "moneyness_points: [-0.1, 0.0]\n"
            "maturity_days: [7, 30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "data" / "features.yaml",
        (
            "lag_windows: [1]\n"
            "include_daily_change: true\n"
            "include_mask: true\n"
            "include_liquidity: true\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "eval" / "walkforward.yaml",
        (
            "train_size: 4\n"
            "validation_size: 1\n"
            "test_size: 1\n"
            "step_size: 1\n"
            "expanding_train: true\n"
        ),
    )
    _write_text(tmp_path / "configs" / "eval" / "metrics.yaml", "positive_floor: 1.0e-8\n")
    _write_text(
        tmp_path / "configs" / "eval" / "stats_tests.yaml",
        (
            'loss_metric: "observed_mse_total_variance"\n'
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
    _write_text(
        tmp_path / "configs" / "eval" / "report_artifacts.yaml",
        (
            'benchmark_model: "no_change"\n'
            'primary_loss_metric: "observed_mse_total_variance"\n'
            'interpolation_comparison_order: ["moneyness", "maturity"]\n'
            "top_models_per_figure: 3\n"
            "stress_windows:\n"
            f'  - label: "sample_window"\n'
            f'    start_date: "{quote_dates[3].isoformat()}"\n'
            f'    end_date: "{quote_dates[-1].isoformat()}"\n'
        ),
    )
    _write_text(
        tmp_path / "configs" / "eval" / "hedging.yaml",
        (
            "risk_free_rate: 0.0\n"
            "level_notional: 1.0\n"
            "skew_notional: 1.0\n"
            "calendar_notional: 0.5\n"
            "skew_moneyness_abs: 0.1\n"
            "short_maturity_days: 7\n"
            "long_maturity_days: 30\n"
            "hedge_maturity_days: 30\n"
            "hedge_straddle_moneyness: 0.0\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "workflow" / "hpo_30_trials.yaml",
        (
            'profile_name: "hpo_30_trials"\n'
            "n_trials: 1\n"
            "tuning_splits_count: 1\n"
            "seed: 7\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "workflow" / "train_30_epochs.yaml",
        (
            'profile_name: "train_30_epochs"\n'
            "epochs: 1\n"
            "neural_early_stopping_patience: 1\n"
            "neural_early_stopping_min_delta: 0.0\n"
            "lightgbm_early_stopping_rounds: 1\n"
            "lightgbm_early_stopping_min_delta: 0.0\n"
            "lightgbm_first_metric_only: true\n"
        ),
    )
    _write_text(tmp_path / "configs" / "models" / "ridge.yaml", 'model_name: "ridge"\nalpha: 1.0\n')
    _write_text(
        tmp_path / "configs" / "models" / "elasticnet.yaml",
        'model_name: "elasticnet"\nalpha: 0.01\nl1_ratio: 0.1\nmax_iter: 1000\n',
    )
    _write_text(
        tmp_path / "configs" / "models" / "har_factor.yaml",
        'model_name: "har_factor"\nn_factors: 2\nalpha: 1.0\n',
    )
    _write_text(
        tmp_path / "configs" / "models" / "lightgbm.yaml",
        (
            'model_name: "lightgbm"\n'
            'device_type: "cpu"\n'
            "n_estimators: 10\n"
            "learning_rate: 0.1\n"
            "num_leaves: 7\n"
            "max_depth: 3\n"
            "min_child_samples: 1\n"
            "feature_fraction: 1.0\n"
            "lambda_l2: 0.0\n"
            'objective: "regression"\n'
            'metric: "l2"\n'
            "verbosity: -1\n"
            "n_jobs: -1\n"
            "random_state: 7\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "models" / "random_forest.yaml",
        (
            'model_name: "random_forest"\n'
            "n_estimators: 10\n"
            "max_depth: 4\n"
            "min_samples_leaf: 1\n"
            "random_state: 7\n"
            "n_jobs: -1\n"
        ),
    )
    _write_text(
        tmp_path / "configs" / "models" / "neural_surface.yaml",
        (
            'model_name: "neural_surface"\n'
            "hidden_width: 16\n"
            "depth: 1\n"
            "dropout: 0.0\n"
            "learning_rate: 0.001\n"
            "weight_decay: 0.0\n"
            "epochs: 1\n"
            "batch_size: 2\n"
            "seed: 7\n"
            "observed_loss_weight: 1.0\n"
            "imputed_loss_weight: 0.25\n"
            "calendar_penalty_weight: 0.01\n"
            "convexity_penalty_weight: 0.01\n"
            "roughness_penalty_weight: 0.001\n"
            'device: "cpu"\n'
        ),
    )

    stage02 = _load_script_module(repo_root / "scripts" / "02_build_option_panel.py", "stage02")
    stage03 = _load_script_module(repo_root / "scripts" / "03_build_surfaces.py", "stage03")
    stage04 = _load_script_module(repo_root / "scripts" / "04_build_features.py", "stage04")
    stage05 = _load_script_module(repo_root / "scripts" / "05_tune_models.py", "stage05")
    stage06 = _load_script_module(repo_root / "scripts" / "06_run_walkforward.py", "stage06")
    stage07 = _load_script_module(repo_root / "scripts" / "07_run_stats.py", "stage07")
    stage08 = _load_script_module(repo_root / "scripts" / "08_run_hedging_eval.py", "stage08")
    stage09 = _load_script_module(repo_root / "scripts" / "09_make_report_artifacts.py", "stage09")

    original_cwd = Path.cwd()
    os.chdir(tmp_path)
    try:
        stage02.main()
        stage03.main()
        stage04.main()
        for model_name in (
            "ridge",
            "elasticnet",
            "har_factor",
            "lightgbm",
            "random_forest",
            "neural_surface",
        ):
            stage05.main(model_name=model_name)
        stage06.main()
        stage07.main()
        stage08.main()
        stage09.main()
    finally:
        os.chdir(original_cwd)

    report_dir = manifests_dir / "report_artifacts" / workflow_label
    assert (report_dir / "index.md").exists()
    assert (report_dir / "tables" / "ranked_loss_summary.csv").exists()
    assert (report_dir / "details" / "daily_loss_frame.csv").exists()
    assert (report_dir / "details" / "slice_metric_frame.csv").exists()

    ranked_loss_csv = (report_dir / "tables" / "ranked_loss_summary.csv").read_text(
        encoding="utf-8"
    )
    daily_loss_csv = (report_dir / "details" / "daily_loss_frame.csv").read_text(encoding="utf-8")
    slice_metric_csv = (report_dir / "details" / "slice_metric_frame.csv").read_text(
        encoding="utf-8"
    )

    assert "mean_observed_mse_total_variance" in ranked_loss_csv
    assert "observed_mse_total_variance" in daily_loss_csv
    assert "observed_qlike_total_variance" in daily_loss_csv
    assert "mse_total_variance" in slice_metric_csv
    assert "qlike_total_variance" in slice_metric_csv

from __future__ import annotations

import math
import os
from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import polars as pl
import pytest

from ivsurf.calendar import MarketCalendar
from ivsurf.runtime_preflight import run_runtime_preflight


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
    calendar = MarketCalendar()
    dates: list[date] = []
    current = start
    if not calendar.is_session(current):
        current = calendar.next_trading_session(current)
    while len(dates) < count:
        if calendar.is_session(current):
            dates.append(current)
        current = calendar.next_trading_session(current)
    return dates


def _raw_rows(quote_date: date, spot: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expirations = (quote_date + timedelta(days=7), quote_date + timedelta(days=30))
    for expiration in expirations:
        for option_type in ("C", "P"):
            for moneyness_point in (-0.1, 0.0):
                strike = spot * math.exp(moneyness_point)
                maturity_shift = 0.01 if expiration == expirations[1] else 0.0
                iv = 0.18 + maturity_shift + (0.01 * abs(moneyness_point))
                rows.append(
                    {
                        "underlying_symbol": "^SPX",
                        "quote_date": quote_date.isoformat(),
                        "root": "SPXW",
                        "expiration": expiration.isoformat(),
                        "strike": float(strike),
                        "option_type": option_type,
                        "trade_volume": 10,
                        "bid_size_1545": 5,
                        "bid_1545": 1.0,
                        "ask_size_1545": 5,
                        "ask_1545": 1.2,
                        "underlying_bid_1545": 0.0,
                        "underlying_ask_1545": 0.0,
                        "implied_underlying_price_1545": spot,
                        "active_underlying_price_1545": spot,
                        "implied_volatility_1545": iv,
                        "delta_1545": 0.5 if option_type == "C" else -0.5,
                        "gamma_1545": 0.1,
                        "theta_1545": -0.01,
                        "vega_1545": 1.0,
                        "rho_1545": 0.01 if option_type == "C" else -0.01,
                        "open_interest": 100,
                    }
                )
    return rows


def _write_raw_zip(zip_path: Path, rows: list[dict[str, object]]) -> None:
    csv_text = pl.DataFrame(rows).write_csv()
    if csv_text is None:
        raise ValueError(f"Unable to serialize raw rows for {zip_path.name}.")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, mode="w") as archive:
        archive.writestr(f"{zip_path.stem}.csv", csv_text)


def test_synthetic_stage01_to_stage09_pipeline_runs_through_stage09_with_committed_gpu_configs(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    workflow_label = "hpo_30_trials__train_30_epochs"
    quote_dates = _business_dates(date(2021, 1, 4), count=31)
    raw_dir = tmp_path / "raw"
    manifests_dir = tmp_path / "data" / "manifests"

    for quote_date, spot in zip(quote_dates, range(100, 131), strict=True):
        _write_raw_zip(
            raw_dir / f"UnderlyingOptionsEODCalcs_{quote_date.strftime('%Y%m%d')}.zip",
            _raw_rows(quote_date, float(spot)),
        )

    _write_text(
        tmp_path / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{raw_dir.as_posix()}'\n"
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
            "min_valid_bid_exclusive: 0.0\n"
            "min_valid_ask_exclusive: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_valid_mid_price_exclusive: 0.0\n"
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
            "lag_windows: [1, 5, 22]\n"
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
    _write_text(
        tmp_path / "configs" / "eval" / "report_artifacts.yaml",
        (
            'benchmark_model: "no_change"\n'
            "official_loss_metrics:\n"
            '  - "observed_mse_total_variance"\n'
            '  - "observed_qlike_total_variance"\n'
            'primary_loss_metric: "observed_mse_total_variance"\n'
            'interpolation_comparison_order: ["moneyness", "maturity"]\n'
            "top_models_per_figure: 3\n"
            "stress_windows:\n"
            '  - label: "sample_window"\n'
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
            "tuning_splits_count: 3\n"
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
    lightgbm_config_path = repo_root / "configs" / "models" / "lightgbm.yaml"
    neural_config_path = repo_root / "configs" / "models" / "neural_surface.yaml"
    ridge_config_path = repo_root / "configs" / "models" / "ridge.yaml"
    elasticnet_config_path = repo_root / "configs" / "models" / "elasticnet.yaml"
    har_config_path = repo_root / "configs" / "models" / "har_factor.yaml"
    random_forest_config_path = repo_root / "configs" / "models" / "random_forest.yaml"

    try:
        run_runtime_preflight(
            raw_config_path=tmp_path / "configs" / "data" / "raw.yaml",
            lightgbm_config_path=lightgbm_config_path,
            neural_config_path=neural_config_path,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        pytest.skip(f"Official GPU runtime is unavailable in this environment: {exc}")

    stage01 = _load_script_module(repo_root / "scripts" / "01_ingest_cboe.py", "stage01")
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
        stage01.main()
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
            stage05.main(
                model_name=model_name,
                lightgbm_config_path=lightgbm_config_path,
                neural_config_path=neural_config_path,
                hpo_profile_config_path=tmp_path / "configs" / "workflow" / "hpo_30_trials.yaml",
                training_profile_config_path=tmp_path
                / "configs"
                / "workflow"
                / "train_30_epochs.yaml",
            )
        stage06.main(
            ridge_config_path=ridge_config_path,
            elasticnet_config_path=elasticnet_config_path,
            har_config_path=har_config_path,
            lightgbm_config_path=lightgbm_config_path,
            random_forest_config_path=random_forest_config_path,
            neural_config_path=neural_config_path,
            hpo_profile_config_path=tmp_path / "configs" / "workflow" / "hpo_30_trials.yaml",
            training_profile_config_path=tmp_path / "configs" / "workflow" / "train_30_epochs.yaml",
        )
        stage07.main()
        stage08.main()
        stage09.main()
    finally:
        os.chdir(original_cwd)

    report_dir = manifests_dir / "report_artifacts" / workflow_label
    assert (report_dir / "index.md").exists()
    assert (report_dir / "tables" / "ranked_loss_summary.csv").exists()
    assert (
        report_dir
        / "tables"
        / "ranked_loss_summary__observed_qlike_total_variance.csv"
    ).exists()
    assert (report_dir / "tables" / "dm_results__observed_qlike_total_variance.csv").exists()
    assert (report_dir / "details" / "daily_loss_frame.csv").exists()
    assert (report_dir / "details" / "slice_metric_frame.csv").exists()

    bronze_files = sorted((tmp_path / "data" / "bronze").glob("year=*/*.parquet"))
    assert len(bronze_files) == len(quote_dates)

    ranked_loss_csv = (report_dir / "tables" / "ranked_loss_summary.csv").read_text(
        encoding="utf-8"
    )
    qlike_ranked_loss_csv = (
        report_dir / "tables" / "ranked_loss_summary__observed_qlike_total_variance.csv"
    ).read_text(encoding="utf-8")
    daily_loss_csv = (report_dir / "details" / "daily_loss_frame.csv").read_text(encoding="utf-8")
    report_index = (report_dir / "index.md").read_text(encoding="utf-8")

    assert "mean_observed_mse_total_variance" in ranked_loss_csv
    assert "mean_observed_qlike_total_variance" in qlike_ranked_loss_csv
    assert "observed_mse_total_variance" in daily_loss_csv
    assert "observed_qlike_total_variance" in daily_loss_csv
    assert "Official loss metrics" in report_index
    assert "observed_qlike_total_variance" in report_index

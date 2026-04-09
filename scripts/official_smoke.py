from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from zipfile import ZipFile

import orjson
import polars as pl
import typer

from ivsurf.calendar import MarketCalendar
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.runtime_preflight import RuntimePreflightReport, run_runtime_preflight
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES

app = typer.Typer(add_completion=False)


@dataclass(frozen=True, slots=True)
class OfficialSmokeResult:
    """Persisted artifact bundle from the official runtime smoke run."""

    run_root: Path
    report_dir: Path
    summary_path: Path
    runtime_report: RuntimePreflightReport


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        message = f"Unable to load module from {script_path}"
        raise RuntimeError(message)
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
        dates.append(current)
        current = calendar.next_trading_session(current)
    return dates


def _raw_rows(quote_date: date, spot: float) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    expirations = (quote_date + timedelta(days=7), quote_date + timedelta(days=30))
    for expiration in expirations:
        for option_type in ("C", "P"):
            for moneyness_point in (-0.10, 0.00):
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
                        "underlying_bid_1545": spot - 0.1,
                        "underlying_ask_1545": spot + 0.1,
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
        message = f"Unable to serialize raw rows for {zip_path.name}."
        raise ValueError(message)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path, mode="w") as archive:
        archive.writestr(f"{zip_path.stem}.csv", csv_text)


def _write_raw_config(run_root: Path, quote_dates: list[date], raw_dir: Path) -> Path:
    return _write_text(
        run_root / "configs" / "data" / "raw.yaml",
        (
            f"raw_options_dir: '{raw_dir.as_posix()}'\n"
            f"bronze_dir: '{(run_root / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(run_root / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(run_root / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{(run_root / 'data' / 'manifests').as_posix()}'\n"
            'target_symbol: "^SPX"\n'
            'calendar_name: "XNYS"\n'
            'timezone: "America/New_York"\n'
            'decision_time: "15:45:00"\n'
            f'sample_start_date: "{quote_dates[0].isoformat()}"\n'
            f'sample_end_date: "{quote_dates[-1].isoformat()}"\n'
            'am_settled_roots: ["SPX"]\n'
        ),
    )


def _runtime_report_payload(report: RuntimePreflightReport) -> dict[str, object]:
    return {
        "platform_system": report.platform_system,
        "raw_options_dir": str(report.raw_options_dir.resolve()),
        "torch_cuda_available": report.torch_cuda_available,
        "lightgbm_gpu_available": report.lightgbm_gpu_available,
    }


def run_official_smoke(
    *,
    output_root: Path,
    run_name: str | None = None,
) -> OfficialSmokeResult:
    """Run the official Windows/CUDA/GPU smoke pipeline and persist its artifacts."""

    repo_root = _repo_root()
    resolved_output_root = output_root.resolve()
    resolved_output_root.mkdir(parents=True, exist_ok=True)
    effective_run_name = run_name or datetime.now(UTC).strftime("official_smoke_%Y%m%dT%H%M%SZ")
    run_root = resolved_output_root / effective_run_name
    if run_root.exists():
        message = f"Official smoke output directory already exists: {run_root}"
        raise FileExistsError(message)
    run_root.mkdir(parents=True, exist_ok=False)

    quote_dates = _business_dates(date(2021, 1, 4), count=35)
    raw_dir = run_root / "raw"
    for quote_date, spot in zip(quote_dates, range(4000, 4035), strict=True):
        _write_raw_zip(
            raw_dir / f"UnderlyingOptionsEODCalcs_{quote_date.strftime('%Y%m%d')}.zip",
            _raw_rows(quote_date, float(spot)),
        )

    raw_config_path = _write_raw_config(run_root, quote_dates, raw_dir)
    smoke_surface_config_path = repo_root / "configs" / "official_smoke" / "data" / "surface.yaml"
    smoke_feature_config_path = repo_root / "configs" / "official_smoke" / "data" / "features.yaml"
    smoke_walkforward_config_path = (
        repo_root / "configs" / "official_smoke" / "eval" / "walkforward.yaml"
    )
    smoke_stats_config_path = repo_root / "configs" / "official_smoke" / "eval" / "stats_tests.yaml"
    smoke_report_config_path = (
        repo_root / "configs" / "official_smoke" / "eval" / "report_artifacts.yaml"
    )
    smoke_hedging_config_path = repo_root / "configs" / "official_smoke" / "eval" / "hedging.yaml"
    smoke_hpo_profile_path = (
        repo_root / "configs" / "official_smoke" / "workflow" / "hpo_smoke.yaml"
    )
    smoke_train_profile_path = (
        repo_root / "configs" / "official_smoke" / "workflow" / "train_smoke.yaml"
    )
    cleaning_config_path = repo_root / "configs" / "data" / "cleaning.yaml"
    metrics_config_path = repo_root / "configs" / "eval" / "metrics.yaml"
    lightgbm_config_path = repo_root / "configs" / "models" / "lightgbm.yaml"
    neural_config_path = repo_root / "configs" / "models" / "neural_surface.yaml"
    ridge_config_path = repo_root / "configs" / "models" / "ridge.yaml"
    elasticnet_config_path = repo_root / "configs" / "models" / "elasticnet.yaml"
    har_config_path = repo_root / "configs" / "models" / "har_factor.yaml"
    random_forest_config_path = repo_root / "configs" / "models" / "random_forest.yaml"

    runtime_report = run_runtime_preflight(
        raw_config_path=raw_config_path,
        lightgbm_config_path=lightgbm_config_path,
        neural_config_path=neural_config_path,
    )

    stage01 = _load_script_module(repo_root / "scripts" / "01_ingest_cboe.py", "official_smoke_01")
    stage02 = _load_script_module(
        repo_root / "scripts" / "02_build_option_panel.py",
        "official_smoke_02",
    )
    stage03 = _load_script_module(
        repo_root / "scripts" / "03_build_surfaces.py",
        "official_smoke_03",
    )
    stage04 = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "official_smoke_04",
    )
    stage05 = _load_script_module(repo_root / "scripts" / "05_tune_models.py", "official_smoke_05")
    stage06 = _load_script_module(
        repo_root / "scripts" / "06_run_walkforward.py",
        "official_smoke_06",
    )
    stage07 = _load_script_module(repo_root / "scripts" / "07_run_stats.py", "official_smoke_07")
    stage08 = _load_script_module(
        repo_root / "scripts" / "08_run_hedging_eval.py",
        "official_smoke_08",
    )
    stage09 = _load_script_module(
        repo_root / "scripts" / "09_make_report_artifacts.py",
        "official_smoke_09",
    )

    stage01.main(raw_config_path=raw_config_path)
    stage02.main(
        raw_config_path=raw_config_path,
        cleaning_config_path=cleaning_config_path,
    )
    stage03.main(
        raw_config_path=raw_config_path,
        surface_config_path=smoke_surface_config_path,
    )
    stage04.main(
        raw_config_path=raw_config_path,
        surface_config_path=smoke_surface_config_path,
        feature_config_path=smoke_feature_config_path,
        walkforward_config_path=smoke_walkforward_config_path,
    )
    for model_name in TUNABLE_MODEL_NAMES:
        stage05.main(
            model_name=model_name,
            raw_config_path=raw_config_path,
            surface_config_path=smoke_surface_config_path,
            lightgbm_config_path=lightgbm_config_path,
            neural_config_path=neural_config_path,
            hpo_profile_config_path=smoke_hpo_profile_path,
            training_profile_config_path=smoke_train_profile_path,
        )
    stage06.main(
        raw_config_path=raw_config_path,
        surface_config_path=smoke_surface_config_path,
        ridge_config_path=ridge_config_path,
        elasticnet_config_path=elasticnet_config_path,
        har_config_path=har_config_path,
        lightgbm_config_path=lightgbm_config_path,
        random_forest_config_path=random_forest_config_path,
        neural_config_path=neural_config_path,
        hpo_profile_config_path=smoke_hpo_profile_path,
        training_profile_config_path=smoke_train_profile_path,
    )
    stage07.main(
        raw_config_path=raw_config_path,
        metrics_config_path=metrics_config_path,
        stats_config_path=smoke_stats_config_path,
        hpo_profile_config_path=smoke_hpo_profile_path,
        training_profile_config_path=smoke_train_profile_path,
    )
    stage08.main(
        raw_config_path=raw_config_path,
        hedging_config_path=smoke_hedging_config_path,
        hpo_profile_config_path=smoke_hpo_profile_path,
        training_profile_config_path=smoke_train_profile_path,
    )
    stage09.main(
        raw_config_path=raw_config_path,
        surface_config_path=smoke_surface_config_path,
        metrics_config_path=metrics_config_path,
        stats_config_path=smoke_stats_config_path,
        report_config_path=smoke_report_config_path,
        hpo_profile_config_path=smoke_hpo_profile_path,
        training_profile_config_path=smoke_train_profile_path,
    )

    workflow_label = "hpo_smoke__train_smoke"
    report_dir = run_root / "data" / "manifests" / "report_artifacts" / workflow_label
    required_report_paths = (
        report_dir / "index.md",
        report_dir / "tables" / "ranked_loss_summary.csv",
        report_dir / "details" / "daily_loss_frame.csv",
    )
    for required_path in required_report_paths:
        if not required_path.exists():
            message = f"Official smoke did not produce required artifact: {required_path}"
            raise FileNotFoundError(message)

    summary_path = run_root / "official_smoke_summary.json"
    write_bytes_atomic(
        summary_path,
        orjson.dumps(
            {
                "run_root": str(run_root.resolve()),
                "workflow_label": workflow_label,
                "quote_dates": [quote_date.isoformat() for quote_date in quote_dates],
                "runtime_report": _runtime_report_payload(runtime_report),
                "report_dir": str(report_dir.resolve()),
                "key_artifacts": {
                    "report_index": str((report_dir / "index.md").resolve()),
                    "ranked_loss_summary": str(
                        (report_dir / "tables" / "ranked_loss_summary.csv").resolve()
                    ),
                    "daily_loss_frame": str(
                        (report_dir / "details" / "daily_loss_frame.csv").resolve()
                    ),
                },
            },
            option=orjson.OPT_INDENT_2,
        ),
    )
    return OfficialSmokeResult(
        run_root=run_root,
        report_dir=report_dir,
        summary_path=summary_path,
        runtime_report=runtime_report,
    )


@app.command()
def main(
    output_root: Path = Path("data/official_smoke"),
    run_name: str | None = None,
) -> None:
    result = run_official_smoke(output_root=output_root, run_name=run_name)
    typer.echo(f"Official smoke artifacts saved under {result.run_root}")
    typer.echo(f"Official smoke summary saved to {result.summary_path}")
    typer.echo(f"Official smoke report directory: {result.report_dir}")


if __name__ == "__main__":
    app()

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import polars as pl
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.evaluation.alignment import (
    load_actual_surface_frame,
    load_daily_spot_frame,
    load_forecast_frame,
)
from ivsurf.hedging.pnl import evaluate_model_hedging, summarize_hedging_results
from ivsurf.hedging.revaluation import surface_interpolator_from_frame

app = typer.Typer(add_completion=False)


def _actual_surface_lookup(actual_surface_frame: pl.DataFrame) -> dict[object, pl.DataFrame]:
    groups = actual_surface_frame.partition_by("quote_date", as_dict=True)
    return {key[0]: value for key, value in groups.items()}


def _forecast_surface_groups(forecast_frame: pl.DataFrame) -> list[pl.DataFrame]:
    return forecast_frame.partition_by(
        ["model_name", "quote_date", "target_date"],
        maintain_order=True,
    )


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hedging_config_path: Path = Path("configs/eval/hedging.yaml"),
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    hedging_config = load_yaml_config(hedging_config_path)

    actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
    forecast_frame = load_forecast_frame(raw_config.gold_dir / "forecasts")
    spot_frame = load_daily_spot_frame(raw_config.silver_dir)
    spot_lookup = {
        row["quote_date"]: float(row["spot_1545"])
        for row in spot_frame.iter_rows(named=True)
    }
    actual_lookup = _actual_surface_lookup(actual_surface_frame)

    results = []
    for group in _forecast_surface_groups(forecast_frame):
        model_name = str(group["model_name"][0])
        quote_date = group["quote_date"][0]
        target_date = group["target_date"][0]
        if quote_date not in actual_lookup or target_date not in actual_lookup:
            message = (
                f"Missing actual surface for quote_date={quote_date} "
                f"or target_date={target_date}."
            )
            raise ValueError(message)
        if quote_date not in spot_lookup or target_date not in spot_lookup:
            message = (
                f"Missing spot state for quote_date={quote_date} "
                f"or target_date={target_date}."
            )
            raise ValueError(message)

        result = evaluate_model_hedging(
            model_name=model_name,
            quote_date=quote_date,
            target_date=target_date,
            trade_spot=spot_lookup[quote_date],
            target_spot=spot_lookup[target_date],
            actual_surface_t=surface_interpolator_from_frame(
                actual_lookup[quote_date],
                total_variance_column="completed_total_variance",
            ),
            actual_surface_t1=surface_interpolator_from_frame(
                actual_lookup[target_date],
                total_variance_column="completed_total_variance",
            ),
            predicted_surface_t1=surface_interpolator_from_frame(
                group,
                total_variance_column="predicted_total_variance",
            ),
            rate=float(hedging_config["risk_free_rate"]),
            level_notional=float(hedging_config["level_notional"]),
            skew_notional=float(hedging_config["skew_notional"]),
            calendar_notional=float(hedging_config["calendar_notional"]),
            skew_moneyness_abs=float(hedging_config["skew_moneyness_abs"]),
            short_maturity_days=int(hedging_config["short_maturity_days"]),
            long_maturity_days=int(hedging_config["long_maturity_days"]),
            hedge_maturity_days=int(hedging_config["hedge_maturity_days"]),
            hedge_straddle_moneyness=float(hedging_config["hedge_straddle_moneyness"]),
        )
        results.append(asdict(result))

    output_dir = raw_config.manifests_dir / "hedging"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pl.DataFrame(results).sort(["model_name", "quote_date", "target_date"])
    results_frame.write_parquet(output_dir / "hedging_results.parquet", compression="zstd")

    summary_frame = summarize_hedging_results(results_frame)
    summary_frame.write_parquet(output_dir / "hedging_summary.parquet", compression="zstd")
    typer.echo(f"Saved hedging outputs to {output_dir}")


if __name__ == "__main__":
    app()

from __future__ import annotations

from datetime import date
from pathlib import Path

import orjson
import polars as pl
import typer

from ivsurf.calendar import MarketCalendar
from ivsurf.cleaning.derived_fields import add_derived_fields, build_tau_lookup
from ivsurf.cleaning.option_filters import apply_option_quality_flags
from ivsurf.config import CleaningConfig, MarketCalendarConfig, RawDataConfig, load_yaml_config

app = typer.Typer(add_completion=False)


def _silver_path(bronze_path: Path, raw_config: RawDataConfig) -> Path:
    year_partition = bronze_path.parent.name
    output_dir = raw_config.silver_dir / year_partition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / bronze_path.name


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    cleaning_config_path: Path = Path("configs/data/cleaning.yaml"),
    limit: int | None = None,
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    cleaning_config = CleaningConfig.model_validate(load_yaml_config(cleaning_config_path))
    calendar_config = MarketCalendarConfig.model_validate(load_yaml_config(raw_config_path))
    market_calendar = MarketCalendar(
        calendar_name=calendar_config.calendar_name,
        timezone=calendar_config.timezone,
        decision_time=calendar_config.decision_time,
        am_settled_roots=calendar_config.am_settled_roots,
    )

    bronze_files = sorted(raw_config.bronze_dir.glob("year=*/*.parquet"))
    if limit is not None:
        bronze_files = bronze_files[:limit]

    summary_rows: list[dict[str, object]] = []
    for bronze_path in bronze_files:
        frame = pl.read_parquet(bronze_path)
        quote_date = frame["quote_date"][0]
        if not isinstance(quote_date, date):
            message = f"Unexpected quote_date type in {bronze_path}"
            raise TypeError(message)

        if not market_calendar.session_has_decision_time(quote_date):
            tau_lookup = frame.select("root", "expiration").unique().with_columns(
                pl.lit(0.0).alias("tau_years")
            )
            enriched = add_derived_fields(frame=frame, tau_lookup=tau_lookup).with_columns(
                pl.lit("EARLY_CLOSE_SESSION").alias("invalid_reason"),
                pl.lit(False).alias("is_valid_observation"),
            )
        else:
            tau_lookup = build_tau_lookup(frame=frame, calendar_config=calendar_config)
            enriched = add_derived_fields(frame=frame, tau_lookup=tau_lookup)
            enriched = apply_option_quality_flags(frame=enriched, config=cleaning_config)

        output_path = _silver_path(bronze_path=bronze_path, raw_config=raw_config)
        enriched.write_parquet(output_path, compression="zstd", statistics=True)
        summary_rows.append(
            {
                "silver_path": str(output_path),
                "quote_date": quote_date.isoformat(),
                "rows": enriched.height,
                "valid_rows": enriched.filter(pl.col("is_valid_observation")).height,
            }
        )

    summary_path = raw_config.manifests_dir / "silver_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    app()


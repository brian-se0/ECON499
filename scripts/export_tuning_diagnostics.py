"""Export parquet tuning diagnostics to reviewer-friendly CSV and JSON files."""

from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.workflow import tuning_diagnostics_path

app = typer.Typer(add_completion=False)

DIAGNOSTIC_MODELS: tuple[str, ...] = (
    "ridge",
    "elasticnet",
    "har_factor",
    "lightgbm",
    "random_forest",
    "neural_surface",
)


def _summary_payload(frame: pl.DataFrame, model_name: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "model_name": model_name,
        "n_rows": frame.height,
        "columns": frame.columns,
    }
    if frame.is_empty():
        return payload

    median_columns = [
        column
        for column in (
            "selected_metric_value",
            "neural_prediction_target_ratio",
            "neural_prediction_below_1e_6_share",
            "neural_best_epoch",
        )
        if column in frame.columns
    ]
    if median_columns:
        medians = frame.select(
            [pl.col(column).median().alias(column) for column in median_columns]
        ).row(0, named=True)
        payload["medians"] = {
            key: (None if value is None else float(value)) for key, value in medians.items()
        }
    return payload


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hpo_profile_name: str = "hpo_30_trials",
    model_name: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Export one or all tuning diagnostic parquet files under an HPO profile."""

    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    models = (model_name,) if model_name is not None else DIAGNOSTIC_MODELS
    if output_dir is None:
        output_dir = raw_config.manifests_dir / "tuning" / hpo_profile_name / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for diagnostic_model_name in models:
        diagnostics_path = tuning_diagnostics_path(
            raw_config.manifests_dir,
            hpo_profile_name,
            diagnostic_model_name,
        )
        if not diagnostics_path.exists():
            message = f"Missing tuning diagnostics parquet: {diagnostics_path}"
            raise FileNotFoundError(message)

        frame = pl.read_parquet(diagnostics_path)
        csv_path = output_dir / f"{diagnostic_model_name}__diagnostics.csv"
        json_path = output_dir / f"{diagnostic_model_name}__diagnostics.json"
        summary_path = output_dir / f"{diagnostic_model_name}__diagnostics_summary.json"

        frame.write_csv(csv_path)
        frame.write_json(json_path)
        summary_payload = _summary_payload(frame, diagnostic_model_name)
        write_bytes_atomic(
            summary_path,
            orjson.dumps(summary_payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
        )
        summary_rows.append(summary_payload)

    index_path = output_dir / "diagnostics_export_index.json"
    write_bytes_atomic(
        index_path,
        orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
    )
    typer.echo(f"Exported tuning diagnostics to {output_dir}")


if __name__ == "__main__":
    app()

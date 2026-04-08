from __future__ import annotations

from pathlib import Path

import typer

from ivsurf.cleanup import CleanupSelection, build_cleanup_plan, execute_cleanup_plan
from ivsurf.config import RawDataConfig, load_yaml_config

app = typer.Typer(add_completion=False)


def _parse_selection(value: str) -> CleanupSelection:
    valid_selections: tuple[CleanupSelection, ...] = (
        "all",
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
    if value not in valid_selections:
        message = (
            "selection must be one of: all, ingest, silver, surfaces, features, hpo, "
            "train, stats, hedging, report."
        )
        raise typer.BadParameter(message)
    return value


@app.command()
def main(
    selection: str = typer.Argument(
        ...,
        help=(
            "Cleanup selection: all, ingest, silver, surfaces, features, hpo, "
            "train, stats, hedging, or report."
        ),
    ),
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hpo_profile_name: str = "hpo_30_trials",
    training_profile_name: str = "train_30_epochs",
    dry_run: bool = False,
) -> None:
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    typed_selection = _parse_selection(selection)
    plan = build_cleanup_plan(
        raw_config=raw_config,
        selection=typed_selection,
        hpo_profile_name=hpo_profile_name,
        training_profile_name=training_profile_name,
    )
    typer.echo(f"Protected raw options directory: {plan.protected_raw_options_dir}")
    typer.echo(f"Cleanup selection: {plan.selection}")
    typer.echo(f"Invalidated stages: {', '.join(plan.stage_names)}")
    if not plan.paths:
        typer.echo("No derived artifact paths were selected for cleanup.")
        return
    typer.echo("Derived artifact targets:")
    for path in plan.paths:
        typer.echo(f"  - {path}")
    if dry_run:
        typer.echo("Dry run only; no files or directories were deleted.")
        return
    removed_paths = execute_cleanup_plan(plan, raw_config=raw_config)
    typer.echo(f"Removed {len(removed_paths)} derived artifact path(s).")


if __name__ == "__main__":
    app()

from __future__ import annotations

from pathlib import Path

import typer

from ivsurf.runtime_preflight import run_runtime_preflight

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    lightgbm_config_path: Path = Path("configs/models/lightgbm.yaml"),
    neural_config_path: Path = Path("configs/models/neural_surface.yaml"),
) -> None:
    report = run_runtime_preflight(
        raw_config_path=raw_config_path,
        lightgbm_config_path=lightgbm_config_path,
        neural_config_path=neural_config_path,
    )
    typer.echo("Official runtime preflight passed.")
    typer.echo(f"Platform: {report.platform_system}")
    typer.echo(f"Raw options root: {report.raw_options_dir}")
    typer.echo(f"torch.cuda.is_available(): {report.torch_cuda_available}")
    typer.echo(f"LightGBM GPU mode: {report.lightgbm_gpu_available}")


if __name__ == "__main__":
    app()

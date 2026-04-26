"""Write checksum provenance for a profile-backed research run."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CalledProcessError, run

import orjson
import typer

from ivsurf.config import (
    HpoProfileConfig,
    RawDataConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.reproducibility import sha256_bytes, sha256_file
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


@dataclass(frozen=True, slots=True)
class CompactArtifactRecord:
    """Compact artifact checksum record for supplement-scale inventories."""

    path: str
    sha256: str
    size_bytes: int
    modified_at_utc: str


def _iter_files(root: Path, pattern: str = "*") -> list[Path]:
    return sorted(
        path
        for path in root.rglob(pattern)
        if path.is_file() and not path.name.startswith("._")
    )


def _compact_record(path: Path, *, base: Path | None = None) -> CompactArtifactRecord:
    stat = path.stat()
    display_path = str(path if base is None else path.relative_to(base))
    return CompactArtifactRecord(
        path=display_path,
        sha256=sha256_file(path),
        size_bytes=stat.st_size,
        modified_at_utc=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    )


def _records(paths: Iterable[Path], *, base: Path | None = None) -> list[CompactArtifactRecord]:
    return [_compact_record(path, base=base) for path in sorted(paths)]


def _combined_hash(records: list[CompactArtifactRecord]) -> str | None:
    if not records:
        return None
    payload = [asdict(record) for record in records]
    return sha256_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))


def _git_output(args: list[str], repo_root: Path) -> str | None:
    try:
        result = run(args, cwd=repo_root, capture_output=True, check=True, text=True)
    except (CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip()


def _inventory(name: str, paths: Iterable[Path], *, base: Path | None = None) -> dict[str, object]:
    records = _records(paths, base=base)
    return {
        "name": name,
        "n_files": len(records),
        "combined_sha256": _combined_hash(records),
        "files": [asdict(record) for record in records],
    }


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str = "mac_cpu",
    output_path: Path | None = None,
) -> None:
    """Write a checksum supplement for raw, derived, forecast, and report artifacts."""

    repo_root = Path.cwd()
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    workflow_paths = resolve_workflow_run_paths(
        raw_config.gold_dir,
        raw_config.manifests_dir,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
    )

    if output_path is None:
        output_path = Path("provenance") / f"{workflow_paths.run_label}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_config.raw_options_dir.glob(raw_config.raw_file_glob))
    if not raw_files:
        message = (
            f"No raw files matched {raw_config.raw_file_glob!r} "
            f"under {raw_config.raw_options_dir}"
        )
        raise FileNotFoundError(message)

    run_manifest_paths = [
        *(raw_config.manifests_dir / "runs" / "01_ingest_cboe").glob("*_01_ingest_cboe.json"),
        *(raw_config.manifests_dir / "runs" / "02_build_option_panel").glob(
            "*_02_build_option_panel.json"
        ),
        *(raw_config.manifests_dir / "runs" / "03_build_surfaces").glob(
            "*_03_build_surfaces.json"
        ),
        *(raw_config.manifests_dir / "runs" / "04_build_features").glob(
            "*_04_build_features.json"
        ),
        *(raw_config.manifests_dir / "runs" / "05_tune_models").glob("*_05_tune_models.json"),
        *(raw_config.manifests_dir / "runs" / "06_run_walkforward").glob(
            "*_06_run_walkforward.json"
        ),
        *(raw_config.manifests_dir / "runs" / "07_run_stats").glob("*_07_run_stats.json"),
        *(raw_config.manifests_dir / "runs" / "08_run_hedging_eval").glob(
            "*_08_run_hedging_eval.json"
        ),
        *(raw_config.manifests_dir / "runs" / "09_make_report_artifacts").glob(
            "*_09_make_report_artifacts.json"
        ),
    ]

    inventories = [
        _inventory("raw_options", raw_files, base=raw_config.raw_options_dir),
        _inventory("bronze", _iter_files(raw_config.bronze_dir), base=repo_root),
        _inventory("silver", _iter_files(raw_config.silver_dir), base=repo_root),
        _inventory(
            "daily_gold_surfaces",
            [
                path
                for path in _iter_files(raw_config.gold_dir, "*.parquet")
                if "forecasts" not in path.parts and path.name != "daily_features.parquet"
            ],
            base=repo_root,
        ),
        _inventory(
            "daily_features",
            [raw_config.gold_dir / "daily_features.parquet"],
            base=repo_root,
        ),
        _inventory("forecasts", _iter_files(workflow_paths.forecast_dir), base=repo_root),
        _inventory("stats_outputs", _iter_files(workflow_paths.stats_dir), base=repo_root),
        _inventory("hedging_outputs", _iter_files(workflow_paths.hedging_dir), base=repo_root),
        _inventory("report_artifacts", _iter_files(workflow_paths.report_dir), base=repo_root),
        _inventory(
            "tuning_manifests",
            _iter_files(raw_config.manifests_dir / "tuning" / hpo_profile.profile_name),
            base=repo_root,
        ),
        _inventory("run_manifests", run_manifest_paths, base=repo_root),
    ]

    supplement = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "workflow_run_label": workflow_paths.run_label,
        "run_profile_name": run_profile_name,
        "git_commit_hash": _git_output(["git", "rev-parse", "HEAD"], repo_root),
        "git_status_porcelain": _git_output(["git", "status", "--short"], repo_root),
        "raw_options_dir": str(raw_config.raw_options_dir),
        "sample_start_date": raw_config.sample_start_date.isoformat(),
        "sample_end_date": raw_config.sample_end_date.isoformat(),
        "reproduction_commands": [
            "uv sync --extra dev",
            "uv run python scripts/install_mac_lightgbm_no_openmp.py",
            (
                "uv run python scripts/check_runtime.py --raw-config-path "
                "configs/data/raw.mac.yaml --run-profile-name mac_cpu "
                "--neural-config-path configs/models/neural_surface.mac_cpu.yaml "
                "--lightgbm-config-path configs/models/lightgbm.mac_cpu.yaml"
            ),
            (
                "uv run python scripts/05_tune_models.py neural_surface --raw-config-path "
                "configs/data/raw.mac.yaml --neural-config-path "
                "configs/models/neural_surface.mac_cpu.yaml --lightgbm-config-path "
                "configs/models/lightgbm.mac_cpu.yaml"
            ),
            (
                "uv run python scripts/06_run_walkforward.py --raw-config-path "
                "configs/data/raw.mac.yaml --run-profile-name mac_cpu "
                "--neural-config-path configs/models/neural_surface.mac_cpu.yaml "
                "--lightgbm-config-path configs/models/lightgbm.mac_cpu.yaml"
            ),
            (
                "uv run python scripts/07_run_stats.py --raw-config-path "
                "configs/data/raw.mac.yaml --run-profile-name mac_cpu"
            ),
            (
                "uv run python scripts/08_run_hedging_eval.py --raw-config-path "
                "configs/data/raw.mac.yaml --run-profile-name mac_cpu"
            ),
            (
                "uv run python scripts/09_make_report_artifacts.py --raw-config-path "
                "configs/data/raw.mac.yaml --run-profile-name mac_cpu"
            ),
            (
                "uv run python scripts/export_tuning_diagnostics.py --raw-config-path "
                "configs/data/raw.mac.yaml"
            ),
        ],
        "inventories": inventories,
    }
    write_bytes_atomic(
        output_path,
        orjson.dumps(supplement, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
    )
    typer.echo(f"Wrote provenance supplement to {output_path}")


if __name__ == "__main__":
    app()

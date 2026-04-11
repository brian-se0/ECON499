"""Profile-aware workflow path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ivsurf.config import RawDataConfig


def workflow_run_label(hpo_profile_name: str, training_profile_name: str) -> str:
    """Build a deterministic label for one tuned/trained workflow run."""

    return f"{hpo_profile_name}__{training_profile_name}"


def tuning_manifest_path(manifests_dir: Path, hpo_profile_name: str, model_name: str) -> Path:
    """Return the profile-specific tuning manifest path for one model."""

    return manifests_dir / "tuning" / hpo_profile_name / f"{model_name}.json"


def tuning_diagnostics_path(manifests_dir: Path, hpo_profile_name: str, model_name: str) -> Path:
    """Return the profile-specific tuning diagnostics parquet path for one model."""

    return manifests_dir / "tuning" / hpo_profile_name / f"{model_name}__diagnostics.parquet"


@dataclass(frozen=True, slots=True)
class WorkflowRunPaths:
    """Filesystem paths for one HPO/training profile combination."""

    run_label: str
    forecast_dir: Path
    stats_dir: Path
    hedging_dir: Path
    report_dir: Path


def resolve_workflow_run_paths(
    raw_config: RawDataConfig,
    *,
    hpo_profile_name: str,
    training_profile_name: str,
) -> WorkflowRunPaths:
    """Resolve forecast and evaluation directories for one official profile pair."""

    run_label = workflow_run_label(
        hpo_profile_name=hpo_profile_name,
        training_profile_name=training_profile_name,
    )
    return WorkflowRunPaths(
        run_label=run_label,
        forecast_dir=raw_config.gold_dir / "forecasts" / run_label,
        stats_dir=raw_config.manifests_dir / "stats" / run_label,
        hedging_dir=raw_config.manifests_dir / "hedging" / run_label,
        report_dir=raw_config.manifests_dir / "report_artifacts" / run_label,
    )

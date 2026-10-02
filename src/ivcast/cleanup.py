"""Safe stage-aware cleanup planning for derived pipeline artifacts."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ivcast.config import RawDataConfig
from ivcast.workflow import resolve_workflow_run_paths

CleanStageName = Literal[
    "ingest",
    "silver",
    "surfaces",
    "features",
    "hpo",
    "train",
    "stats",
    "hedging",
    "report",
]

CleanupSelection = Literal[
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
]

CLEAN_STAGE_ORDER: tuple[CleanStageName, ...] = (
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

_SCRIPT_NAME_BY_STAGE: dict[CleanStageName, str] = {
    "ingest": "01_ingest_cboe",
    "silver": "02_build_option_panel",
    "surfaces": "03_build_surfaces",
    "features": "04_build_features",
    "hpo": "05_tune_models",
    "train": "06_run_walkforward",
    "stats": "07_run_stats",
    "hedging": "08_run_hedging_eval",
    "report": "09_make_report_artifacts",
}


@dataclass(frozen=True, slots=True)
class CleanupPlan:
    """One explicit derived-artifact cleanup plan."""

    selection: CleanupSelection
    stage_names: tuple[CleanStageName, ...]
    protected_raw_options_dir: Path
    paths: tuple[Path, ...]


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _is_within(path: Path, root: Path) -> bool:
    if path.is_relative_to(root):
        return True
    # POSIX paths remain case-sensitive as strings even on case-insensitive
    # macOS volumes. Compare directory identities as well as resolved spellings.
    try:
        root_stat = root.stat()
    except FileNotFoundError:
        return False
    for ancestor in (path, *path.parents):
        try:
            ancestor_stat = ancestor.stat()
        except FileNotFoundError:
            continue
        if os.path.samestat(ancestor_stat, root_stat):
            return True
    return False


def _paths_overlap(first: Path, second: Path) -> bool:
    return _is_within(first, second) or _is_within(second, first)


def _validate_cleanup_roots(raw_config: RawDataConfig) -> tuple[Path, ...]:
    # resolve() follows ancestor symlinks and Windows junctions, even when the
    # final artifact does not exist. Keep lexical paths only for actual deletion.
    raw_options_dir = raw_config.raw_options_dir.resolve()
    cleanup_roots = tuple(
        path.resolve()
        for path in (
            raw_config.bronze_dir,
            raw_config.silver_dir,
            raw_config.gold_dir,
            raw_config.manifests_dir,
        )
    )
    for cleanup_root in cleanup_roots:
        if _paths_overlap(cleanup_root, raw_options_dir):
            message = (
                "Refusing to build a cleanup plan because a configured derived-artifact root "
                f"overlaps the protected raw options directory: {cleanup_root} vs "
                f"{raw_options_dir}."
            )
            raise ValueError(message)
    return cleanup_roots


def _validate_cleanup_target(path: Path, *, raw_config: RawDataConfig) -> None:
    raw_options_dir = raw_config.raw_options_dir.resolve()
    cleanup_roots = _validate_cleanup_roots(raw_config)
    resolved_path = path.resolve()
    if _paths_overlap(resolved_path, raw_options_dir):
        message = (
            "Refusing to delete a path that overlaps the protected raw options directory: "
            f"{resolved_path}."
        )
        raise ValueError(message)
    if not any(
        _is_within(resolved_path, cleanup_root)
        for cleanup_root in cleanup_roots
    ):
        message = (
            "Refusing to delete a path outside the configured derived-artifact roots: "
            f"{resolved_path}."
        )
        raise ValueError(message)


def _collapse_paths(paths: list[Path]) -> tuple[Path, ...]:
    unique_paths = sorted({_absolute_path(path) for path in paths}, key=lambda value: str(value))
    kept: list[Path] = []
    for path in sorted(unique_paths, key=lambda value: (len(value.parts), str(value))):
        if any(path.is_relative_to(parent) for parent in kept):
            continue
        kept.append(path)
    return tuple(kept)


def cleanup_stage_names(selection: CleanupSelection) -> tuple[CleanStageName, ...]:
    """Return the stage sequence invalidated by a cleanup selection."""

    if selection == "all":
        return CLEAN_STAGE_ORDER
    start_index = CLEAN_STAGE_ORDER.index(selection)
    return CLEAN_STAGE_ORDER[start_index:]


def _resume_dir(manifests_dir: Path, stage_name: CleanStageName) -> Path:
    return manifests_dir / "resume" / _SCRIPT_NAME_BY_STAGE[stage_name]


def _run_manifest_dir(manifests_dir: Path, stage_name: CleanStageName) -> Path:
    return manifests_dir / "runs" / _SCRIPT_NAME_BY_STAGE[stage_name]


def _stage_output_paths(
    stage_name: CleanStageName,
    raw_config: RawDataConfig,
    *,
    hpo_profile_name: str,
    training_profile_name: str,
) -> tuple[Path, ...]:
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile_name,
        training_profile_name=training_profile_name,
    )
    manifests_dir = raw_config.manifests_dir

    match stage_name:
        case "ingest":
            return (
                raw_config.bronze_dir,
                manifests_dir / "bronze_ingestion_summary.json",
                _resume_dir(manifests_dir, "ingest"),
                _run_manifest_dir(manifests_dir, "ingest"),
            )
        case "silver":
            return (
                raw_config.silver_dir,
                manifests_dir / "silver_build_summary.json",
                _resume_dir(manifests_dir, "silver"),
                _run_manifest_dir(manifests_dir, "silver"),
            )
        case "surfaces":
            return (
                raw_config.gold_dir,
                manifests_dir / "gold_surface_summary.json",
                _resume_dir(manifests_dir, "surfaces"),
                _run_manifest_dir(manifests_dir, "surfaces"),
            )
        case "features":
            return (
                raw_config.gold_dir / "daily_features.parquet",
                manifests_dir / "walkforward_splits.json",
                _resume_dir(manifests_dir, "features"),
                _run_manifest_dir(manifests_dir, "features"),
            )
        case "hpo":
            return (
                manifests_dir / "tuning" / hpo_profile_name,
                _resume_dir(manifests_dir, "hpo"),
                _run_manifest_dir(manifests_dir, "hpo"),
            )
        case "train":
            return (
                workflow_paths.forecast_dir,
                _resume_dir(manifests_dir, "train"),
                _run_manifest_dir(manifests_dir, "train"),
            )
        case "stats":
            return (
                workflow_paths.stats_dir,
                _resume_dir(manifests_dir, "stats"),
                _run_manifest_dir(manifests_dir, "stats"),
            )
        case "hedging":
            return (
                workflow_paths.hedging_dir,
                _resume_dir(manifests_dir, "hedging"),
                _run_manifest_dir(manifests_dir, "hedging"),
            )
        case "report":
            return (
                workflow_paths.report_dir,
                _resume_dir(manifests_dir, "report"),
                _run_manifest_dir(manifests_dir, "report"),
            )


def build_cleanup_plan(
    *,
    raw_config: RawDataConfig,
    selection: CleanupSelection,
    hpo_profile_name: str,
    training_profile_name: str,
) -> CleanupPlan:
    """Build an explicit cleanup plan without mutating the filesystem."""

    _validate_cleanup_roots(raw_config)
    stage_names = cleanup_stage_names(selection)
    paths: list[Path] = []
    for stage_name in stage_names:
        paths.extend(
            _stage_output_paths(
                stage_name,
                raw_config,
                hpo_profile_name=hpo_profile_name,
                training_profile_name=training_profile_name,
            )
        )
    planned_paths = _collapse_paths(paths)
    for path in planned_paths:
        _validate_cleanup_target(path, raw_config=raw_config)
    return CleanupPlan(
        selection=selection,
        stage_names=stage_names,
        protected_raw_options_dir=_absolute_path(raw_config.raw_options_dir),
        paths=planned_paths,
    )


def execute_cleanup_plan(plan: CleanupPlan, *, raw_config: RawDataConfig) -> tuple[Path, ...]:
    """Execute a cleanup plan after validating every target path."""

    # Validate the complete plan before the first mutation. Repeat this at
    # execution because links may have changed since planning.
    for path in plan.paths:
        _validate_cleanup_target(path, raw_config=raw_config)
    removed_paths: list[Path] = []
    for path in plan.paths:
        if path.is_symlink():
            path.unlink()
            removed_paths.append(path)
            continue
        if not path.exists():
            continue
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            message = f"Unsupported cleanup target type: {path}"
            raise ValueError(message)
        removed_paths.append(path)
    return tuple(removed_paths)

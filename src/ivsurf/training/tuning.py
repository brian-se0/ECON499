"""Profile-aware tuning manifest persistence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.workflow import tuning_manifest_path

type HyperparameterValue = bool | float | int | str

TUNING_RESULT_SCHEMA_VERSION: Literal["tuning_result_v2"] = "tuning_result_v2"


class TuningResult(BaseModel):
    """Serialized Optuna result for one tuned model/profile."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["tuning_result_v2"]
    model_name: str
    hpo_profile_name: str
    training_profile_name: str
    primary_loss_metric: str
    best_value: float
    best_params: dict[str, HyperparameterValue]
    n_trials_requested: PositiveInt
    n_trials_completed: int = Field(ge=0)
    n_trials_pruned: int = Field(ge=0)
    tuning_splits_count: PositiveInt
    max_hpo_validation_date: date
    first_clean_test_split_id: str
    seed: int
    sampler: str
    pruner: str


@dataclass(frozen=True, slots=True)
class CleanEvaluationPolicy:
    """Shared clean-evaluation policy loaded from tuning manifests."""

    tuning_splits_count: int
    max_hpo_validation_date: date
    first_clean_test_split_id: str


def write_tuning_result(result: TuningResult, output_path: Path) -> None:
    """Persist a tuning manifest with explicit profile metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(
        output_path,
        orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )


def load_tuning_result(path: Path) -> TuningResult:
    """Load a tuning manifest from disk."""

    try:
        return TuningResult.model_validate(orjson.loads(path.read_bytes()))
    except ValidationError as exc:
        message = (
            "Tuning manifest schema is stale or invalid. Delete the saved tuning artifacts "
            f"and rerun stage 05: {path}"
        )
        raise ValueError(message) from exc


def load_required_tuning_results(
    manifests_dir: Path,
    *,
    hpo_profile_name: str,
    model_names: Iterable[str],
) -> dict[str, TuningResult]:
    """Load all required profile-specific tuning manifests or fail fast."""

    loaded: dict[str, TuningResult] = {}
    missing: list[str] = []
    for model_name in model_names:
        path = tuning_manifest_path(manifests_dir, hpo_profile_name, model_name)
        if not path.exists():
            missing.append(f"{model_name} ({path})")
            continue
        loaded[model_name] = load_tuning_result(path)
    if missing:
        message = (
            "Mandatory tuning artifacts are missing. Run stage 05 first for the selected "
            f"HPO profile {hpo_profile_name!r}. Missing: {missing}"
        )
        raise FileNotFoundError(message)
    return loaded


def require_consistent_clean_evaluation_policy(
    tuning_results: Iterable[TuningResult],
) -> CleanEvaluationPolicy:
    """Fail fast unless all tuning manifests agree on the evaluation boundary."""

    results = tuple(tuning_results)
    if not results:
        message = "At least one tuning result is required to define the evaluation boundary."
        raise ValueError(message)

    tuning_counts = {result.tuning_splits_count for result in results}
    if len(tuning_counts) != 1:
        message = (
            "Tuning manifests disagree on tuning_splits_count: "
            f"{sorted(tuning_counts)}."
        )
        raise ValueError(message)

    max_validation_dates = {result.max_hpo_validation_date for result in results}
    if len(max_validation_dates) != 1:
        message = (
            "Tuning manifests disagree on max_hpo_validation_date: "
            f"{sorted(value.isoformat() for value in max_validation_dates)}."
        )
        raise ValueError(message)

    first_clean_split_ids = {result.first_clean_test_split_id for result in results}
    if len(first_clean_split_ids) != 1:
        message = (
            "Tuning manifests disagree on first_clean_test_split_id: "
            f"{sorted(first_clean_split_ids)}."
        )
        raise ValueError(message)

    return CleanEvaluationPolicy(
        tuning_splits_count=results[0].tuning_splits_count,
        max_hpo_validation_date=results[0].max_hpo_validation_date,
        first_clean_test_split_id=results[0].first_clean_test_split_id,
    )


def require_matching_primary_loss_metric(
    tuning_results: Iterable[TuningResult],
    *,
    expected_primary_loss_metric: str,
) -> str:
    """Fail fast unless tuning manifests match the configured primary loss metric."""

    results = tuple(tuning_results)
    if not results:
        message = "At least one tuning result is required to validate the loss metric."
        raise ValueError(message)

    primary_loss_metrics = {result.primary_loss_metric for result in results}
    if len(primary_loss_metrics) != 1:
        message = (
            "Tuning manifests disagree on primary_loss_metric: "
            f"{sorted(primary_loss_metrics)}."
        )
        raise ValueError(message)

    manifest_metric = results[0].primary_loss_metric
    if manifest_metric != expected_primary_loss_metric:
        message = (
            "Tuning manifests were optimized against a different primary_loss_metric: "
            f"{manifest_metric!r} != {expected_primary_loss_metric!r}. "
            "Delete stale stage-05 artifacts and rerun stage 05."
        )
        raise ValueError(message)
    return manifest_metric

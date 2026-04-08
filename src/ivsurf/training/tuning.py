"""Profile-aware tuning manifest persistence."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import orjson
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.workflow import tuning_manifest_path

type HyperparameterValue = bool | float | int | str


class TuningResult(BaseModel):
    """Serialized Optuna result for one tuned model/profile."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    hpo_profile_name: str
    training_profile_name: str
    best_value: float
    best_params: dict[str, HyperparameterValue]
    n_trials_requested: PositiveInt
    n_trials_completed: int = Field(ge=0)
    n_trials_pruned: int = Field(ge=0)
    tuning_splits_count: PositiveInt
    seed: int
    sampler: str
    pruner: str


def write_tuning_result(result: TuningResult, output_path: Path) -> None:
    """Persist a tuning manifest with explicit profile metadata."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(
        output_path,
        orjson.dumps(result.model_dump(mode="json"), option=orjson.OPT_INDENT_2)
    )


def load_tuning_result(path: Path) -> TuningResult:
    """Load a tuning manifest from disk."""

    return TuningResult.model_validate(orjson.loads(path.read_bytes()))


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

"""Split manifest serialization and hashing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import orjson

from ivsurf.io.atomic import write_bytes_atomic

SPLIT_MANIFEST_SCHEMA_VERSION = "walkforward_split_manifest_v2"
SPLIT_MANIFEST_GENERATOR = "ivsurf.splits.walkforward.build_walkforward_splits"


@dataclass(frozen=True, slots=True)
class WalkforwardSplit:
    """Explicit walk-forward split."""

    split_id: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class WalkforwardSplitManifest:
    """Versioned walk-forward split manifest."""

    schema_version: str
    generator: str
    walkforward_config: dict[str, object]
    sample_window: dict[str, str]
    date_universe_hash: str
    feature_dataset_hash: str
    splits: list[WalkforwardSplit]


def date_universe_hash(dates: Sequence[object]) -> str:
    """Hash the ordered quote-date universe used to generate walk-forward splits."""

    normalized_dates = [str(value) for value in dates]
    if not normalized_dates:
        message = "Split manifest date universe cannot be empty."
        raise ValueError(message)
    raw = orjson.dumps(normalized_dates)
    return sha256(raw).hexdigest()


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        message = f"Split manifest field {field_name} must be a non-empty string."
        raise ValueError(message)
    return value


def _normalize_metadata_mapping(
    values: Mapping[str, object],
    *,
    field_name: str,
) -> dict[str, object]:
    if not values:
        message = f"Split manifest field {field_name} cannot be empty."
        raise ValueError(message)
    return dict(values)


def _manifest_payload(
    *,
    splits: list[WalkforwardSplit],
    walkforward_config: Mapping[str, object],
    sample_window: Mapping[str, str],
    date_universe: Sequence[object],
    feature_dataset_hash: str,
) -> dict[str, object]:
    _require_non_empty_string(feature_dataset_hash, field_name="feature_dataset_hash")
    if not splits:
        message = "Split manifest must contain at least one split."
        raise ValueError(message)
    return {
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "generator": SPLIT_MANIFEST_GENERATOR,
        "walkforward_config": _normalize_metadata_mapping(
            walkforward_config,
            field_name="walkforward_config",
        ),
        "sample_window": _normalize_metadata_mapping(sample_window, field_name="sample_window"),
        "date_universe_hash": date_universe_hash(date_universe),
        "feature_dataset_hash": feature_dataset_hash,
        "splits": [asdict(split) for split in splits],
    }


def serialize_splits(
    splits: list[WalkforwardSplit],
    output_path: Path,
    *,
    walkforward_config: Mapping[str, object],
    sample_window: Mapping[str, str],
    date_universe: Sequence[object],
    feature_dataset_hash: str,
) -> str:
    """Write split manifest and return its SHA256 hash."""

    payload = _manifest_payload(
        splits=splits,
        walkforward_config=walkforward_config,
        sample_window=sample_window,
        date_universe=date_universe,
        feature_dataset_hash=feature_dataset_hash,
    )
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(output_path, raw)
    return sha256(raw).hexdigest()


def load_split_manifest(path: Path) -> WalkforwardSplitManifest:
    """Load and validate a versioned split manifest from disk."""

    payload = orjson.loads(path.read_bytes())
    if isinstance(payload, list):
        message = (
            "Legacy bare-list split manifests are not supported. "
            "Regenerate splits with Stage 04."
        )
        raise ValueError(message)
    if not isinstance(payload, dict):
        message = "Split manifest must be a versioned JSON object."
        raise ValueError(message)
    schema_version = _require_non_empty_string(
        payload.get("schema_version"),
        field_name="schema_version",
    )
    if schema_version != SPLIT_MANIFEST_SCHEMA_VERSION:
        message = (
            "Unsupported split manifest schema_version: "
            f"{schema_version!r}; expected {SPLIT_MANIFEST_SCHEMA_VERSION!r}."
        )
        raise ValueError(message)
    generator = _require_non_empty_string(payload.get("generator"), field_name="generator")
    if generator != SPLIT_MANIFEST_GENERATOR:
        message = (
            "Unsupported split manifest generator: "
            f"{generator!r}; expected {SPLIT_MANIFEST_GENERATOR!r}."
        )
        raise ValueError(message)
    splits_payload = payload.get("splits")
    if not isinstance(splits_payload, list) or not splits_payload:
        message = "Split manifest must contain a non-empty splits list."
        raise ValueError(message)
    splits = [_split_from_payload(item, index=index) for index, item in enumerate(splits_payload)]
    return WalkforwardSplitManifest(
        schema_version=schema_version,
        generator=generator,
        walkforward_config=_dict_payload(payload.get("walkforward_config"), "walkforward_config"),
        sample_window=_string_dict_payload(payload.get("sample_window"), "sample_window"),
        date_universe_hash=_require_non_empty_string(
            payload.get("date_universe_hash"),
            field_name="date_universe_hash",
        ),
        feature_dataset_hash=_require_non_empty_string(
            payload.get("feature_dataset_hash"),
            field_name="feature_dataset_hash",
        ),
        splits=splits,
    )


def _string_tuple_payload(value: object, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not value:
        message = f"Split manifest field {field_name} must be a non-empty string array."
        raise ValueError(message)
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            message = f"Split manifest field {field_name} must contain non-empty strings."
            raise ValueError(message)
        normalized.append(item)
    return tuple(normalized)


def _split_from_payload(value: object, *, index: int) -> WalkforwardSplit:
    if not isinstance(value, dict):
        message = f"Split manifest split at index {index} must be an object."
        raise ValueError(message)
    return WalkforwardSplit(
        split_id=_require_non_empty_string(value.get("split_id"), field_name="split_id"),
        train_dates=_string_tuple_payload(value.get("train_dates"), "train_dates"),
        validation_dates=_string_tuple_payload(
            value.get("validation_dates"),
            "validation_dates",
        ),
        test_dates=_string_tuple_payload(value.get("test_dates"), "test_dates"),
    )


def _dict_payload(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        message = f"Split manifest field {field_name} must be a non-empty object."
        raise ValueError(message)
    return dict(value)


def _string_dict_payload(value: object, field_name: str) -> dict[str, str]:
    payload = _dict_payload(value, field_name)
    normalized: dict[str, str] = {}
    for key, item in payload.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item:
            message = f"Split manifest field {field_name} must contain string keys and values."
            raise ValueError(message)
        normalized[key] = item
    return normalized


def require_split_manifest_matches_artifacts(
    manifest: WalkforwardSplitManifest,
    *,
    date_universe: Sequence[object],
    feature_dataset_hash: str,
) -> None:
    """Fail fast if a split manifest was built for a different feature artifact."""

    actual_date_hash = date_universe_hash(date_universe)
    if manifest.date_universe_hash != actual_date_hash:
        message = (
            "Split manifest date_universe_hash does not match daily_features quote dates: "
            f"{manifest.date_universe_hash!r} != {actual_date_hash!r}."
        )
        raise ValueError(message)
    if manifest.feature_dataset_hash != feature_dataset_hash:
        message = (
            "Split manifest feature_dataset_hash does not match daily_features.parquet: "
            f"{manifest.feature_dataset_hash!r} != {feature_dataset_hash!r}."
        )
        raise ValueError(message)
    require_split_manifest_semantics(manifest, date_universe=date_universe)


def _date_universe_as_dates(date_universe: Sequence[object]) -> list[date]:
    normalized: list[date] = []
    for index, value in enumerate(date_universe):
        if isinstance(value, date):
            normalized.append(value)
            continue
        if isinstance(value, str):
            try:
                parsed = date.fromisoformat(value)
            except ValueError as exc:
                message = f"date_universe[{index}] is not an ISO calendar date: {value!r}."
                raise ValueError(message) from exc
            if parsed.isoformat() != value:
                message = f"date_universe[{index}] is not canonical ISO date text: {value!r}."
                raise ValueError(message)
            normalized.append(parsed)
            continue
        message = (
            f"date_universe[{index}] must be a datetime.date or canonical ISO date string; "
            f"got {type(value).__name__}."
        )
        raise ValueError(message)
    if not normalized:
        message = "Split manifest date universe cannot be empty."
        raise ValueError(message)
    if len(set(normalized)) != len(normalized):
        message = "Split manifest date universe must be duplicate-free."
        raise ValueError(message)
    if any(left >= right for left, right in pairwise(normalized)):
        message = "Split manifest date universe must be strictly increasing."
        raise ValueError(message)
    return normalized


def _split_dates(split: WalkforwardSplit) -> tuple[str, ...]:
    return (*split.train_dates, *split.validation_dates, *split.test_dates)


def _require_split_date_arrays_valid(
    split: WalkforwardSplit,
    *,
    split_index: int,
    date_position_by_iso: Mapping[str, int],
) -> None:
    arrays = {
        "train_dates": split.train_dates,
        "validation_dates": split.validation_dates,
        "test_dates": split.test_dates,
    }
    positions_by_array: dict[str, list[int]] = {}
    for field_name, values in arrays.items():
        if len(set(values)) != len(values):
            message = f"Split {split.split_id} {field_name} contains duplicate dates."
            raise ValueError(message)
        positions: list[int] = []
        previous_position = -1
        for value in values:
            position = date_position_by_iso.get(value)
            if position is None:
                message = (
                    f"Split {split.split_id} {field_name} contains date {value!r} "
                    "outside the feature date universe."
                )
                raise ValueError(message)
            if position <= previous_position:
                message = (
                    f"Split {split.split_id} {field_name} must be strictly increasing "
                    "in feature-date order."
                )
                raise ValueError(message)
            positions.append(position)
            previous_position = position
        positions_by_array[field_name] = positions

    all_dates = _split_dates(split)
    if len(set(all_dates)) != len(all_dates):
        message = f"Split {split.split_id} train/validation/test arrays must be disjoint."
        raise ValueError(message)
    train_positions = positions_by_array["train_dates"]
    validation_positions = positions_by_array["validation_dates"]
    test_positions = positions_by_array["test_dates"]
    if max(train_positions) >= min(validation_positions):
        message = f"Split {split.split_id} train dates must precede validation dates."
        raise ValueError(message)
    if max(validation_positions) >= min(test_positions):
        message = f"Split {split.split_id} validation dates must precede test dates."
        raise ValueError(message)
    expected_split_id = f"split_{split_index:04d}"
    if split.split_id != expected_split_id:
        message = (
            f"Split manifest split_id sequence is invalid at index {split_index}: "
            f"{split.split_id!r} != {expected_split_id!r}."
        )
        raise ValueError(message)


def require_split_manifest_semantics(
    manifest: WalkforwardSplitManifest,
    *,
    date_universe: Sequence[object],
) -> None:
    """Require loaded splits to match deterministic walk-forward semantics."""

    from ivsurf.config import WalkforwardConfig
    from ivsurf.splits.walkforward import build_walkforward_splits

    ordered_dates = _date_universe_as_dates(date_universe)
    date_position_by_iso = {value.isoformat(): index for index, value in enumerate(ordered_dates)}
    config = WalkforwardConfig.model_validate(manifest.walkforward_config)
    for split_index, split in enumerate(manifest.splits):
        _require_split_date_arrays_valid(
            split,
            split_index=split_index,
            date_position_by_iso=date_position_by_iso,
        )
    expected_splits = build_walkforward_splits(ordered_dates, config)
    if manifest.splits == expected_splits:
        return
    first_mismatch = next(
        (
            index
            for index, (actual, expected) in enumerate(
                zip(manifest.splits, expected_splits, strict=False)
            )
            if actual != expected
        ),
        None,
    )
    message = (
        "Split manifest does not match deterministic walk-forward output for the recorded "
        f"walkforward_config; actual_splits={len(manifest.splits)}, "
        f"expected_splits={len(expected_splits)}, first_mismatch={first_mismatch!r}."
    )
    raise ValueError(message)


def load_splits(path: Path) -> list[WalkforwardSplit]:
    """Load a versioned split manifest from disk."""

    return load_split_manifest(path).splits

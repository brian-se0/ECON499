from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

import orjson
import pytest

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.manifests import (
    SPLIT_MANIFEST_GENERATOR,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    WalkforwardSplit,
    date_universe_hash,
    load_split_manifest,
    require_split_manifest_matches_artifacts,
    require_split_manifest_semantics,
    serialize_splits,
)
from ivsurf.splits.walkforward import build_walkforward_splits


def _split() -> WalkforwardSplit:
    return WalkforwardSplit(
        split_id="split_0000",
        train_dates=("2021-01-04", "2021-01-05"),
        validation_dates=("2021-01-06",),
        test_dates=("2021-01-07",),
    )


def _date_universe() -> list[date]:
    return [
        date(2021, 1, 4),
        date(2021, 1, 5),
        date(2021, 1, 6),
        date(2021, 1, 7),
        date(2021, 1, 8),
        date(2021, 1, 11),
    ]


def _walkforward_config(*, expanding_train: bool = True) -> WalkforwardConfig:
    return WalkforwardConfig(
        train_size=2,
        validation_size=1,
        test_size=1,
        step_size=1,
        expanding_train=expanding_train,
    )


def _write_semantic_manifest(
    tmp_path: Path,
    *,
    payload_update: dict[str, object] | None = None,
) -> Path:
    manifest_path = tmp_path / "walkforward_splits.json"
    dates = _date_universe()
    config = _walkforward_config()
    serialize_splits(
        build_walkforward_splits(dates, config),
        manifest_path,
        walkforward_config=config.model_dump(mode="json"),
        sample_window={
            "sample_start_date": dates[0].isoformat(),
            "sample_end_date": dates[-1].isoformat(),
        },
        date_universe=dates,
        feature_dataset_hash="daily-feature-sha",
    )
    if payload_update is not None:
        payload = orjson.loads(manifest_path.read_bytes())
        payload.update(payload_update)
        manifest_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    return manifest_path


def test_serialize_splits_writes_versioned_artifact_bound_manifest(tmp_path: Path) -> None:
    dates = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)]
    walkforward_config = WalkforwardConfig(
        train_size=2,
        validation_size=1,
        test_size=1,
        step_size=1,
        expanding_train=True,
    )
    manifest_path = tmp_path / "walkforward_splits.json"

    split_hash = serialize_splits(
        [_split()],
        manifest_path,
        walkforward_config=walkforward_config.model_dump(mode="json"),
        sample_window={
            "sample_start_date": "2021-01-04",
            "sample_end_date": "2021-01-07",
        },
        date_universe=dates,
        feature_dataset_hash="daily-feature-sha",
    )

    payload = orjson.loads(manifest_path.read_bytes())
    manifest = load_split_manifest(manifest_path)

    assert split_hash
    assert payload["schema_version"] == SPLIT_MANIFEST_SCHEMA_VERSION
    assert payload["generator"] == SPLIT_MANIFEST_GENERATOR
    assert payload["date_universe_hash"] == date_universe_hash(dates)
    assert payload["feature_dataset_hash"] == "daily-feature-sha"
    assert manifest.splits == [_split()]


def test_load_split_manifest_rejects_legacy_bare_list(tmp_path: Path) -> None:
    manifest_path = tmp_path / "walkforward_splits.json"
    manifest_path.write_bytes(orjson.dumps([asdict(_split())]))

    with pytest.raises(ValueError, match="Legacy bare-list split manifests"):
        load_split_manifest(manifest_path)


def test_load_split_manifest_rejects_old_schema_version(tmp_path: Path) -> None:
    manifest_path = _write_semantic_manifest(tmp_path)
    payload = orjson.loads(manifest_path.read_bytes())
    payload["schema_version"] = "walkforward_split_manifest_v1"
    manifest_path.write_bytes(orjson.dumps(payload))

    with pytest.raises(ValueError, match="Unsupported split manifest schema_version"):
        load_split_manifest(manifest_path)


def test_require_split_manifest_matches_artifacts_rejects_stale_inputs(tmp_path: Path) -> None:
    dates = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)]
    manifest_path = tmp_path / "walkforward_splits.json"
    serialize_splits(
        [_split()],
        manifest_path,
        walkforward_config={
            "train_size": 2,
            "validation_size": 1,
            "test_size": 1,
            "step_size": 1,
            "expanding_train": True,
        },
        sample_window={
            "sample_start_date": "2021-01-04",
            "sample_end_date": "2021-01-07",
        },
        date_universe=dates,
        feature_dataset_hash="daily-feature-sha",
    )
    manifest = load_split_manifest(manifest_path)

    require_split_manifest_matches_artifacts(
        manifest,
        date_universe=dates,
        feature_dataset_hash="daily-feature-sha",
    )
    with pytest.raises(ValueError, match="date_universe_hash"):
        require_split_manifest_matches_artifacts(
            manifest,
            date_universe=dates[:-1],
            feature_dataset_hash="daily-feature-sha",
        )
    with pytest.raises(ValueError, match="feature_dataset_hash"):
        require_split_manifest_matches_artifacts(
            manifest,
            date_universe=dates,
            feature_dataset_hash="different-daily-feature-sha",
        )


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("overlap_train_test", "disjoint"),
        ("reversed_train", "strictly increasing"),
        ("duplicate_validation", "duplicate"),
        ("unknown_date", "outside the feature date universe"),
        ("wrong_split_id", "split_id sequence"),
        ("wrong_step_size", "deterministic walk-forward output"),
        ("wrong_expanding_geometry", "deterministic walk-forward output"),
    ],
)
def test_split_manifest_semantic_validation_rejects_tampering(
    tmp_path: Path,
    tamper: str,
    match: str,
) -> None:
    manifest_path = _write_semantic_manifest(tmp_path)
    payload = orjson.loads(manifest_path.read_bytes())
    if tamper == "overlap_train_test":
        payload["splits"][0]["test_dates"] = [payload["splits"][0]["train_dates"][0]]
    elif tamper == "reversed_train":
        payload["splits"][1]["train_dates"] = list(reversed(payload["splits"][1]["train_dates"]))
    elif tamper == "duplicate_validation":
        payload["splits"][0]["validation_dates"] = [
            payload["splits"][0]["validation_dates"][0],
            payload["splits"][0]["validation_dates"][0],
        ]
    elif tamper == "unknown_date":
        payload["splits"][0]["test_dates"] = ["2021-02-01"]
    elif tamper == "wrong_split_id":
        payload["splits"][0]["split_id"] = "split_9999"
    elif tamper == "wrong_step_size":
        payload["walkforward_config"]["step_size"] = 2
    elif tamper == "wrong_expanding_geometry":
        payload["walkforward_config"]["expanding_train"] = False
    else:
        raise AssertionError(tamper)
    manifest_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))

    manifest = load_split_manifest(manifest_path)
    with pytest.raises(ValueError, match=match):
        require_split_manifest_matches_artifacts(
            manifest,
            date_universe=_date_universe(),
            feature_dataset_hash="daily-feature-sha",
        )


def test_split_manifest_semantic_validation_accepts_deterministic_manifest(
    tmp_path: Path,
) -> None:
    manifest = load_split_manifest(_write_semantic_manifest(tmp_path))

    require_split_manifest_semantics(manifest, date_universe=_date_universe())

from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl
import pytest

from ivsurf.reproducibility import sha256_file
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
)
from ivsurf.surfaces.interpolation import COMPLETED_SURFACE_SCHEMA_VERSION


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _silver_rows(
    quote_date: date,
    *,
    total_variance: float,
    is_valid: bool,
    include_outside_grid_row: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    decision_timestamp = f"{quote_date.isoformat()}T15:45:00-05:00"
    for multiplier, log_moneyness in ((1.0, -0.1), (1.2, 0.1)):
        cell_total_variance = total_variance * multiplier
        rows.append(
            {
                "quote_date": quote_date,
                "effective_decision_timestamp": decision_timestamp,
                "tau_years": 30.0 / 365.0,
                "log_moneyness": log_moneyness,
                "total_variance": cell_total_variance,
                "implied_volatility_1545": float(
                    (cell_total_variance / (30.0 / 365.0)) ** 0.5
                ),
                "vega_1545": 1.0,
                "spread_1545": 0.01,
                "is_valid_observation": is_valid,
            }
        )
    if include_outside_grid_row:
        rows.append(
            {
                "quote_date": quote_date,
                "effective_decision_timestamp": decision_timestamp,
                "tau_years": 30.0 / 365.0,
                "log_moneyness": 0.3,
                "total_variance": total_variance * 100.0,
                "implied_volatility_1545": float(
                    ((total_variance * 100.0) / (30.0 / 365.0)) ** 0.5
                ),
                "vega_1545": 1.0,
                "spread_1545": 0.01,
                "is_valid_observation": is_valid,
            }
        )
    return rows


def test_stage03_and_stage04_preserve_skipped_day_gaps_and_gold_input_provenance(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    silver_year = tmp_path / "data" / "silver" / "year=2021"
    silver_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    dates_and_validity = [
        (date(2021, 1, 4), True),
        (date(2021, 1, 5), True),
        (date(2021, 1, 6), False),
        (date(2021, 1, 7), True),
        (date(2021, 1, 8), True),
    ]
    silver_summary: list[dict[str, object]] = []
    for offset, (quote_date_value, is_valid) in enumerate(dates_and_validity, start=1):
        silver_path = silver_year / f"{quote_date_value.isoformat()}.parquet"
        rows = _silver_rows(
            quote_date_value,
            total_variance=0.001 * offset,
            is_valid=is_valid,
            include_outside_grid_row=quote_date_value == date(2021, 1, 4),
        )
        pl.DataFrame(rows).write_parquet(silver_path)
        silver_summary.append(
            {
                "silver_path": str(silver_path),
                "quote_date": quote_date_value.isoformat(),
                "status": "built",
                "rows": len(rows),
                "valid_rows": sum(1 for row in rows if row["is_valid_observation"]),
            }
        )
    (manifests_dir / "silver_build_summary.json").write_bytes(
        orjson.dumps(silver_summary, option=orjson.OPT_INDENT_2)
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-08"\n'
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [-0.2, -0.1, 0.0, 0.1, 0.2]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    feature_config_path = _write_yaml(
        tmp_path / "features.yaml",
        (
            "lag_windows: [1]\n"
            "include_daily_change: false\n"
            "include_mask: true\n"
            "include_liquidity: false\n"
        ),
    )
    walkforward_config_path = _write_yaml(
        tmp_path / "walkforward.yaml",
        (
            "train_size: 1\n"
            "validation_size: 1\n"
            "test_size: 1\n"
            "step_size: 1\n"
            "expanding_train: true\n"
        ),
    )

    stage03 = _load_script_module(
        repo_root / "scripts" / "03_build_surfaces.py",
        "stage03_target_gap_alignment",
    )
    stage04 = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "stage04_target_gap_alignment",
    )

    stage03.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
    )
    grid = SurfaceGrid(maturity_days=(30,), moneyness_points=(-0.2, -0.1, 0.0, 0.1, 0.2))
    surface_config_hash = sha256_file(surface_config_path)

    skipped_payload = orjson.loads((manifests_dir / "gold_surface_skipped_dates.json").read_bytes())
    assert skipped_payload == [
        {
            "quote_date": "2021-01-06",
            "reason": "NO_VALID_ROWS_AFTER_CLEANING",
            "silver_path": str(silver_year / "2021-01-06.parquet"),
        }
    ]

    stage03_run_manifest = sorted((manifests_dir / "runs" / "03_build_surfaces").glob("*.json"))[-1]
    stage03_manifest_payload = orjson.loads(stage03_run_manifest.read_bytes())
    stage03_output_paths = {
        artifact["path"] for artifact in stage03_manifest_payload["output_artifacts"]
    }
    assert (
        str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage03_output_paths
    )
    gold_surface_summary = orjson.loads((manifests_dir / "gold_surface_summary.json").read_bytes())
    first_built_row = next(row for row in gold_surface_summary if row["quote_date"] == "2021-01-04")
    assert first_built_row["rows_inside_grid_domain"] == 2
    assert first_built_row["rows_outside_grid_domain"] == 1
    assert first_built_row["grid_domain_reason_counts"] == {
        "VALID_GRID_DOMAIN": 2,
        "OUTSIDE_MONEYNESS_GRID_DOMAIN": 1,
    }
    assert first_built_row["completion_status_counts"] == {
        "extrapolated_boundary_fill": 2,
        "interpolated": 1,
        "observed": 2,
    }
    assert first_built_row["effective_decision_timestamp"] == "2021-01-04T15:45:00-05:00"
    assert first_built_row["surface_grid_hash"] == grid.grid_hash
    assert first_built_row["maturity_coordinate"] == MATURITY_COORDINATE
    assert first_built_row["moneyness_coordinate"] == MONEYNESS_COORDINATE
    assert first_built_row["target_surface_version"] == COMPLETED_SURFACE_SCHEMA_VERSION
    assert first_built_row["surface_config_hash"] == surface_config_hash
    first_gold_surface = pl.read_parquet(
        tmp_path / "data" / "gold" / "year=2021" / "2021-01-04.parquet"
    )
    assert first_gold_surface["completion_status"].to_list() == [
        "extrapolated_boundary_fill",
        "observed",
        "interpolated",
        "observed",
        "extrapolated_boundary_fill",
    ]
    assert first_gold_surface["surface_grid_schema_version"].unique().to_list() == [
        SURFACE_GRID_SCHEMA_VERSION
    ]
    assert first_gold_surface["surface_grid_hash"].unique().to_list() == [grid.grid_hash]
    assert first_gold_surface["target_surface_version"].unique().to_list() == [
        COMPLETED_SURFACE_SCHEMA_VERSION
    ]
    assert first_gold_surface["surface_config_hash"].unique().to_list() == [surface_config_hash]

    stage04.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        feature_config_path=feature_config_path,
        walkforward_config_path=walkforward_config_path,
    )

    feature_frame = pl.read_parquet(tmp_path / "data" / "gold" / "daily_features.parquet")
    target_gap_row = feature_frame.filter(pl.col("quote_date") == date(2021, 1, 5))
    assert target_gap_row.height == 1
    assert target_gap_row["target_date"].to_list() == [date(2021, 1, 7)]
    assert target_gap_row["target_gap_sessions"].to_list() == [1]
    assert target_gap_row["effective_decision_timestamp"].to_list() == [
        "2021-01-05T15:45:00-05:00"
    ]
    assert target_gap_row["target_effective_decision_timestamp"].to_list() == [
        "2021-01-07T15:45:00-05:00"
    ]
    assert target_gap_row["surface_config_hash"].to_list() == [surface_config_hash]
    availability_manifest = orjson.loads(
        (manifests_dir / "feature_availability_manifest.json").read_bytes()
    )
    assert {row["column_name"] for row in availability_manifest} == set(feature_frame.columns)

    gold_files = sorted((tmp_path / "data" / "gold").glob("year=*/*.parquet"))
    stage04_run_manifest = sorted((manifests_dir / "runs" / "04_build_features").glob("*.json"))[-1]
    stage04_manifest_payload = orjson.loads(stage04_run_manifest.read_bytes())
    stage04_input_paths = {
        artifact["path"] for artifact in stage04_manifest_payload["input_artifacts"]
    }
    assert str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage04_input_paths
    for gold_file in gold_files:
        assert str(gold_file.resolve()) in stage04_input_paths


def test_stage03_rejects_nonfinite_valid_silver_surface_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    silver_year = tmp_path / "data" / "silver" / "year=2021"
    silver_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    silver_path = silver_year / "2021-01-04.parquet"
    pl.DataFrame(
        [
            {
                "quote_date": date(2021, 1, 4),
                "effective_decision_timestamp": "2021-01-04T15:45:00-05:00",
                "tau_years": 30.0 / 365.0,
                "log_moneyness": 0.0,
                "total_variance": float("inf"),
                "implied_volatility_1545": 0.2,
                "vega_1545": 1.0,
                "spread_1545": 0.01,
                "is_valid_observation": True,
            }
        ]
    ).write_parquet(silver_path)
    (manifests_dir / "silver_build_summary.json").write_bytes(orjson.dumps([]))

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-04"\n'
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [0.0]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )

    stage03 = _load_script_module(
        repo_root / "scripts" / "03_build_surfaces.py",
        "stage03_nonfinite_valid_silver_script",
    )
    with pytest.raises(ValueError, match="null or non-finite surface inputs"):
        stage03.main(
            raw_config_path=raw_config_path,
            surface_config_path=surface_config_path,
        )

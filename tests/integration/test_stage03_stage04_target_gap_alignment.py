from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl


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
) -> list[dict[str, object]]:
    return [
        {
            "quote_date": quote_date,
            "tau_years": 30.0 / 365.0,
            "log_moneyness": 0.0,
            "total_variance": total_variance,
            "implied_volatility_1545": float((total_variance / (30.0 / 365.0)) ** 0.5),
            "vega_1545": 1.0,
            "spread_1545": 0.01,
            "is_valid_observation": is_valid,
        }
    ]


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
        pl.DataFrame(
            _silver_rows(
                quote_date_value,
                total_variance=0.001 * offset,
                is_valid=is_valid,
            )
        ).write_parquet(silver_path)
        silver_summary.append(
            {
                "silver_path": str(silver_path),
                "quote_date": quote_date_value.isoformat(),
                "status": "built",
                "rows": 1,
                "valid_rows": 1 if is_valid else 0,
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
            "moneyness_points: [0.0]\n"
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

    gold_files = sorted((tmp_path / "data" / "gold").glob("year=*/*.parquet"))
    stage04_run_manifest = sorted((manifests_dir / "runs" / "04_build_features").glob("*.json"))[-1]
    stage04_manifest_payload = orjson.loads(stage04_run_manifest.read_bytes())
    stage04_input_paths = {
        artifact["path"] for artifact in stage04_manifest_payload["input_artifacts"]
    }
    assert str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage04_input_paths
    for gold_file in gold_files:
        assert str(gold_file.resolve()) in stage04_input_paths

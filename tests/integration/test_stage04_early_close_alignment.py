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


def _gold_rows(quote_date: date, total_variance: float) -> list[dict[str, object]]:
    maturity_days = 30
    maturity_years = maturity_days / 365.0
    return [
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "maturity_days": maturity_days,
            "moneyness_index": 0,
            "moneyness_point": 0.0,
            "observed_total_variance": total_variance,
            "observed_iv": float((total_variance / maturity_years) ** 0.5),
            "completed_total_variance": total_variance,
            "completed_iv": float((total_variance / maturity_years) ** 0.5),
            "observed_mask": True,
            "vega_sum": 1.0,
            "observation_count": 1,
            "weighted_spread_1545": 0.01,
        }
    ]


def test_stage04_aligns_pre_early_close_features_to_the_early_close_target(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_year = tmp_path / "data" / "gold" / "year=2019"
    gold_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    quote_dates = [
        date(2019, 11, 26),
        date(2019, 11, 27),
        date(2019, 11, 29),
        date(2019, 12, 2),
        date(2019, 12, 3),
    ]
    gold_summary: list[dict[str, object]] = []
    for offset, quote_date_value in enumerate(quote_dates, start=1):
        output_path = gold_year / f"{quote_date_value.isoformat()}.parquet"
        pl.DataFrame(_gold_rows(quote_date_value, total_variance=0.001 * offset)).write_parquet(
            output_path
        )
        gold_summary.append(
            {
                "gold_path": str(output_path),
                "quote_date": quote_date_value.isoformat(),
                "observed_cells": 1,
            }
        )
    (manifests_dir / "gold_surface_summary.json").write_bytes(
        orjson.dumps(gold_summary, option=orjson.OPT_INDENT_2)
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'decision_time: "15:45:00"\n'
            "decision_snapshot_minutes_before_close: 15\n"
            'sample_start_date: "2019-11-26"\n'
            'sample_end_date: "2019-12-03"\n'
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

    script_module = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "stage04_early_close_alignment_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        feature_config_path=feature_config_path,
        walkforward_config_path=walkforward_config_path,
    )

    feature_frame = pl.read_parquet(tmp_path / "data" / "gold" / "daily_features.parquet")
    aligned_row = feature_frame.filter(pl.col("quote_date") == date(2019, 11, 27))

    assert aligned_row.height == 1
    assert aligned_row["target_date"].to_list() == [date(2019, 11, 29)]

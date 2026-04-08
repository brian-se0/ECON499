from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from ivsurf.exceptions import DataValidationError


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


def test_stage04_rejects_out_of_window_gold_artifacts(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_dir = tmp_path / "data" / "gold" / "year=2021"
    gold_dir.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    in_window_frame = pl.DataFrame({"quote_date": [date(2021, 4, 8)]})
    out_of_window_frame = pl.DataFrame({"quote_date": [date(2021, 4, 12)]})
    in_window_frame.write_parquet(gold_dir / "2021-04-08.parquet")
    out_of_window_frame.write_parquet(gold_dir / "2021-04-12.parquet")

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-04-08"\n'
            'sample_end_date: "2021-04-09"\n'
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
            "include_daily_change: true\n"
            "include_mask: true\n"
            "include_liquidity: true\n"
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
        "sample_window_stage04_script",
    )
    with pytest.raises(DataValidationError, match="outside the configured sample window"):
        script_module.main(
            raw_config_path=raw_config_path,
            surface_config_path=surface_config_path,
            feature_config_path=feature_config_path,
            walkforward_config_path=walkforward_config_path,
        )

    assert not (tmp_path / "data" / "gold" / "daily_features.parquet").exists()

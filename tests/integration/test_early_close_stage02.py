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


def test_stage02_keeps_early_close_sessions_in_the_supervised_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bronze_dir = tmp_path / "data" / "bronze" / "year=2019"
    bronze_dir.mkdir(parents=True)
    silver_dir = tmp_path / "data" / "silver"
    gold_dir = tmp_path / "data" / "gold"
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "bronze_ingestion_summary.json").write_bytes(orjson.dumps([]))

    bronze_frame = pl.DataFrame(
        [
            {
                "quote_date": date(2019, 11, 29),
                "expiration": date(2019, 12, 20),
                "root": "SPXW",
                "strike": 3100.0,
                "option_type": "C",
                "bid_1545": 10.0,
                "ask_1545": 10.5,
                "vega_1545": 1.0,
                "active_underlying_price_1545": 3140.0,
                "implied_volatility_1545": 0.2,
            }
        ]
    )
    bronze_path = bronze_dir / "2019-11-29.parquet"
    bronze_frame.write_parquet(bronze_path)

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{silver_dir.as_posix()}'\n"
            f"gold_dir: '{gold_dir.as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2019-11-29"\n'
            'sample_end_date: "2019-11-29"\n'
        ),
    )
    cleaning_config_path = _write_yaml(
        tmp_path / "cleaning.yaml",
        (
            'target_symbol: "^SPX"\n'
            'allowed_option_types: ["C", "P"]\n'
            "min_bid: 0.0\n"
            "min_ask: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_mid_price: 0.0\n"
            "max_abs_log_moneyness: 0.5\n"
            "min_tau_years: 0.0001\n"
            "max_tau_years: 2.5\n"
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "02_build_option_panel.py",
        "stage02_early_close_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        cleaning_config_path=cleaning_config_path,
    )

    silver_output = pl.read_parquet(silver_dir / "year=2019" / "2019-11-29.parquet")
    summary = orjson.loads((manifests_dir / "silver_build_summary.json").read_bytes())

    assert silver_output["is_valid_observation"].to_list() == [True]
    assert silver_output["invalid_reason"].to_list() == [None]
    assert float(silver_output["tau_years"][0]) > 0.0
    assert summary[0]["status"] == "built"
    assert summary[0]["valid_rows"] == 1

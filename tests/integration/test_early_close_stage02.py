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


def _bronze_row(
    *,
    bid: float | None = 10.0,
    ask: float | None = 10.5,
    strike: float | None = 3100.0,
    implied_volatility: float | None = 0.2,
) -> dict[str, object]:
    return {
        "quote_date": date(2019, 11, 29),
        "expiration": date(2019, 12, 20),
        "root": "SPXW",
        "strike": strike,
        "option_type": "C",
        "bid_1545": bid,
        "ask_1545": ask,
        "vega_1545": 1.0,
        "active_underlying_price_1545": 3140.0,
        "implied_volatility_1545": implied_volatility,
    }


def test_stage02_keeps_early_close_sessions_in_the_supervised_path(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    bronze_dir = tmp_path / "data" / "bronze" / "year=2019"
    bronze_dir.mkdir(parents=True)
    silver_dir = tmp_path / "data" / "silver"
    gold_dir = tmp_path / "data" / "gold"
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)
    (manifests_dir / "bronze_ingestion_summary.json").write_bytes(orjson.dumps([]))

    bronze_frame = pl.DataFrame([_bronze_row()])
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
            "min_valid_bid_exclusive: 0.0\n"
            "min_valid_ask_exclusive: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_valid_mid_price_exclusive: 0.0\n"
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
    assert silver_output["effective_decision_timestamp"].to_list() == [
        "2019-11-29T12:45:00-05:00"
    ]
    assert float(silver_output["tau_years"][0]) > 0.0
    assert summary[0]["status"] == "built"
    assert summary[0]["valid_rows"] == 1
    assert summary[0]["invalid_reason_counts"] == {"VALID": 1}
    assert summary[0]["effective_decision_timestamp"] == "2019-11-29T12:45:00-05:00"
    run_manifest_path = sorted(
        (manifests_dir / "runs" / "02_build_option_panel").glob("*.json")
    )[-1]
    run_manifest_payload = orjson.loads(run_manifest_path.read_bytes())
    output_artifact_paths = {
        artifact["path"] for artifact in run_manifest_payload["output_artifacts"]
    }
    assert str((silver_dir / "year=2019" / "2019-11-29.parquet").resolve()) in output_artifact_paths


def test_stage02_summary_counts_invalid_reasons_by_date(tmp_path: Path) -> None:
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
            _bronze_row(),
            _bronze_row(bid=None),
            _bronze_row(bid=1.0, ask=0.0),
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
            "min_valid_bid_exclusive: 0.0\n"
            "min_valid_ask_exclusive: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_valid_mid_price_exclusive: 0.0\n"
            "max_abs_log_moneyness: 0.5\n"
            "min_tau_years: 0.0001\n"
            "max_tau_years: 2.5\n"
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "02_build_option_panel.py",
        "stage02_invalid_reason_summary_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        cleaning_config_path=cleaning_config_path,
    )

    silver_output = pl.read_parquet(silver_dir / "year=2019" / "2019-11-29.parquet")
    summary = orjson.loads((manifests_dir / "silver_build_summary.json").read_bytes())

    expected_counts = {
        "MISSING_BID_1545": 1,
        "NON_POSITIVE_ASK": 1,
        "VALID": 1,
    }
    assert silver_output["invalid_reason"].to_list() == [
        None,
        "MISSING_BID_1545",
        "NON_POSITIVE_ASK",
    ]
    assert summary[0]["rows"] == 3
    assert summary[0]["valid_rows"] == 1
    assert summary[0]["invalid_reason_counts"] == expected_counts
    assert list(summary[0]["invalid_reason_counts"]) == sorted(expected_counts)
    assert sum(summary[0]["invalid_reason_counts"].values()) == summary[0]["rows"]
    assert summary[0]["invalid_reason_counts"]["VALID"] == summary[0]["valid_rows"]


def test_stage02_accounts_for_nonfinite_and_nonpositive_inputs_as_cleaning_invalids(
    tmp_path: Path,
) -> None:
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
            _bronze_row(),
            _bronze_row(bid=float("nan")),
            _bronze_row(ask=float("inf")),
            _bronze_row(strike=0.0),
            _bronze_row(implied_volatility=float("-inf")),
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
            "min_valid_bid_exclusive: 0.0\n"
            "min_valid_ask_exclusive: 0.0\n"
            "require_ask_ge_bid: true\n"
            "require_positive_iv: true\n"
            "require_positive_vega: true\n"
            "require_positive_underlying_price: true\n"
            "min_valid_mid_price_exclusive: 0.0\n"
            "max_abs_log_moneyness: 0.5\n"
            "min_tau_years: 0.0001\n"
            "max_tau_years: 2.5\n"
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "02_build_option_panel.py",
        "stage02_nonfinite_reason_summary_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        cleaning_config_path=cleaning_config_path,
    )

    silver_output = pl.read_parquet(silver_dir / "year=2019" / "2019-11-29.parquet")
    summary = orjson.loads((manifests_dir / "silver_build_summary.json").read_bytes())

    assert silver_output["invalid_reason"].to_list() == [
        None,
        "NONFINITE_BID_1545",
        "NONFINITE_ASK_1545",
        "NON_POSITIVE_STRIKE",
        "NONFINITE_IV_1545",
    ]
    assert silver_output["is_valid_observation"].to_list() == [True, False, False, False, False]
    assert summary[0]["valid_rows"] == 1
    assert summary[0]["invalid_reason_counts"] == {
        "NONFINITE_ASK_1545": 1,
        "NONFINITE_BID_1545": 1,
        "NONFINITE_IV_1545": 1,
        "NON_POSITIVE_STRIKE": 1,
        "VALID": 1,
    }

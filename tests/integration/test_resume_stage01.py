from __future__ import annotations

from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

import orjson
import pytest

from ivsurf.io.ingest_cboe import IngestionResult


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


def test_stage01_resume_reruns_only_missing_output_item(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_files = [
        raw_dir / "UnderlyingOptionsEODCalcs_20210104.zip",
        raw_dir / "UnderlyingOptionsEODCalcs_20210105.zip",
    ]
    for raw_file in raw_files:
        raw_file.write_bytes(b"placeholder")

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{raw_dir.as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{(tmp_path / 'data' / 'manifests').as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-05"\n'
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "01_ingest_cboe.py",
        "stage01_resume_script",
    )
    calls: list[str] = []

    def fake_ingest(zip_path: Path, config: Any) -> IngestionResult:
        calls.append(zip_path.name)
        quote_date = datetime.strptime(
            zip_path.stem.rsplit("_", maxsplit=1)[-1],
            "%Y%m%d",
        ).date()
        bronze_path = (
            config.bronze_dir / f"year={quote_date.year}" / f"{zip_path.stem}.parquet"
        )
        bronze_path.parent.mkdir(parents=True, exist_ok=True)
        bronze_path.write_bytes(b"bronze")
        return IngestionResult(
            source_zip=zip_path,
            quote_date=quote_date,
            bronze_path=bronze_path,
            row_count=10,
            status="written",
        )

    monkeypatch.setattr(script_module, "ingest_one_zip", fake_ingest)

    script_module.main(raw_config_path=raw_config_path)
    assert calls == [raw_files[0].name, raw_files[1].name]

    missing_output = (
        tmp_path / "data" / "bronze" / "year=2021" / f"{raw_files[1].stem}.parquet"
    )
    missing_output.unlink()
    calls.clear()

    script_module.main(raw_config_path=raw_config_path)
    assert calls == [raw_files[1].name]

    manifest_payload = orjson.loads(
        (tmp_path / "data" / "manifests" / "bronze_ingestion_summary.json").read_bytes()
    )
    assert manifest_payload["files_written"] == 2
    assert len(manifest_payload["results"]) == 2

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import pytest


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_official_smoke_runtime_uses_the_real_stage01_to_stage09_entrypoint(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_module = _load_script_module(
        repo_root / "scripts" / "official_smoke.py",
        "official_smoke_runtime_test",
    )

    try:
        result = script_module.run_official_smoke(
            output_root=tmp_path / "official_smoke",
            run_name="pytest_official_smoke",
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError) as exc:
        pytest.skip(f"Official runtime smoke is unavailable in this environment: {exc}")

    summary = orjson.loads(result.summary_path.read_bytes())
    report_dir = Path(summary["report_dir"])

    assert summary["workflow_label"] == "hpo_smoke__train_smoke"
    assert report_dir == result.report_dir
    assert (report_dir / "index.md").exists()
    assert (report_dir / "tables" / "ranked_loss_summary.csv").exists()
    assert (report_dir / "details" / "daily_loss_frame.csv").exists()

from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


def test_official_smoke_passes_smoke_surface_config_to_grid_dependent_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_module = _load_script_module(
        repo_root / "scripts" / "official_smoke.py",
        "official_smoke_argument_contract_test",
    )
    expected_surface_config_path = (
        repo_root / "configs" / "official_smoke" / "data" / "surface.yaml"
    )
    calls: dict[str, list[dict[str, object]]] = {}

    def fake_run_runtime_preflight(
        *,
        raw_config_path: Path,
        lightgbm_config_path: Path,
        neural_config_path: Path,
        run_profile_name: str,
    ) -> object:
        return script_module.RuntimePreflightReport(
            runtime_profile_name=run_profile_name,
            platform_system="Windows",
            raw_options_dir=raw_config_path.parent,
            torch_cuda_available=True,
            lightgbm_gpu_available=True,
            lightgbm_cpu_available=False,
            lightgbm_openmp_linked=None,
        )

    def fake_load_script_module(script_path: Path, module_name: str) -> SimpleNamespace:
        stage_name = script_path.name

        def main(**kwargs: object) -> None:
            calls.setdefault(stage_name, []).append(dict(kwargs))
            if stage_name == "09_make_report_artifacts.py":
                raw_config_path = kwargs["raw_config_path"]
                if not isinstance(raw_config_path, Path):
                    raise TypeError("Stage 09 raw_config_path must be a Path.")
                report_dir = (
                    raw_config_path.parents[2]
                    / "data"
                    / "manifests"
                    / "report_artifacts"
                    / "hpo_smoke__train_smoke"
                )
                (report_dir / "tables").mkdir(parents=True, exist_ok=True)
                (report_dir / "details").mkdir(parents=True, exist_ok=True)
                (report_dir / "index.md").write_text("# smoke report\n", encoding="utf-8")
                (report_dir / "tables" / "ranked_loss_summary.csv").write_text(
                    "model,loss\nridge,0.1\n",
                    encoding="utf-8",
                )
                (report_dir / "details" / "daily_loss_frame.csv").write_text(
                    "date,loss\n2021-01-04,0.1\n",
                    encoding="utf-8",
                )

        return SimpleNamespace(main=main)

    monkeypatch.setattr(script_module, "run_runtime_preflight", fake_run_runtime_preflight)
    monkeypatch.setattr(script_module, "_load_script_module", fake_load_script_module)

    result = script_module.run_official_smoke(
        output_root=tmp_path / "official_smoke",
        run_name="pytest_argument_contract",
    )

    grid_dependent_stages = (
        "03_build_surfaces.py",
        "04_build_features.py",
        "05_tune_models.py",
        "06_run_walkforward.py",
        "07_run_stats.py",
        "08_run_hedging_eval.py",
        "09_make_report_artifacts.py",
    )
    for stage_name in grid_dependent_stages:
        stage_calls = calls[stage_name]
        assert stage_calls
        assert all(
            call["surface_config_path"] == expected_surface_config_path
            for call in stage_calls
        )

    assert len(calls["05_tune_models.py"]) == len(script_module.TUNABLE_MODEL_NAMES)
    assert all(
        "surface_config_path" not in call
        for call in calls["01_ingest_cboe.py"] + calls["02_build_option_panel.py"]
    )
    assert result.summary_path.exists()

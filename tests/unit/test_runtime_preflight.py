from __future__ import annotations

from pathlib import Path

import pytest

from ivsurf import runtime_preflight


def _write_config_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    raw_config_path = tmp_path / "raw.yaml"
    raw_config_path.write_text(
        "\n".join(
            (
                f'raw_options_dir: "{raw_root}"',
                'bronze_dir: "data/bronze"',
                'silver_dir: "data/silver"',
                'gold_dir: "data/gold"',
                'manifests_dir: "data/manifests"',
                'raw_file_glob: "UnderlyingOptionsEODCalcs_*.zip"',
                'target_symbol: "^SPX"',
                'calendar_name: "XNYS"',
                'timezone: "America/New_York"',
                'decision_time: "15:45:00"',
                "decision_snapshot_minutes_before_close: 15",
                'sample_start_date: "2004-01-02"',
                'sample_end_date: "2021-04-09"',
                "am_settled_roots:",
                '  - "SPX"',
                "",
            )
        ),
        encoding="utf-8",
    )
    lightgbm_config_path = tmp_path / "lightgbm.yaml"
    lightgbm_config_path.write_text(
        "\n".join(
            (
                'model_name: "lightgbm"',
                "n_factors: 2",
                'device_type: "cpu"',
                "n_estimators: 5",
                "learning_rate: 0.1",
                "num_leaves: 7",
                "max_depth: 3",
                "min_child_samples: 1",
                "feature_fraction: 1.0",
                "lambda_l2: 0.0",
                'objective: "regression"',
                'metric: "l2"',
                "verbosity: -1",
                "n_jobs: 1",
                "random_state: 7",
                "",
            )
        ),
        encoding="utf-8",
    )
    neural_config_path = tmp_path / "neural.yaml"
    neural_config_path.write_text(
        "\n".join(
            (
                'model_name: "neural_surface"',
                "hidden_width: 8",
                "depth: 1",
                "dropout: 0.0",
                "learning_rate: 0.001",
                "weight_decay: 0.0",
                "epochs: 2",
                "batch_size: 2",
                "seed: 7",
                "observed_loss_weight: 1.0",
                "imputed_loss_weight: 0.25",
                "calendar_penalty_weight: 0.0",
                "convexity_penalty_weight: 0.0",
                "roughness_penalty_weight: 0.0",
                "output_total_variance_floor: 1.0e-8",
                'device: "cpu"',
                "",
            )
        ),
        encoding="utf-8",
    )
    return raw_config_path, lightgbm_config_path, neural_config_path


def test_mac_cpu_preflight_rejects_openmp_linked_lightgbm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_config_path, lightgbm_config_path, neural_config_path = _write_config_files(tmp_path)
    fake_library = tmp_path / "lib_lightgbm.dylib"
    fake_library.touch()

    monkeypatch.setattr(runtime_preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_preflight, "_lightgbm_library_path", lambda: fake_library)
    monkeypatch.setattr(runtime_preflight, "_lightgbm_links_openmp", lambda _: True)

    with pytest.raises(RuntimeError, match="without OpenMP"):
        runtime_preflight.run_runtime_preflight(
            raw_config_path=raw_config_path,
            lightgbm_config_path=lightgbm_config_path,
            neural_config_path=neural_config_path,
            run_profile_name="mac_cpu",
        )


def test_mac_cpu_preflight_accepts_no_openmp_lightgbm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_config_path, lightgbm_config_path, neural_config_path = _write_config_files(tmp_path)
    fake_library = tmp_path / "lib_lightgbm.dylib"
    fake_library.touch()

    monkeypatch.setattr(runtime_preflight.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(runtime_preflight, "_lightgbm_library_path", lambda: fake_library)
    monkeypatch.setattr(runtime_preflight, "_lightgbm_links_openmp", lambda _: False)
    monkeypatch.setattr(
        runtime_preflight,
        "_check_lightgbm_runtime",
        lambda *_args, **_kwargs: None,
    )

    report = runtime_preflight.run_runtime_preflight(
        raw_config_path=raw_config_path,
        lightgbm_config_path=lightgbm_config_path,
        neural_config_path=neural_config_path,
        run_profile_name="mac_cpu",
    )

    assert report.runtime_profile_name == "mac_cpu"
    assert report.lightgbm_cpu_available is True
    assert report.lightgbm_gpu_available is False
    assert report.lightgbm_openmp_linked is False


def test_mac_cpu_preflight_requires_single_threaded_lightgbm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_config_path, lightgbm_config_path, neural_config_path = _write_config_files(tmp_path)
    lightgbm_config_path.write_text(
        lightgbm_config_path.read_text(encoding="utf-8").replace("n_jobs: 1", "n_jobs: 2"),
        encoding="utf-8",
    )

    monkeypatch.setattr(runtime_preflight.platform, "system", lambda: "Darwin")

    with pytest.raises(RuntimeError, match="n_jobs=1"):
        runtime_preflight.run_runtime_preflight(
            raw_config_path=raw_config_path,
            lightgbm_config_path=lightgbm_config_path,
            neural_config_path=neural_config_path,
            run_profile_name="mac_cpu",
        )

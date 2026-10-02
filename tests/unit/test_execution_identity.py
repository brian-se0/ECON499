from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import orjson
import pytest

from ivcast import reproducibility, runtime_preflight
from ivcast.resume import build_resume_context_hash


def test_native_identity_supports_pinned_windows_wheel_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_dir = tmp_path / "lightgbm"
    library = package_dir / "bin/lib_lightgbm.dll"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"native library in pinned Windows wheel layout")
    monkeypatch.setattr(
        runtime_preflight, "find_spec",
        lambda _: SimpleNamespace(origin=str(package_dir / "__init__.py")),
    )
    identity = reproducibility.collect_native_library_hashes()
    assert identity == {
        "lightgbm": {
            "path": str(library.resolve()), "sha256": reproducibility.sha256_file(library),
        },
    }


@pytest.fixture
def identity_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, str]]:
    source = tmp_path / "src/ivcast/transform.py"
    source.parent.mkdir(parents=True)
    source.write_text("RESULT = 1\n")
    native_library = tmp_path / "lib_lightgbm.dylib"
    native_library.write_bytes(b"stock build of the same package version")
    versions = {"lightgbm": "4.6.0", "numpy": "2.3.0"}
    monkeypatch.setattr(runtime_preflight, "_lightgbm_library_path", lambda: native_library)
    monkeypatch.setattr(reproducibility, "collect_package_versions", lambda: dict(versions))
    monkeypatch.setattr(reproducibility, "collect_hardware_metadata", dict)
    monkeypatch.setattr(reproducibility, "git_commit_hash", lambda _: "same-commit")
    return source, native_library, versions


def test_same_version_native_rebuild_invalidates_execution_identity(
    tmp_path: Path, identity_inputs: tuple[Path, Path, dict[str, str]],
) -> None:
    _, library, _ = identity_inputs
    before = reproducibility.collect_execution_identity(tmp_path)
    previous_context = build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], execution_identity=before,
    )
    library.write_bytes(b"no-OpenMP rebuild of the same package version")
    after = reproducibility.collect_execution_identity(tmp_path)
    assert before["source_sha256"] == after["source_sha256"]
    assert before["runtime"]["package_versions"] == after["runtime"]["package_versions"]
    assert before["runtime"]["native_libraries"] != after["runtime"]["native_libraries"]
    assert build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], execution_identity=after,
    ) != previous_context


@pytest.mark.parametrize("change", ["source", "package_version", "native_build"])
def test_manifest_rejects_changes_after_stage_start_without_publishing(
    tmp_path: Path, identity_inputs: tuple[Path, Path, dict[str, str]], change: str,
) -> None:
    source, library, versions = identity_inputs
    started_at = datetime.now(UTC)
    startup_identity = reproducibility.collect_execution_identity(tmp_path)
    initial_context = build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], execution_identity=startup_identity,
    )
    output = tmp_path / "result.parquet"
    output.write_bytes(b"historical result is retained")
    manifest_arguments: dict[str, Any] = {
        "repo_root": tmp_path, "manifests_dir": tmp_path / "manifests",
        "script_name": "audit_stage", "started_at": started_at,
        "config_paths": [], "input_artifact_paths": [], "output_artifact_paths": [output],
        "execution_identity": startup_identity,
    }
    original_manifest = reproducibility.write_run_manifest(**manifest_arguments)
    original_manifest_bytes = original_manifest.read_bytes()

    if change == "source":
        source.write_text("RESULT = 2\n")
    elif change == "package_version":
        versions["numpy"] = "2.3.1"
    else:
        library.write_bytes(b"replacement native build")

    # Resume receives the startup snapshot, not a later re-scan that could
    # incorrectly associate an in-flight computation with changed sources.
    assert build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], execution_identity=startup_identity,
    ) == initial_context
    with pytest.raises(RuntimeError, match="Execution identity changed during the stage"):
        reproducibility.write_run_manifest(**manifest_arguments)
    assert list(original_manifest.parent.glob("*.json")) == [original_manifest]
    assert original_manifest.read_bytes() == original_manifest_bytes
    assert output.read_bytes() == b"historical result is retained"


def test_unchanged_stage_preserves_startup_identity_in_manifest(
    tmp_path: Path, identity_inputs: tuple[Path, Path, dict[str, str]],
) -> None:
    startup_identity = reproducibility.collect_execution_identity(tmp_path)
    path = reproducibility.write_run_manifest(
        repo_root=tmp_path, manifests_dir=tmp_path / "manifests", script_name="audit_stage",
        started_at=datetime.now(UTC), config_paths=[], input_artifact_paths=[],
        output_artifact_paths=[], execution_identity=startup_identity,
    )
    payload = orjson.loads(path.read_bytes())
    assert payload["execution_identity"] == startup_identity
    assert payload["execution_identity_capture"] == "stage_start"

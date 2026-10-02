from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import orjson
import pytest

from ivcast import reproducibility
from ivcast.io.atomic import write_bytes_atomic


def test_new_atomic_output_never_overwrites_existing_file(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    write_bytes_atomic(output, b"first", overwrite=False)
    with pytest.raises(FileExistsError):
        write_bytes_atomic(output, b"second", overwrite=False)
    assert output.read_bytes() == b"first"
    assert list(tmp_path.glob("*.tmp")) == []


def test_manifest_calls_at_same_instant_preserve_both_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FixedClock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> FixedClock:
            return cls(2026, 9, 7, 20, 0, 0, tzinfo=UTC)

    monkeypatch.setattr(reproducibility, "datetime", FixedClock)
    monkeypatch.setattr(reproducibility, "git_commit_hash", lambda _: "same-commit")
    monkeypatch.setattr(reproducibility, "collect_package_versions", dict)
    monkeypatch.setattr(reproducibility, "collect_hardware_metadata", dict)
    common: dict[str, Any] = {
        "repo_root": tmp_path,
        "manifests_dir": tmp_path,
        "script_name": "05_tune_models",
        "started_at": FixedClock.now(),
        "config_paths": [],
        "input_artifact_paths": [],
        "output_artifact_paths": [],
    }
    first = reproducibility.write_run_manifest(**common, extra_metadata={"model": "ridge"})
    second = reproducibility.write_run_manifest(**common, extra_metadata={"model": "elasticnet"})
    assert first != second
    assert {orjson.loads(path.read_bytes())["extra_metadata"]["model"] for path in (
        first, second
    )} == {"ridge", "elasticnet"}
    assert len(list(first.parent.glob("*_05_tune_models.json"))) == 2
    assert "execution_identity" in orjson.loads(first.read_bytes())

    # Even a deliberately repeated run identifier must fail rather than erase.
    monkeypatch.setattr(reproducibility, "uuid4", lambda: UUID(int=0))
    third = reproducibility.write_run_manifest(**common, extra_metadata={"model": "har_factor"})
    with pytest.raises(FileExistsError):
        reproducibility.write_run_manifest(**common, extra_metadata={"model": "overwrite"})
    assert orjson.loads(third.read_bytes())["extra_metadata"]["model"] == "har_factor"


def test_execution_identity_records_dirty_code_despite_same_git_revision(tmp_path: Path) -> None:
    code = tmp_path / "scripts" / "run.py"
    code.parent.mkdir()
    code.write_text("RESULT = 1\n", encoding="utf-8")
    before = reproducibility.collect_execution_identity(tmp_path)
    code.write_text("RESULT = 2\n", encoding="utf-8")
    after = reproducibility.collect_execution_identity(tmp_path)
    assert before["source_sha256"] != after["source_sha256"]
    assert after["source_files"] == [
        {"path": "scripts/run.py", "sha256": reproducibility.sha256_file(code)}
    ]

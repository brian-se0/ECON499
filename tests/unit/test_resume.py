from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from ivcast import reproducibility
from ivcast.io.atomic import write_text_atomic
from ivcast.resume import StageResumer, build_resume_context_hash, resume_state_path


def test_stage_resumer_resets_on_context_change(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "output.txt"
    config_path.write_text("alpha: 1\n", encoding="utf-8")
    input_path.write_text("input\n", encoding="utf-8")
    write_text_atomic(output_path, "done\n")

    initial_context = build_resume_context_hash(
        config_paths=[config_path],
        input_artifact_paths=[input_path],
    )
    first = StageResumer(
        state_path=resume_state_path(tmp_path, "unit_stage"),
        stage_name="unit_stage",
        context_hash=initial_context,
    )
    first.mark_complete(
        "item_a",
        output_paths=[output_path],
        metadata={"rows": 1},
    )
    assert first.item_complete("item_a", required_output_paths=[output_path])

    config_path.write_text("alpha: 2\n", encoding="utf-8")
    updated_context = build_resume_context_hash(
        config_paths=[config_path],
        input_artifact_paths=[input_path],
    )
    second = StageResumer(
        state_path=resume_state_path(tmp_path, "unit_stage"),
        stage_name="unit_stage",
        context_hash=updated_context,
    )
    assert not second.item_complete("item_a", required_output_paths=[output_path])
    assert second.output_paths_for("item_a") == []


def test_stage_resumer_tracks_outputless_completed_items(tmp_path: Path) -> None:
    context = build_resume_context_hash(config_paths=[], input_artifact_paths=[])
    resumer = StageResumer(
        state_path=resume_state_path(tmp_path, "unit_stage"),
        stage_name="unit_stage",
        context_hash=context,
    )
    resumer.mark_complete("item_without_outputs", output_paths=[], metadata={"status": "skipped"})

    assert resumer.item_complete("item_without_outputs")
    assert resumer.metadata_for("item_without_outputs")["status"] == "skipped"


def test_resume_invalidates_uncommitted_source_and_runtime_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "src" / "ivcast" / "transform.py"
    source.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(reproducibility, "collect_package_versions", lambda: {"numpy": "1"})
    original = build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], repo_root=tmp_path
    )
    output = tmp_path / "output.txt"
    output.write_text("old result", encoding="utf-8")
    state_path = tmp_path / "state.json"
    resumer = StageResumer(state_path=state_path, stage_name="test", context_hash=original)
    resumer.mark_complete("item", output_paths=[output])

    (tmp_path / "README.md").write_text("Documentation only", encoding="utf-8")
    source.touch()
    assert build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], repo_root=tmp_path
    ) == original

    source.write_text("VALUE = 2\n", encoding="utf-8")
    changed = build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], repo_root=tmp_path
    )
    assert changed != original
    assert not StageResumer(
        state_path=state_path, stage_name="test", context_hash=changed
    ).item_complete("item")
    monkeypatch.setattr(reproducibility, "collect_package_versions", lambda: {"numpy": "2"})
    assert build_resume_context_hash(
        config_paths=[], input_artifact_paths=[], repo_root=tmp_path
    ) != changed


@pytest.mark.parametrize("replacement", [b"", b"evil"])
def test_resume_rejects_modified_outputs(tmp_path: Path, replacement: bytes) -> None:
    output = tmp_path / "output.bin"
    output.write_bytes(b"good")
    state_path = tmp_path / "state.json"
    resumer = StageResumer(state_path=state_path, stage_name="test", context_hash="context")
    resumer.mark_complete("item", output_paths=[output])
    assert resumer.item_complete("item", required_output_paths=[output])
    output.write_bytes(replacement)
    reloaded = StageResumer(state_path=state_path, stage_name="test", context_hash="context")
    assert not reloaded.item_complete("item", required_output_paths=[output])


def test_resume_checks_all_recorded_outputs_and_rejects_unrecorded_paths(tmp_path: Path) -> None:
    first, second, unrecorded = [tmp_path / name for name in ("first", "second", "extra")]
    for path in (first, second, unrecorded):
        path.write_bytes(b"data")
    resumer = StageResumer(
        state_path=tmp_path / "state.json", stage_name="test", context_hash="context"
    )
    resumer.mark_complete("item", output_paths=[first, second])
    assert not resumer.item_complete("item", required_output_paths=[unrecorded])
    second.unlink()
    assert not resumer.item_complete("item", required_output_paths=[first])


def test_legacy_resume_ledger_is_invalidated_without_deleting_outputs(tmp_path: Path) -> None:
    output = tmp_path / "result.bin"
    output.write_bytes(b"historical result")
    state_path = tmp_path / "state.json"
    state_path.write_bytes(orjson.dumps({
        "schema_version": 1,
        "stage_name": "test",
        "context_hash": "context",
        "items": {"item": {"output_paths": [str(output)], "completed_at_utc": "old"}},
    }))
    resumer = StageResumer(state_path=state_path, stage_name="test", context_hash="context")
    assert not resumer.item_complete("item")
    assert output.read_bytes() == b"historical result"
    assert orjson.loads(state_path.read_bytes())["schema_version"] == 2

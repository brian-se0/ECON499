from __future__ import annotations

from pathlib import Path

from ivsurf.io.atomic import write_text_atomic
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path


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

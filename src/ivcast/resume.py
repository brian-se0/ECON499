"""Context-aware stage resume ledgers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from ivcast.io.atomic import write_bytes_atomic
from ivcast.reproducibility import (
    collect_artifact_records,
    collect_execution_identity,
    sha256_bytes,
    sha256_file,
    snapshot_configs,
)

RESUME_SCHEMA_VERSION = 2


def resume_state_path(manifests_dir: Path, stage_name: str) -> Path:
    """Return the canonical resume ledger path for one stage."""

    return manifests_dir / "resume" / stage_name / "state.json"


@dataclass(frozen=True, slots=True)
class ResumeItemState:
    """One completed resumable item within a stage."""

    output_paths: tuple[str, ...]
    output_hashes: dict[str, str]
    completed_at_utc: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ResumeState:
    """Serialized resume ledger for one stage and one input/config context."""

    schema_version: int
    stage_name: str
    context_hash: str
    items: dict[str, ResumeItemState]


def build_resume_context_hash(
    *,
    config_paths: list[Path],
    input_artifact_paths: list[Path],
    extra_tokens: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    execution_identity: Mapping[str, Any] | None = None,
) -> str:
    """Hash the executable stage context used to decide resume compatibility."""

    payload = {
        "execution_identity": (
            collect_execution_identity(repo_root)
            if execution_identity is None
            else dict(execution_identity)
        ),
        "config_snapshots": [asdict(snapshot) for snapshot in snapshot_configs(config_paths)],
        "input_artifacts": [
            asdict(record) for record in collect_artifact_records(input_artifact_paths)
        ],
        "extra_tokens": dict(extra_tokens or {}),
    }
    return sha256_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS))


class StageResumer:
    """Context-aware item ledger for resumable stage execution."""

    def __init__(
        self,
        *,
        state_path: Path,
        stage_name: str,
        context_hash: str,
    ) -> None:
        self._state_path = state_path
        self._stage_name = stage_name
        self._context_hash = context_hash
        self._state = self._load_or_initialize()

    @property
    def context_hash(self) -> str:
        return self._context_hash

    def _empty_state(self) -> ResumeState:
        return ResumeState(
            schema_version=RESUME_SCHEMA_VERSION,
            stage_name=self._stage_name,
            context_hash=self._context_hash,
            items={},
        )

    def _load_or_initialize(self) -> ResumeState:
        if not self._state_path.exists():
            state = self._empty_state()
            self._persist(state)
            return state
        payload = orjson.loads(self._state_path.read_bytes())
        # Legacy records have no output hashes or executable identity. Do not
        # silently bless their outputs when upgrading the ledger format.
        if (
            payload.get("schema_version") != RESUME_SCHEMA_VERSION
            or payload.get("stage_name") != self._stage_name
            or payload.get("context_hash") != self._context_hash
        ):
            state = self._empty_state()
            self._persist(state)
            return state
        item_payload = payload.get("items", {})
        items = {
            str(item_id): ResumeItemState(
                output_paths=tuple(str(path) for path in item_state["output_paths"]),
                output_hashes={
                    str(path): str(digest)
                    for path, digest in item_state.get("output_hashes", {}).items()
                },
                completed_at_utc=str(item_state["completed_at_utc"]),
                metadata=dict(item_state.get("metadata", {})),
            )
            for item_id, item_state in item_payload.items()
        }
        loaded = ResumeState(
            schema_version=int(payload["schema_version"]),
            stage_name=str(payload["stage_name"]),
            context_hash=str(payload["context_hash"]),
            items=items,
        )
        return loaded

    def _persist(self, state: ResumeState) -> None:
        payload = {
            "schema_version": state.schema_version,
            "stage_name": state.stage_name,
            "context_hash": state.context_hash,
            "items": {
                item_id: asdict(item_state) for item_id, item_state in sorted(state.items.items())
            },
        }
        write_bytes_atomic(
            self._state_path,
            orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
        )

    def item_complete(
        self,
        item_id: str,
        *,
        required_output_paths: list[Path] | None = None,
    ) -> bool:
        """Return whether an item is completed for the current context."""

        record = self._state.items.get(item_id)
        if record is None:
            return False
        # Verify every originally produced output, even if the caller requests
        # only a subset; adding an unrecorded required output must also fail.
        recorded_paths = set(record.output_paths)
        if set(record.output_hashes) != recorded_paths:
            return False
        required_paths = {str(path.resolve()) for path in (required_output_paths or [])}
        if not required_paths.issubset(recorded_paths):
            return False
        for stored_path, expected_hash in record.output_hashes.items():
            path = Path(stored_path)
            if not path.is_file():
                return False
            try:
                if sha256_file(path) != expected_hash:
                    return False
            except FileNotFoundError:
                return False
        return True

    def mark_complete(
        self,
        item_id: str,
        *,
        output_paths: list[Path],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist one item as completed for the current context."""

        output_records = collect_artifact_records(output_paths)
        updated = dict(self._state.items)
        updated[item_id] = ResumeItemState(
            output_paths=tuple(record.path for record in output_records),
            output_hashes={record.path: record.sha256 for record in output_records},
            completed_at_utc=datetime.now(UTC).isoformat(),
            metadata=dict(metadata or {}),
        )
        self._state = ResumeState(
            schema_version=self._state.schema_version,
            stage_name=self._state.stage_name,
            context_hash=self._state.context_hash,
            items=updated,
        )
        self._persist(self._state)

    def clear_item(self, item_id: str, *, output_paths: list[Path]) -> None:
        """Discard one item's recorded state and owned output files."""

        for output_path in output_paths:
            if output_path.exists():
                output_path.unlink()
        if item_id not in self._state.items:
            return
        updated = dict(self._state.items)
        updated.pop(item_id)
        self._state = ResumeState(
            schema_version=self._state.schema_version,
            stage_name=self._state.stage_name,
            context_hash=self._state.context_hash,
            items=updated,
        )
        self._persist(self._state)

    def metadata_for(self, item_id: str) -> dict[str, Any]:
        """Return stored metadata for a completed item."""

        record = self._state.items.get(item_id)
        if record is None:
            message = f"No resume metadata recorded for {item_id!r}."
            raise KeyError(message)
        return dict(record.metadata)

    def output_paths_for(self, item_id: str) -> list[Path]:
        """Return stored output paths for a completed item."""

        record = self._state.items.get(item_id)
        if record is None:
            return []
        return [Path(path) for path in record.output_paths]

    def completed_metadata_in_order(self, item_ids: list[str]) -> list[dict[str, Any]]:
        """Return metadata for completed items in caller-specified deterministic order."""

        rows: list[dict[str, Any]] = []
        for item_id in item_ids:
            if item_id in self._state.items:
                rows.append(self.metadata_for(item_id))
        return rows

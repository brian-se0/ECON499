"""Run-manifest persistence and reproducibility metadata helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from platform import machine, platform, processor, python_version
from subprocess import CalledProcessError, run
from typing import Any

import orjson
import torch

from ivsurf.io.atomic import write_bytes_atomic


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    """Content hash and basic metadata for one artifact."""

    path: str
    sha256: str
    size_bytes: int
    modified_at_utc: str


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """Exact config-file snapshot included in a run manifest."""

    path: str
    sha256: str
    content: str


def sha256_bytes(raw: bytes) -> str:
    """Return a SHA256 hash for the provided byte string."""

    return sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without loading it all into Python-managed text objects."""

    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique = {path.resolve(): path.resolve() for path in paths}
    return [unique[key] for key in sorted(unique)]


def _artifact_record(path: Path) -> ArtifactRecord:
    if not path.exists():
        message = f"Expected artifact path does not exist: {path}"
        raise FileNotFoundError(message)
    if not path.is_file():
        message = f"Expected file artifact path, found non-file path: {path}"
        raise ValueError(message)
    stat = path.stat()
    return ArtifactRecord(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        size_bytes=stat.st_size,
        modified_at_utc=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
    )


def collect_artifact_records(paths: list[Path]) -> list[ArtifactRecord]:
    """Collect sorted artifact metadata for a set of files."""

    return [_artifact_record(path) for path in _unique_paths(paths)]


def combined_hash(records: list[ArtifactRecord]) -> str | None:
    """Hash a collection of artifact records deterministically."""

    if not records:
        return None
    payload = [asdict(record) for record in records]
    raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return sha256_bytes(raw)


def snapshot_configs(config_paths: list[Path]) -> list[ConfigSnapshot]:
    """Read config files verbatim so runs can be reconstructed exactly."""

    snapshots: list[ConfigSnapshot] = []
    for path in _unique_paths(config_paths):
        raw = path.read_text(encoding="utf-8")
        snapshots.append(
            ConfigSnapshot(
                path=str(path.resolve()),
                sha256=sha256_bytes(raw.encode("utf-8")),
                content=raw,
            )
        )
    return snapshots


def git_commit_hash(repo_root: Path) -> str | None:
    """Return the current git commit hash when available."""

    try:
        result = run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=True,
            text=True,
        )
    except (CalledProcessError, FileNotFoundError):
        return None
    commit = result.stdout.strip()
    return commit or None


def collect_package_versions() -> dict[str, str]:
    """Collect installed package versions for auditability."""

    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name is None:
            continue
        versions[name] = distribution.version
    return dict(sorted(versions.items()))


def collect_hardware_metadata() -> dict[str, Any]:
    """Collect deterministic hardware metadata relevant to reproducibility."""

    gpu_devices: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "multi_processor_count": int(properties.multi_processor_count),
                }
            )

    return {
        "platform": platform(),
        "python_version": python_version(),
        "machine": machine(),
        "processor": processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": gpu_devices,
    }


def _log_to_mlflow(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    tracking_uri: str,
    experiment_name: str,
) -> str:
    try:
        import mlflow
    except ImportError as exc:
        message = (
            "MLflow logging was requested, but mlflow is not installed. "
            "Install the tracking extra with `uv sync --extra tracking`."
        )
        raise RuntimeError(message) from exc

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=str(manifest["script_name"])) as run_context:
        mlflow.set_tags(
            {
                "script_name": str(manifest["script_name"]),
                "git_commit_hash": str(manifest.get("git_commit_hash")),
            }
        )
        random_seed = manifest.get("random_seed")
        if random_seed is not None:
            mlflow.log_param("random_seed", random_seed)
        data_manifest_hash = manifest.get("data_manifest_hash")
        if data_manifest_hash is not None:
            mlflow.log_param("data_manifest_hash", data_manifest_hash)
        split_manifest_hash = manifest.get("split_manifest_hash")
        if split_manifest_hash is not None:
            mlflow.log_param("split_manifest_hash", split_manifest_hash)
        mlflow.log_artifact(str(manifest_path), artifact_path="run_manifests")
        run_id = str(run_context.info.run_id)
    return run_id


def write_run_manifest(
    *,
    manifests_dir: Path,
    repo_root: Path,
    script_name: str,
    started_at: datetime,
    config_paths: list[Path],
    input_artifact_paths: list[Path],
    output_artifact_paths: list[Path],
    data_manifest_paths: list[Path] | None = None,
    split_manifest_path: Path | None = None,
    random_seed: int | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> Path:
    """Persist a complete run manifest for one pipeline stage."""

    finished_at = datetime.now(UTC)
    config_snapshots = snapshot_configs(config_paths)
    input_records = collect_artifact_records(input_artifact_paths)
    output_records = collect_artifact_records(output_artifact_paths)
    if data_manifest_paths is None:
        data_manifest_paths = input_artifact_paths
    data_manifest_hash = combined_hash(collect_artifact_records(data_manifest_paths))
    split_manifest_hash = (
        sha256_file(split_manifest_path.resolve()) if split_manifest_path is not None else None
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "script_name": script_name,
        "started_at_utc": started_at.astimezone(UTC).isoformat(),
        "completed_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at.astimezone(UTC)).total_seconds(),
        "git_commit_hash": git_commit_hash(repo_root),
        "random_seed": random_seed,
        "config_snapshots": [asdict(snapshot) for snapshot in config_snapshots],
        "package_versions": collect_package_versions(),
        "hardware_metadata": collect_hardware_metadata(),
        "data_manifest_hash": data_manifest_hash,
        "split_manifest_hash": split_manifest_hash,
        "input_artifacts": [asdict(record) for record in input_records],
        "output_artifacts": [asdict(record) for record in output_records],
        "extra_metadata": dict(extra_metadata or {}),
    }

    run_dir = manifests_dir / "runs" / script_name
    run_dir.mkdir(parents=True, exist_ok=True)
    timestamp = finished_at.strftime("%Y%m%dT%H%M%SZ")
    manifest_path = run_dir / f"{timestamp}_{script_name}.json"
    write_bytes_atomic(
        manifest_path,
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
    )

    if mlflow_tracking_uri is not None:
        run_id = _log_to_mlflow(
            manifest_path=manifest_path,
            manifest=manifest,
            tracking_uri=mlflow_tracking_uri,
            experiment_name=mlflow_experiment_name,
        )
        manifest["mlflow_run_id"] = run_id
        write_bytes_atomic(
            manifest_path,
            orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)
        )

    return manifest_path

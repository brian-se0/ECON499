"""Run-manifest persistence and reproducibility metadata helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from platform import machine, platform, processor, python_implementation, python_version
from subprocess import CalledProcessError, run
from typing import Any
from uuid import uuid4

import orjson
import torch

from ivcast.io.atomic import write_bytes_atomic


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


def _is_appledouble(path: Path) -> bool:
    """Return whether a path is macOS AppleDouble metadata (`._name`), not content."""

    return path.name.startswith("._")


def collect_package_versions() -> dict[str, str]:
    """Collect installed package versions for auditability.

    macOS writes AppleDouble `._*` metadata files on non-Apple filesystems; a
    `._name.egg-info` file on sys.path is not a distribution and is skipped.
    """

    versions: dict[str, str] = {}
    for distribution in metadata.distributions():
        distribution_path = getattr(distribution, "_path", None)
        if isinstance(distribution_path, Path) and _is_appledouble(distribution_path):
            continue
        name = distribution.metadata.get("Name")
        if name is None:
            continue
        versions[name] = distribution.version
    return dict(sorted(versions.items()))


def collect_execution_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Fingerprint executable sources and runtime, including uncommitted code.

    Hash the whole pipeline conservatively: a shared helper change can affect
    multiple stages. Documentation-only edits and file mtimes do not invalidate
    this identity. Recalculate on each stage invocation, never cache across runs.
    Stages must start in a fresh process without source edits after imports; this
    fingerprints disk contents, not already-imported Python bytecode.
    """

    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    source_paths = {
        *root.glob("src/ivcast/**/*.py"),
        *root.glob("scripts/**/*.py"),
        *(path for path in (root / "pyproject.toml", root / "uv.lock") if path.is_file()),
    }
    source_records = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(source_paths)
        if path.is_file() and not _is_appledouble(path)
    ]
    return {
        "source_files": source_records,
        "source_sha256": sha256_bytes(orjson.dumps(source_records, option=orjson.OPT_SORT_KEYS)),
        "runtime": {
            "python_version": python_version(),
            "python_implementation": python_implementation(),
            "platform": platform(),
            "machine": machine(),
            "torch_cuda_version": torch.version.cuda,
            "package_versions": collect_package_versions(),
            "native_libraries": collect_native_library_hashes(),
        },
    }


def collect_native_library_hashes() -> dict[str, dict[str, str]]:
    """Distinguish supported same-version LightGBM builds without loading the library."""

    from ivcast.runtime_preflight import _lightgbm_library_path

    library_path = _lightgbm_library_path().resolve()
    return {
        "lightgbm": {"path": str(library_path), "sha256": sha256_file(library_path)},
    }


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
    execution_identity: Mapping[str, Any] | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivcast",
) -> Path:
    """Persist a run manifest after checking the stage's captured execution identity.

    Official stages pass their startup snapshot, also used for resume decisions.
    Auxiliary callers without a snapshot record only the state at manifest write.
    """

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
    commit_hash = git_commit_hash(repo_root)
    if commit_hash is None:
        message = (
            "Cannot write run manifest without a git commit hash. "
            f"repo_root={repo_root.resolve()} is not a usable Git checkout."
        )
        raise RuntimeError(message)
    completed_identity = collect_execution_identity(repo_root)
    if execution_identity is not None and completed_identity != execution_identity:
        message = (
            "Execution identity changed during the stage: source files, package versions, "
            "or native runtime libraries differ from the startup snapshot. "
            "No run manifest was published; rerun from a fresh process with stable sources "
            "and runtime."
        )
        raise RuntimeError(message)
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "script_name": script_name,
        "started_at_utc": started_at.astimezone(UTC).isoformat(),
        "completed_at_utc": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at.astimezone(UTC)).total_seconds(),
        "git_commit_hash": commit_hash,
        "execution_identity": (
            completed_identity if execution_identity is None else dict(execution_identity)
        ),
        "execution_identity_capture": (
            "manifest_write" if execution_identity is None else "stage_start"
        ),
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
    timestamp = finished_at.strftime("%Y%m%dT%H%M%S%fZ")
    manifest_path = run_dir / f"{timestamp}_{uuid4().hex}_{script_name}.json"
    write_bytes_atomic(
        manifest_path,
        orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS),
        overwrite=False,
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

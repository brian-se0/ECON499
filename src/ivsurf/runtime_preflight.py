"""Official runtime preflight checks for supported hardware profiles."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ivsurf.config import NeuralModelConfig, RawDataConfig, load_yaml_config

WINDOWS_CUDA_PROFILE = "windows_cuda"
MAC_CPU_PROFILE = "mac_cpu"
SUPPORTED_RUNTIME_PROFILES = frozenset({WINDOWS_CUDA_PROFILE, MAC_CPU_PROFILE})


@dataclass(frozen=True, slots=True)
class RuntimePreflightReport:
    """Validated runtime facts for the official pipeline contract."""

    runtime_profile_name: str
    platform_system: str
    raw_options_dir: Path
    torch_cuda_available: bool
    lightgbm_gpu_available: bool
    lightgbm_cpu_available: bool
    lightgbm_openmp_linked: bool | None


def _require_existing_file(path: Path, description: str) -> None:
    if not path.exists():
        message = f"Required {description} does not exist: {path}"
        raise FileNotFoundError(message)
    if not path.is_file():
        message = f"Required {description} is not a file: {path}"
        raise FileNotFoundError(message)


def _require_existing_directory(path: Path, description: str) -> None:
    if not path.exists():
        message = f"Required {description} does not exist: {path}"
        raise FileNotFoundError(message)
    if not path.is_dir():
        message = f"Required {description} is not a directory: {path}"
        raise NotADirectoryError(message)


def _check_lightgbm_runtime(lightgbm_params: dict[str, object], *, device_type: str) -> None:
    from lightgbm import LGBMRegressor

    params: dict[str, Any] = {
        key: value
        for key, value in lightgbm_params.items()
        if key != "model_name"
    }
    params.setdefault("device_type", device_type)
    params.setdefault("n_estimators", 5)
    params.setdefault("learning_rate", 0.1)
    params.setdefault("num_leaves", 7)
    params.setdefault("max_depth", 3)
    params.setdefault("min_child_samples", 1)
    params.setdefault("feature_fraction", 1.0)
    params.setdefault("lambda_l2", 0.0)
    params.setdefault("objective", "regression")
    params.setdefault("metric", "l2")
    params.setdefault("verbosity", -1)
    params.setdefault("random_state", 7)

    features = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    targets = np.asarray([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    model = LGBMRegressor(**cast(dict[str, Any], params))
    model.fit(features, targets)


def _lightgbm_library_path() -> Path:
    spec = find_spec("lightgbm")
    if spec is None or spec.origin is None:
        message = "LightGBM is not importable in the current Python environment."
        raise RuntimeError(message)

    package_dir = Path(spec.origin).resolve().parent
    candidates = (
        package_dir / "lib" / "lib_lightgbm.dylib",
        package_dir / "lib" / "lib_lightgbm.so",
        package_dir / "lib" / "lib_lightgbm.dll",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    message = f"Could not locate the LightGBM native library under {package_dir}."
    raise RuntimeError(message)


def _lightgbm_links_openmp(library_path: Path) -> bool:
    result = subprocess.run(
        ["otool", "-L", str(library_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    linked_libraries = result.stdout.casefold()
    return any(token in linked_libraries for token in ("libomp", "libgomp", "libiomp"))


def _require_supported_runtime_profile(run_profile_name: str) -> None:
    if run_profile_name not in SUPPORTED_RUNTIME_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_RUNTIME_PROFILES))
        message = (
            f"Unsupported runtime profile {run_profile_name!r}. "
            f"Supported profiles: {supported}."
        )
        raise RuntimeError(message)


def _run_windows_cuda_preflight(
    *,
    raw_config: RawDataConfig,
    lightgbm_params: dict[str, object],
    neural_config: NeuralModelConfig,
) -> RuntimePreflightReport:
    if platform.system() != "Windows":
        message = (
            "The Windows/CUDA runtime profile requires Windows. "
            f"Detected platform.system()={platform.system()!r}."
        )
        raise RuntimeError(message)
    if neural_config.device != "cuda":
        message = (
            "The Windows/CUDA neural runtime must use CUDA. "
            f"Found neural_surface device={neural_config.device!r}."
        )
        raise RuntimeError(message)
    if lightgbm_params.get("device_type") != "gpu":
        message = (
            "The Windows/CUDA LightGBM runtime must use GPU mode. "
            f"Found device_type={lightgbm_params.get('device_type')!r}."
        )
        raise RuntimeError(message)

    _require_existing_directory(raw_config.raw_options_dir, "raw options root")

    torch_cuda_available = bool(torch.cuda.is_available())
    if not torch_cuda_available:
        message = "PyTorch CUDA support is unavailable in the current environment."
        raise RuntimeError(message)

    _check_lightgbm_runtime(lightgbm_params, device_type="gpu")

    return RuntimePreflightReport(
        runtime_profile_name=WINDOWS_CUDA_PROFILE,
        platform_system=platform.system(),
        raw_options_dir=raw_config.raw_options_dir,
        torch_cuda_available=torch_cuda_available,
        lightgbm_gpu_available=True,
        lightgbm_cpu_available=False,
        lightgbm_openmp_linked=None,
    )


def _run_mac_cpu_preflight(
    *,
    raw_config: RawDataConfig,
    lightgbm_params: dict[str, object],
    neural_config: NeuralModelConfig,
) -> RuntimePreflightReport:
    if platform.system() != "Darwin":
        message = (
            "The Mac CPU runtime profile requires macOS. "
            f"Detected platform.system()={platform.system()!r}."
        )
        raise RuntimeError(message)
    if neural_config.device != "cpu":
        message = (
            "The Mac CPU neural runtime must use CPU. "
            f"Found neural_surface device={neural_config.device!r}."
        )
        raise RuntimeError(message)
    if lightgbm_params.get("device_type") != "cpu":
        message = (
            "The Mac CPU LightGBM runtime must use CPU mode. "
            f"Found device_type={lightgbm_params.get('device_type')!r}."
        )
        raise RuntimeError(message)
    if lightgbm_params.get("n_jobs") != 1:
        message = (
            "The Mac CPU LightGBM runtime must use n_jobs=1 to avoid native "
            f"thread-runtime instability. Found n_jobs={lightgbm_params.get('n_jobs')!r}."
        )
        raise RuntimeError(message)

    _require_existing_directory(raw_config.raw_options_dir, "raw options root")

    library_path = _lightgbm_library_path()
    lightgbm_openmp_linked = _lightgbm_links_openmp(library_path)
    if lightgbm_openmp_linked:
        message = (
            "The Mac CPU runtime requires a LightGBM build without OpenMP. "
            f"The installed native library links OpenMP: {library_path}. "
            "Reinstall LightGBM from source with CMake USE_OPENMP=OFF before "
            "running the Mac profile."
        )
        raise RuntimeError(message)

    _check_lightgbm_runtime(lightgbm_params, device_type="cpu")

    return RuntimePreflightReport(
        runtime_profile_name=MAC_CPU_PROFILE,
        platform_system=platform.system(),
        raw_options_dir=raw_config.raw_options_dir,
        torch_cuda_available=False,
        lightgbm_gpu_available=False,
        lightgbm_cpu_available=True,
        lightgbm_openmp_linked=False,
    )


def run_runtime_preflight(
    *,
    raw_config_path: Path,
    lightgbm_config_path: Path,
    neural_config_path: Path,
    run_profile_name: str = WINDOWS_CUDA_PROFILE,
) -> RuntimePreflightReport:
    """Validate one supported runtime profile contract or fail fast."""

    _require_supported_runtime_profile(run_profile_name)
    for config_path, description in (
        (raw_config_path, "raw config"),
        (lightgbm_config_path, "LightGBM model config"),
        (neural_config_path, "neural model config"),
    ):
        _require_existing_file(config_path, description)

    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    lightgbm_params = load_yaml_config(lightgbm_config_path)
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))

    if run_profile_name == WINDOWS_CUDA_PROFILE:
        return _run_windows_cuda_preflight(
            raw_config=raw_config,
            lightgbm_params=lightgbm_params,
            neural_config=neural_config,
        )
    return _run_mac_cpu_preflight(
        raw_config=raw_config,
        lightgbm_params=lightgbm_params,
        neural_config=neural_config,
    )

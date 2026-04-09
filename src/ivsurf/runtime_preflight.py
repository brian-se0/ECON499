"""Official runtime preflight checks for the Windows/GPU pipeline."""

from __future__ import annotations

import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from lightgbm import LGBMRegressor

from ivsurf.config import NeuralModelConfig, RawDataConfig, load_yaml_config


@dataclass(frozen=True, slots=True)
class RuntimePreflightReport:
    """Validated runtime facts for the official pipeline contract."""

    platform_system: str
    raw_options_dir: Path
    torch_cuda_available: bool
    lightgbm_gpu_available: bool


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


def _check_lightgbm_gpu_runtime(lightgbm_params: dict[str, object]) -> None:
    params: dict[str, Any] = {
        key: value
        for key, value in lightgbm_params.items()
        if key != "model_name"
    }
    params.setdefault("device_type", "gpu")
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


def run_runtime_preflight(
    *,
    raw_config_path: Path,
    lightgbm_config_path: Path,
    neural_config_path: Path,
) -> RuntimePreflightReport:
    """Validate the official Windows/GPU runtime contract or fail fast."""

    for config_path, description in (
        (raw_config_path, "raw config"),
        (lightgbm_config_path, "LightGBM model config"),
        (neural_config_path, "neural model config"),
    ):
        _require_existing_file(config_path, description)

    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    lightgbm_params = load_yaml_config(lightgbm_config_path)
    neural_config = NeuralModelConfig.model_validate(load_yaml_config(neural_config_path))

    if platform.system() != "Windows":
        message = (
            "The official runtime contract is Windows-only. "
            f"Detected platform.system()={platform.system()!r}."
        )
        raise RuntimeError(message)
    if neural_config.device != "cuda":
        message = (
            "The official neural runtime must use CUDA. "
            f"Found neural_surface device={neural_config.device!r}."
        )
        raise RuntimeError(message)
    if lightgbm_params.get("device_type") != "gpu":
        message = (
            "The official LightGBM runtime must use GPU mode. "
            f"Found device_type={lightgbm_params.get('device_type')!r}."
        )
        raise RuntimeError(message)

    _require_existing_directory(raw_config.raw_options_dir, "raw options root")

    torch_cuda_available = bool(torch.cuda.is_available())
    if not torch_cuda_available:
        message = "PyTorch CUDA support is unavailable in the current environment."
        raise RuntimeError(message)

    _check_lightgbm_gpu_runtime(lightgbm_params)

    return RuntimePreflightReport(
        platform_system=platform.system(),
        raw_options_dir=raw_config.raw_options_dir,
        torch_cuda_available=torch_cuda_available,
        lightgbm_gpu_available=True,
    )

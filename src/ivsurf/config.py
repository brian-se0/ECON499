"""Typed configuration models and loaders."""

from __future__ import annotations

from datetime import date, time
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class RawDataConfig(BaseModel):
    """Locations and raw-data scope."""

    model_config = ConfigDict(extra="forbid")

    raw_options_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    manifests_dir: Path
    raw_file_glob: str = "UnderlyingOptionsEODCalcs_*.zip"
    target_symbol: str = "^SPX"
    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    decision_time: time = time(15, 45)
    decision_snapshot_minutes_before_close: int = Field(default=15, ge=0)
    sample_start_date: date = date(2004, 1, 2)
    sample_end_date: date = date(2021, 4, 9)
    am_settled_roots: tuple[str, ...] = ("SPX",)

    @field_validator("sample_end_date")
    @classmethod
    def validate_sample_window_order(cls, value: date, info: Any) -> date:
        start_date = info.data.get("sample_start_date")
        if isinstance(start_date, date) and value < start_date:
            message = "sample_end_date must be on or after sample_start_date."
            raise ValueError(message)
        return value


class MarketCalendarConfig(BaseModel):
    """Calendar and timing rules."""

    model_config = ConfigDict(extra="forbid")

    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    decision_time: time = time(15, 45)
    decision_snapshot_minutes_before_close: int = Field(default=15, ge=0)
    am_settled_roots: tuple[str, ...] = ("SPX",)


class CleaningConfig(BaseModel):
    """Quote-level cleaning rules."""

    model_config = ConfigDict(extra="forbid")

    target_symbol: str = "^SPX"
    allowed_option_types: tuple[str, ...] = ("C", "P")
    min_valid_bid_exclusive: float = 0.0
    min_valid_ask_exclusive: float = 0.0
    require_ask_ge_bid: bool = True
    require_positive_iv: bool = True
    require_positive_vega: bool = True
    require_positive_underlying_price: bool = True
    min_valid_mid_price_exclusive: float = 0.0
    max_abs_log_moneyness: float = 0.5
    min_tau_years: float = 1.0e-4
    max_tau_years: float = 2.5


class SurfaceGridConfig(BaseModel):
    """Fixed surface grid and deterministic completion parameters."""

    model_config = ConfigDict(extra="forbid")

    moneyness_points: tuple[float, ...]
    maturity_days: tuple[int, ...]
    interpolation_order: tuple[str, ...] = ("maturity", "moneyness")
    interpolation_cycles: PositiveInt = 2
    total_variance_floor: float = 1.0e-8
    observed_cell_min_count: PositiveInt = 1

    @field_validator("interpolation_order")
    @classmethod
    def validate_interpolation_order(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        allowed = {"maturity", "moneyness"}
        if set(values) != allowed:
            message = "interpolation_order must contain exactly 'maturity' and 'moneyness'"
            raise ValueError(message)
        return values


class FeatureConfig(BaseModel):
    """Daily feature construction parameters."""

    model_config = ConfigDict(extra="forbid")

    lag_windows: tuple[PositiveInt, ...] = (1, 5, 22)
    include_daily_change: bool = True
    include_mask: bool = True
    include_liquidity: bool = True

    @field_validator("lag_windows")
    @classmethod
    def validate_lag_windows(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(set(values)) != len(values):
            message = "lag_windows must not contain duplicate entries."
            raise ValueError(message)
        if 1 not in values:
            message = "lag_windows must include 1 to support the mandatory no-change benchmark."
            raise ValueError(message)
        return values


class WalkforwardConfig(BaseModel):
    """Blocked walk-forward split configuration."""

    model_config = ConfigDict(extra="forbid")

    train_size: PositiveInt
    validation_size: PositiveInt
    test_size: PositiveInt
    step_size: PositiveInt
    expanding_train: bool = True


class StatsTestConfig(BaseModel):
    """Official statistical-evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    loss_metrics: tuple[str, ...] = (
        "observed_mse_total_variance",
        "observed_qlike_total_variance",
    )
    benchmark_model: str = "no_change"
    dm_alternative: Literal["two-sided", "greater", "less"] = "greater"
    dm_max_lag: int = Field(default=0, ge=0)
    spa_block_size: PositiveInt = 5
    spa_bootstrap_reps: PositiveInt = 500
    spa_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    mcs_block_size: PositiveInt = 5
    mcs_bootstrap_reps: PositiveInt = 500
    mcs_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    bootstrap_seed: int = 7
    full_grid_weighting: Literal["uniform"] = "uniform"

    @field_validator("loss_metrics")
    @classmethod
    def validate_loss_metrics(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            message = "loss_metrics must contain at least one daily loss metric."
            raise ValueError(message)
        if len(set(values)) != len(values):
            message = "loss_metrics must not contain duplicate entries."
            raise ValueError(message)
        for value in values:
            if value.startswith("mean_") or value.startswith("std_"):
                message = (
                    "loss_metrics must name base daily loss metrics, "
                    f"found summary-like value {value!r}."
                )
                raise ValueError(message)
        return values


class EvaluationMetricsConfig(BaseModel):
    """Shared evaluation metric configuration."""

    model_config = ConfigDict(extra="forbid")

    positive_floor: float = Field(gt=0.0)
    primary_loss_metric: str

    @field_validator("primary_loss_metric")
    @classmethod
    def validate_primary_loss_metric(cls, value: str) -> str:
        if value.startswith("mean_") or value.startswith("std_"):
            message = (
                "primary_loss_metric must name the base daily loss metric, "
                "not a summary column prefix."
            )
            raise ValueError(message)
        return value


class StressWindowConfig(BaseModel):
    """Named stress subperiod for slice reporting."""

    model_config = ConfigDict(extra="forbid")

    label: str
    start_date: date
    end_date: date

    @field_validator("end_date")
    @classmethod
    def validate_order(cls, value: date, info: Any) -> date:
        start_date = info.data.get("start_date")
        if isinstance(start_date, date) and value < start_date:
            message = "Stress window end_date must be on or after start_date."
            raise ValueError(message)
        return value


class ReportArtifactsConfig(BaseModel):
    """Saved-artifact-only report generation configuration."""

    model_config = ConfigDict(extra="forbid")

    benchmark_model: str = "no_change"
    official_loss_metrics: tuple[str, ...] = (
        "observed_mse_total_variance",
        "observed_qlike_total_variance",
    )
    primary_loss_metric: str = "observed_mse_total_variance"
    interpolation_comparison_order: tuple[str, ...] = ("moneyness", "maturity")
    interpolation_cycles: PositiveInt | None = None
    top_models_per_figure: PositiveInt = 6
    stress_windows: tuple[StressWindowConfig, ...] = Field(
        default_factory=lambda: (
            StressWindowConfig(
                label="gfc_2008_2009",
                start_date=date(2008, 9, 1),
                end_date=date(2009, 6, 30),
            ),
            StressWindowConfig(
                label="volmageddon_2018",
                start_date=date(2018, 2, 1),
                end_date=date(2018, 2, 28),
            ),
            StressWindowConfig(
                label="covid_2020",
                start_date=date(2020, 2, 15),
                end_date=date(2020, 5, 29),
            ),
        )
    )

    @field_validator("primary_loss_metric")
    @classmethod
    def validate_primary_loss_metric(cls, value: str) -> str:
        if value.startswith("mean_") or value.startswith("std_"):
            message = (
                "primary_loss_metric must name the base daily loss metric, "
                "not a summary column prefix."
            )
            raise ValueError(message)
        return value

    @field_validator("official_loss_metrics")
    @classmethod
    def validate_official_loss_metrics(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            message = "official_loss_metrics must contain at least one loss metric."
            raise ValueError(message)
        if len(set(values)) != len(values):
            message = "official_loss_metrics must not contain duplicate entries."
            raise ValueError(message)
        for value in values:
            if value.startswith("mean_") or value.startswith("std_"):
                message = (
                    "official_loss_metrics must name base daily loss metrics, "
                    f"found summary-like value {value!r}."
                )
                raise ValueError(message)
        return values

    @field_validator("primary_loss_metric")
    @classmethod
    def validate_primary_loss_metric_in_official_metrics(cls, value: str, info: Any) -> str:
        official_metrics = info.data.get("official_loss_metrics")
        if isinstance(official_metrics, tuple) and value not in official_metrics:
            message = (
                "primary_loss_metric must be included in official_loss_metrics, "
                f"found {value!r} not in {official_metrics!r}."
            )
            raise ValueError(message)
        return value

    @field_validator("interpolation_comparison_order")
    @classmethod
    def validate_interpolation_comparison_order(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        allowed = {"maturity", "moneyness"}
        if set(values) != allowed:
            message = (
                "interpolation_comparison_order must contain exactly "
                "'maturity' and 'moneyness'"
            )
            raise ValueError(message)
        return values


class HedgingConfig(BaseModel):
    """Explicit hedging-evaluation controls."""

    model_config = ConfigDict(extra="forbid")

    risk_free_rate: float = 0.0
    level_notional: float = 1.0
    skew_notional: float = 1.0
    calendar_notional: float = 0.5
    skew_moneyness_abs: float = 0.10
    short_maturity_days: PositiveInt = 30
    long_maturity_days: PositiveInt = 90
    hedge_maturity_days: PositiveInt = 30
    hedge_straddle_moneyness: float = 0.0


class OptunaPrunerConfig(BaseModel):
    """Optuna pruning configuration."""

    model_config = ConfigDict(extra="forbid")

    name: Literal["median"] = "median"
    n_startup_trials: int = Field(default=5, ge=0)
    n_warmup_steps: int = Field(default=1, ge=0)
    interval_steps: PositiveInt = 1


class HpoProfileConfig(BaseModel):
    """Official Optuna profile used by stage 05."""

    model_config = ConfigDict(extra="forbid")

    profile_name: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    n_trials: PositiveInt
    tuning_splits_count: PositiveInt = 3
    seed: int = 7
    sampler: Literal["tpe"] = "tpe"
    pruner: OptunaPrunerConfig = Field(default_factory=OptunaPrunerConfig)


class TrainingProfileConfig(BaseModel):
    """Official walk-forward training profile."""

    model_config = ConfigDict(extra="forbid")

    profile_name: str = Field(pattern=r"^[A-Za-z0-9_]+$")
    epochs: PositiveInt
    neural_early_stopping_patience: PositiveInt = 10
    neural_early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    neural_min_epochs_before_early_stop: PositiveInt = 1
    lightgbm_early_stopping_rounds: PositiveInt = 25
    lightgbm_early_stopping_min_delta: float = Field(default=0.0, ge=0.0)
    lightgbm_first_metric_only: bool = True


class NeuralModelConfig(BaseModel):
    """Torch training parameters."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = "neural_surface"
    hidden_width: PositiveInt = 256
    depth: PositiveInt = 3
    dropout: float = Field(default=0.10, ge=0.0, lt=1.0)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    weight_decay: float = Field(default=1.0e-4, ge=0.0)
    epochs: PositiveInt = 80
    batch_size: PositiveInt = 64
    seed: int = 7
    observed_loss_weight: float = Field(default=1.0, gt=0.0)
    imputed_loss_weight: float = Field(default=0.25, ge=0.0)
    calendar_penalty_weight: float = Field(default=0.05, ge=0.0)
    convexity_penalty_weight: float = Field(default=0.05, ge=0.0)
    roughness_penalty_weight: float = Field(default=0.005, ge=0.0)
    device: str = "cuda"


def calendar_config_from_raw(raw_config: RawDataConfig) -> MarketCalendarConfig:
    """Project the shared calendar fields from the raw-data config."""

    return MarketCalendarConfig(
        calendar_name=raw_config.calendar_name,
        timezone=raw_config.timezone,
        decision_time=raw_config.decision_time,
        decision_snapshot_minutes_before_close=raw_config.decision_snapshot_minutes_before_close,
        am_settled_roots=raw_config.am_settled_roots,
    )


def load_yaml_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        message = f"Config at {config_path} must deserialize to a mapping."
        raise ValueError(message)
    return payload


def parse_config(model_type: type[BaseModel], path: Path | str) -> BaseModel:
    """Parse a YAML file into a typed config model."""

    payload = load_yaml_config(path)
    return model_type.model_validate(payload)

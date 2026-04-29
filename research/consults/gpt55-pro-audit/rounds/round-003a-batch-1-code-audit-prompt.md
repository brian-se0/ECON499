# Round 003A Prompt: Batch 1 Code Audit Packet

You are GPT 5.5 Pro acting as the literature-grounded audit brain for the ECON499 SPX implied-volatility-surface forecasting project. Codex is the implementation agent.

Use the Batch 1 literature standard you just synthesized. Audit only the code/config/tests pasted below. Do not reopen the literature archives, do not browse GitHub, and do not ask Codex to upload a zip unless the pasted context is genuinely insufficient.

Important repo-layout notes:
- There is no `docs/` directory in this checkout.
- Pro requested `scripts/02_qc_raw.py`; nearest actual stage is `scripts/02_build_option_panel.py`.
- Pro requested `scripts/07_evaluate.py`; nearest actual stage is `scripts/07_run_stats.py`.

Audit scope for this round:
- 15:45 field availability and same-day EOD leakage prevention.
- Raw ingestion, schema strictness, explicit filtering/drop reasons, timestamp preservation.
- Surface coordinates, grid metadata, observed masks, completed-grid provenance, interpolation/smoothing versioning.
- Observed-vs-completed metrics and moneyness/maturity slice diagnostics.
- No-change and factor benchmark coordinate/split safety.
- Neural model target/mask handling and "arbitrage-aware" versus "arbitrage-free" wording.

Return exactly these sections:

## FORMAL_FINDINGS
List only concrete, actionable findings supported by pasted code. For each finding include:
- ID: B1-CODE-###
- Severity: P0/P1/P2/P3
- File and line number(s)
- Evidence from code
- Why it violates the Batch 1 standard
- Minimal fix direction
- Required tests

If there are no findings, write `none`.

## CONTEXT_GAPS
List any missing file or artifact that prevents a confident finding. Do not turn a context gap into a finding.

## FIX_ORDER
If findings exist, give the implementation order. If none, write `none`.

## TEST_COMMANDS_TO_RUN
Suggest focused local test commands Codex should run after fixes.

## SOURCE_FILES


### src/ivsurf/config.py

```python
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
            message = "lag_windows must include 1 to support the mandatory naive benchmark."
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
    benchmark_model: str = "naive"
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

    benchmark_model: str = "naive"
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
    output_total_variance_floor: float = Field(default=1.0e-8, gt=0.0)
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

```

### src/ivsurf/schemas.py

```python
"""Explicit schemas for raw and derived data."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl
import pyarrow as pa

from ivsurf.exceptions import SchemaDriftError

RAW_COLUMNS: tuple[str, ...] = (
    "underlying_symbol",
    "quote_date",
    "root",
    "expiration",
    "strike",
    "option_type",
    "trade_volume",
    "bid_size_1545",
    "bid_1545",
    "ask_size_1545",
    "ask_1545",
    "underlying_bid_1545",
    "underlying_ask_1545",
    "active_underlying_price_1545",
    "implied_volatility_1545",
    "delta_1545",
    "gamma_1545",
    "theta_1545",
    "vega_1545",
    "rho_1545",
    "open_interest",
)

RAW_ALLOWED_EXTRA_COLUMNS: tuple[str, ...] = (
    "implied_underlying_price_1545",
    "open",
    "high",
    "low",
    "close",
    "bid_size_eod",
    "bid_eod",
    "ask_size_eod",
    "ask_eod",
    "underlying_bid_eod",
    "underlying_ask_eod",
    "vwap",
    "delivery_code",
)

RAW_POLARS_SCHEMA: dict[str, object] = {
    "underlying_symbol": pl.String,
    "quote_date": pl.String,
    "root": pl.String,
    "expiration": pl.String,
    "strike": pl.Float64,
    "option_type": pl.String,
    "trade_volume": pl.Int64,
    "bid_size_1545": pl.Int64,
    "bid_1545": pl.Float64,
    "ask_size_1545": pl.Int64,
    "ask_1545": pl.Float64,
    "underlying_bid_1545": pl.Float64,
    "underlying_ask_1545": pl.Float64,
    "active_underlying_price_1545": pl.Float64,
    "implied_volatility_1545": pl.Float64,
    "delta_1545": pl.Float64,
    "gamma_1545": pl.Float64,
    "theta_1545": pl.Float64,
    "vega_1545": pl.Float64,
    "rho_1545": pl.Float64,
    "open_interest": pl.Int64,
}

RAW_ARROW_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("underlying_symbol", pa.string(), nullable=False),
        pa.field("quote_date", pa.date32(), nullable=False),
        pa.field("root", pa.string(), nullable=False),
        pa.field("expiration", pa.date32(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),
        pa.field("trade_volume", pa.int64(), nullable=False),
        pa.field("bid_size_1545", pa.int64(), nullable=False),
        pa.field("bid_1545", pa.float64(), nullable=False),
        pa.field("ask_size_1545", pa.int64(), nullable=False),
        pa.field("ask_1545", pa.float64(), nullable=False),
        pa.field("underlying_bid_1545", pa.float64(), nullable=False),
        pa.field("underlying_ask_1545", pa.float64(), nullable=False),
        pa.field("active_underlying_price_1545", pa.float64(), nullable=False),
        pa.field("implied_volatility_1545", pa.float64(), nullable=False),
        pa.field("delta_1545", pa.float64(), nullable=False),
        pa.field("gamma_1545", pa.float64(), nullable=False),
        pa.field("theta_1545", pa.float64(), nullable=False),
        pa.field("vega_1545", pa.float64(), nullable=False),
        pa.field("rho_1545", pa.float64(), nullable=False),
        pa.field("open_interest", pa.int64(), nullable=True),
        pa.field("source_zip", pa.string(), nullable=False),
    ]
)


def validate_raw_columns(columns: Iterable[str]) -> None:
    """Fail fast on raw schema drift."""

    actual = tuple(columns)
    missing = [name for name in RAW_COLUMNS if name not in actual]
    unexpected = [
        name
        for name in actual
        if name not in RAW_COLUMNS and name not in RAW_ALLOWED_EXTRA_COLUMNS
    ]
    if missing or unexpected:
        message = (
            "Raw schema drift detected. "
            f"missing={missing if missing else '[]'} "
            f"unexpected={unexpected if unexpected else '[]'}"
        )
        raise SchemaDriftError(message)

```

### src/ivsurf/calendar.py

```python
"""Trading-calendar and maturity helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from ivsurf.exceptions import TemporalIntegrityError


@dataclass(slots=True)
class MarketCalendar:
    """Explicit market calendar wrapper for session alignment and maturity timing."""

    calendar_name: str = "XNYS"
    timezone: str = "America/New_York"
    decision_time: time = time(15, 45)
    decision_snapshot_minutes_before_close: int = 15
    am_settled_roots: tuple[str, ...] = ("SPX",)
    _calendar: Any = field(init=False, repr=False, default=None)
    _calendar_start: date | None = field(init=False, repr=False, default=None)
    _calendar_end: date | None = field(init=False, repr=False, default=None)
    _calendar_padding_days: int = field(default=14, init=False, repr=False)

    def __post_init__(self) -> None:
        return None

    def _rebuild_calendar(self, start: date, end: date) -> None:
        self._calendar = xcals.get_calendar(self.calendar_name, start=start, end=end)
        self._calendar_start = cast(date, self._calendar.first_session.date())
        self._calendar_end = cast(date, self._calendar.last_session.date())

    def _ensure_calendar_bounds(self, *session_dates: date) -> None:
        if not session_dates:
            message = "_ensure_calendar_bounds requires at least one session date."
            raise ValueError(message)

        requested_start = min(session_dates) - timedelta(days=self._calendar_padding_days)
        requested_end = max(session_dates) + timedelta(days=self._calendar_padding_days)
        if self._calendar_start is None or self._calendar_end is None or self._calendar is None:
            self._rebuild_calendar(start=requested_start, end=requested_end)
            return
        if requested_start < self._calendar_start or requested_end > self._calendar_end:
            self._rebuild_calendar(
                start=min(requested_start, self._calendar_start),
                end=max(requested_end, self._calendar_end),
            )

    def _to_session_label(self, session_date: date) -> pd.Timestamp:
        return pd.Timestamp(session_date)

    def is_session(self, session_date: date) -> bool:
        self._ensure_calendar_bounds(session_date)
        return bool(self._calendar.is_session(self._to_session_label(session_date)))

    def previous_session(self, session_date: date) -> date:
        self._ensure_calendar_bounds(session_date)
        label = self._to_session_label(session_date)
        if self.is_session(session_date):
            previous = self._calendar.previous_session(label)
        else:
            previous = self._calendar.date_to_session(label, direction="previous")
        return cast(date, previous.date())

    def next_session(self, session_date: date) -> date:
        self._ensure_calendar_bounds(session_date)
        label = self._to_session_label(session_date)
        next_value = self._calendar.next_session(label)
        return cast(date, next_value.date())

    def next_decision_session(self, session_date: date) -> date:
        """Return the next observed trading session after the provided date."""

        return self.next_session(session_date)

    def _session_close_local(self, session_date: date) -> pd.Timestamp:
        self._ensure_calendar_bounds(session_date)
        if not self.is_session(session_date):
            message = f"{session_date.isoformat()} is not a trading session."
            raise TemporalIntegrityError(message)
        return self._calendar.session_close(self._to_session_label(session_date)).tz_convert(
            self.timezone
        )

    def effective_decision_datetime(self, session_date: date) -> datetime:
        """Return the effective vendor decision snapshot timestamp for one session."""

        close_local = self._session_close_local(session_date)
        configured_decision_dt = datetime.combine(
            session_date,
            self.decision_time,
            tzinfo=ZoneInfo(self.timezone),
        )
        close_buffer_dt = (
            close_local - timedelta(minutes=self.decision_snapshot_minutes_before_close)
        ).to_pydatetime()
        return min(configured_decision_dt, cast(datetime, close_buffer_dt))

    def session_has_decision_time(self, session_date: date) -> bool:
        """Return whether the session carries a usable vendor decision snapshot."""

        self._ensure_calendar_bounds(session_date)
        if not self.is_session(session_date):
            return False
        close_local = self._session_close_local(session_date)
        return bool(close_local.to_pydatetime() >= self.effective_decision_datetime(session_date))

    def resolve_last_tradable_session(self, root: str, expiration: date) -> date:
        """Resolve the session on which a contract can last trade."""

        self._ensure_calendar_bounds(expiration)
        settlement_session = (
            expiration if self.is_session(expiration) else self.previous_session(expiration)
        )
        if root in self.am_settled_roots:
            return self.previous_session(settlement_session)
        return settlement_session

    def compute_tau_years(self, quote_date: date, expiration: date, root: str) -> float:
        """Compute ACT/365 time-to-maturity from the effective snapshot to last tradable close."""

        self._ensure_calendar_bounds(quote_date, expiration)
        if not self.session_has_decision_time(quote_date):
            message = (
                "Session "
                f"{quote_date.isoformat()} does not contain a usable decision snapshot."
            )
            raise TemporalIntegrityError(message)

        last_tradable_session = self.resolve_last_tradable_session(root=root, expiration=expiration)
        if quote_date > last_tradable_session:
            return 0.0

        decision_dt = self.effective_decision_datetime(quote_date)
        expiry_close_ts = self._calendar.session_close(
            self._to_session_label(last_tradable_session)
        )
        expiry_close = expiry_close_ts.tz_convert(self.timezone).to_pydatetime()
        delta_seconds = (expiry_close - decision_dt).total_seconds()
        tau_years = max(delta_seconds, 0.0) / (365.0 * 24.0 * 60.0 * 60.0)
        return cast(float, tau_years)

    def next_trading_session(self, session_date: date) -> date:
        """Return the next trading session after the provided session date."""

        return self.next_session(session_date)

```

### src/ivsurf/io/ingest_cboe.py

```python
"""Raw Cboe daily-zip ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from zipfile import ZipFile

import polars as pl

from ivsurf.config import RawDataConfig
from ivsurf.exceptions import DataValidationError, SchemaDriftError
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.qc.raw_checks import assert_single_quote_date, assert_target_symbol_only
from ivsurf.qc.sample_window import quote_date_in_sample_window
from ivsurf.qc.schema_checks import assert_non_null_columns
from ivsurf.schemas import RAW_COLUMNS, RAW_POLARS_SCHEMA, validate_raw_columns


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary for one ingested raw zip."""

    source_zip: Path
    quote_date: date
    bronze_path: Path | None
    row_count: int
    status: str


def list_raw_zip_files(config: RawDataConfig) -> list[Path]:
    """List raw zip files in deterministic order."""

    return sorted(config.raw_options_dir.glob(config.raw_file_glob))


def _extract_single_csv(zip_path: Path, temp_dir: Path) -> Path:
    with ZipFile(zip_path) as archive:
        entries = archive.namelist()
        if len(entries) != 1:
            message = f"{zip_path} must contain exactly one CSV member, found {len(entries)}."
            raise SchemaDriftError(message)
        entry_name = entries[0]
        destination = temp_dir / entry_name
        archive.extract(entry_name, path=temp_dir)
    return destination


def ingest_one_zip(zip_path: Path, config: RawDataConfig) -> IngestionResult:
    """Read one daily zip, filter to SPX, and write partitioned parquet."""

    with TemporaryDirectory() as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        extracted_csv = _extract_single_csv(zip_path, temp_dir)
        header = pl.read_csv(extracted_csv, n_rows=0).columns
        validate_raw_columns(header)

        lazy_frame = pl.scan_csv(
            extracted_csv,
            schema_overrides=cast(dict[str, Any], RAW_POLARS_SCHEMA),
            null_values={"open_interest": ""},
            quote_char='"',
            infer_schema=True,
        )
        frame = (
            lazy_frame.select(RAW_COLUMNS)
            .filter(pl.col("underlying_symbol") == config.target_symbol)
            .with_columns(
                pl.col("quote_date").str.strptime(pl.Date, strict=True),
                pl.col("expiration").str.strptime(pl.Date, strict=True),
                pl.lit(str(zip_path)).alias("source_zip"),
            )
            .collect(engine="streaming")
        )

    if frame.is_empty():
        message = f"No rows for symbol {config.target_symbol} found in {zip_path.name}."
        raise DataValidationError(message)

    assert_target_symbol_only(
        frame,
        symbol_column="underlying_symbol",
        expected_symbol=config.target_symbol,
        dataset_name=zip_path.name,
    )
    assert_single_quote_date(frame, dataset_name=zip_path.name)
    assert_non_null_columns(
        frame,
        columns=("quote_date", "expiration", "root", "strike", "option_type"),
        dataset_name=zip_path.name,
    )

    quote_date = frame["quote_date"][0]
    if not isinstance(quote_date, date):
        message = f"{zip_path.name} quote_date must be parsed as a Polars Date."
        raise TypeError(message)
    if not quote_date_in_sample_window(quote_date, config):
        return IngestionResult(
            source_zip=zip_path,
            quote_date=quote_date,
            bronze_path=None,
            row_count=frame.height,
            status="skipped_out_of_sample_window",
        )

    output_dir = config.bronze_dir / f"year={quote_date.year}"
    output_dir.mkdir(parents=True, exist_ok=True)
    bronze_path = output_dir / f"{zip_path.stem}.parquet"
    write_parquet_frame(frame, bronze_path)
    return IngestionResult(
        source_zip=zip_path,
        quote_date=quote_date,
        bronze_path=bronze_path,
        row_count=frame.height,
        status="written",
    )


def ingest_all(config: RawDataConfig, limit: int | None = None) -> list[IngestionResult]:
    """Ingest all raw zips in deterministic order."""

    zip_paths = list_raw_zip_files(config)
    if limit is not None:
        zip_paths = zip_paths[:limit]
    return [ingest_one_zip(path, config) for path in zip_paths]

```

### src/ivsurf/io/parquet.py

```python
"""Shared parquet IO helpers with explicit defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from ivsurf.io.atomic import cleanup_atomic_temp_files, replace_path_atomic, write_text_atomic


def write_parquet_frame(
    frame: pl.DataFrame,
    output_path: Path,
    *,
    compression: Literal["lz4", "uncompressed", "snappy", "gzip", "brotli", "zstd"] = "zstd",
    statistics: bool = True,
) -> None:
    """Write a parquet artifact with the project's explicit defaults."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_atomic_temp_files(output_path)
    temp_path = output_path.with_name(f"{output_path.name}.write_tmp")
    try:
        if temp_path.exists():
            temp_path.unlink()
        frame.write_parquet(temp_path, compression=compression, statistics=statistics)
        replace_path_atomic(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def write_csv_frame(frame: pl.DataFrame, output_path: Path) -> None:
    """Write a CSV artifact atomically."""

    csv_text = frame.write_csv()
    if csv_text is None:
        message = f"Polars returned no CSV text while writing {output_path}."
        raise ValueError(message)
    write_text_atomic(output_path, csv_text, encoding="utf-8")


def read_parquet_files(paths: list[Path]) -> pl.DataFrame:
    """Read a list of parquet files into one concatenated frame."""

    if not paths:
        message = "read_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return scan_parquet_files(paths).collect(engine="streaming")


def scan_parquet_files(paths: list[Path]) -> pl.LazyFrame:
    """Create one lazy parquet scan from explicit file paths."""

    if not paths:
        message = "scan_parquet_files requires at least one parquet path."
        raise ValueError(message)
    return pl.scan_parquet([str(path) for path in paths])

```

### src/ivsurf/qc/timing_checks.py

```python
"""Temporal-integrity validation helpers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import pairwise

from ivsurf.calendar import MarketCalendar
from ivsurf.exceptions import TemporalIntegrityError


def assert_session_has_decision_time(
    calendar: MarketCalendar,
    session_date: date,
    *,
    context: str,
) -> None:
    """Require a session to include a usable vendor decision snapshot."""

    if not calendar.session_has_decision_time(session_date):
        message = (
            f"{context} requires session {session_date.isoformat()} to contain a usable "
            "vendor decision snapshot."
        )
        raise TemporalIntegrityError(message)


def assert_next_decision_session_alignment(
    calendar: MarketCalendar,
    quote_date: date,
    target_date: date,
) -> None:
    """Require target_date to equal the next observed trading session."""

    expected_target = calendar.next_trading_session(quote_date)
    if expected_target != target_date:
        message = (
            "Feature/target alignment violated next-trading-session causality: "
            f"quote_date={quote_date.isoformat()} "
            f"expected_target_date={expected_target.isoformat()} "
            f"actual_target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)


def assert_strictly_increasing_unique_dates(
    observed_dates: Sequence[date],
    *,
    context: str,
) -> None:
    """Require an observed-date sequence to be unique and strictly increasing."""

    if len(set(observed_dates)) != len(observed_dates):
        message = f"{context} must contain unique dates."
        raise TemporalIntegrityError(message)
    for previous_date, current_date in pairwise(observed_dates):
        if current_date <= previous_date:
            message = (
                f"{context} must be strictly increasing: "
                f"{previous_date.isoformat()} then {current_date.isoformat()}."
            )
            raise TemporalIntegrityError(message)


def assert_next_observed_target_date(
    observed_dates: Sequence[date],
    *,
    position: int,
    quote_date: date,
    target_date: date,
) -> None:
    """Require target_date to equal the next observed gold-surface date."""

    if position < 0 or position >= len(observed_dates) - 1:
        message = (
            "Position for next-observed-date alignment must point to a quote date with a "
            f"subsequent observed target date, found position={position}."
        )
        raise TemporalIntegrityError(message)
    expected_quote_date = observed_dates[position]
    expected_target_date = observed_dates[position + 1]
    if quote_date != expected_quote_date:
        message = (
            "Feature/target alignment quote_date does not match the observed gold-surface "
            f"sequence: expected={expected_quote_date.isoformat()} "
            f"actual={quote_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)
    if target_date <= quote_date:
        message = (
            "Feature/target alignment requires target_date to be after quote_date: "
            f"quote_date={quote_date.isoformat()} "
            f"target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)
    if target_date != expected_target_date:
        message = (
            "Feature/target alignment violated next-observed-date causality: "
            f"quote_date={quote_date.isoformat()} "
            f"expected_target_date={expected_target_date.isoformat()} "
            f"actual_target_date={target_date.isoformat()}."
        )
        raise TemporalIntegrityError(message)

```

### src/ivsurf/qc/schema_checks.py

```python
"""Explicit schema and column-level validation helpers."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from ivsurf.exceptions import DataValidationError, SchemaDriftError


def assert_required_columns(
    columns: Iterable[str],
    *,
    required_columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Fail fast when required columns are absent."""

    actual = tuple(columns)
    required = tuple(required_columns)
    missing = [name for name in required if name not in actual]
    if missing:
        message = f"{dataset_name} is missing required columns: {missing}"
        raise SchemaDriftError(message)


def assert_non_null_columns(
    frame: pl.DataFrame,
    *,
    columns: Iterable[str],
    dataset_name: str,
) -> None:
    """Fail fast when critical columns contain nulls."""

    null_columns = [name for name in columns if frame[name].null_count() > 0]
    if null_columns:
        message = f"{dataset_name} contains nulls in critical columns: {null_columns}"
        raise DataValidationError(message)

```

### src/ivsurf/qc/sample_window.py

```python
"""Explicit thesis sample-window validation helpers."""

from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.config import RawDataConfig
from ivsurf.exceptions import DataValidationError


def sample_window_label(config: RawDataConfig) -> str:
    """Render the configured inclusive sample window."""

    return (
        f"{config.sample_start_date.isoformat()} to {config.sample_end_date.isoformat()}"
    )


def quote_date_in_sample_window(quote_date: date, config: RawDataConfig) -> bool:
    """Return whether a quote date is inside the configured inclusive sample window."""

    return config.sample_start_date <= quote_date <= config.sample_end_date


def require_quote_date_in_sample_window(
    quote_date: date,
    config: RawDataConfig,
    *,
    context: str,
) -> None:
    """Fail fast when one quote date violates the configured thesis sample window."""

    if quote_date_in_sample_window(quote_date, config):
        return
    message = (
        f"{context} quote_date {quote_date.isoformat()} is outside the configured "
        f"sample window {sample_window_label(config)}."
    )
    raise DataValidationError(message)


def sample_window_expr(
    config: RawDataConfig,
    *,
    column: str = "quote_date",
) -> pl.Expr:
    """Return a Polars expression for the configured inclusive sample window."""

    return (pl.col(column) >= pl.lit(config.sample_start_date)) & (
        pl.col(column) <= pl.lit(config.sample_end_date)
    )


def assert_frame_dates_in_sample_window(
    frame: pl.DataFrame,
    config: RawDataConfig,
    *,
    column: str = "quote_date",
    context: str,
) -> None:
    """Require all distinct dates in one frame to stay inside the configured sample window."""

    out_of_window = (
        frame.select(column)
        .unique()
        .filter(~sample_window_expr(config, column=column))
        .sort(column)
    )
    if out_of_window.is_empty():
        return
    offending_dates = ", ".join(
        value.isoformat() for value in out_of_window[column].to_list() if isinstance(value, date)
    )
    message = (
        f"{context} contains out-of-window {column} values: {offending_dates}. "
        f"Configured sample window: {sample_window_label(config)}."
    )
    raise DataValidationError(message)

```

### src/ivsurf/cleaning/option_filters.py

```python
"""Explicit option-level cleaning rules and reason codes."""

from __future__ import annotations

import polars as pl

from ivsurf.config import CleaningConfig


def apply_option_quality_flags(frame: pl.DataFrame, config: CleaningConfig) -> pl.DataFrame:
    """Flag invalid rows with explicit reason codes instead of silently dropping them."""

    invalid_reason = (
        pl.when(~pl.col("option_type").is_in(config.allowed_option_types))
        .then(pl.lit("INVALID_OPTION_TYPE"))
        .when(pl.col("bid_1545") <= config.min_valid_bid_exclusive)
        .then(pl.lit("NON_POSITIVE_BID"))
        .when(pl.col("ask_1545") <= config.min_valid_ask_exclusive)
        .then(pl.lit("NON_POSITIVE_ASK"))
        .when(config.require_ask_ge_bid & (pl.col("ask_1545") < pl.col("bid_1545")))
        .then(pl.lit("ASK_LT_BID"))
        .when(config.require_positive_iv & (pl.col("implied_volatility_1545") <= 0.0))
        .then(pl.lit("NON_POSITIVE_IV"))
        .when(config.require_positive_vega & (pl.col("vega_1545") <= 0.0))
        .then(pl.lit("NON_POSITIVE_VEGA"))
        .when(
            config.require_positive_underlying_price
            & (pl.col("active_underlying_price_1545") <= 0.0)
        )
        .then(pl.lit("NON_POSITIVE_UNDERLYING_PRICE"))
        .when(pl.col("mid_1545") <= config.min_valid_mid_price_exclusive)
        .then(pl.lit("NON_POSITIVE_MID"))
        .when(pl.col("tau_years") < config.min_tau_years)
        .then(pl.lit("TAU_TOO_SHORT"))
        .when(pl.col("tau_years") > config.max_tau_years)
        .then(pl.lit("TAU_TOO_LONG"))
        .when(pl.col("log_moneyness").abs() > config.max_abs_log_moneyness)
        .then(pl.lit("OUTSIDE_MONEYNESS_RANGE"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )

    return (
        frame.with_columns(invalid_reason.alias("invalid_reason"))
        .with_columns(pl.col("invalid_reason").is_null().alias("is_valid_observation"))
    )


def valid_option_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Return only valid rows."""

    return frame.filter(pl.col("is_valid_observation"))

```

### src/ivsurf/cleaning/derived_fields.py

```python
"""Derived-field construction for decision-snapshot SPX option rows."""

from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.calendar import MarketCalendar
from ivsurf.config import MarketCalendarConfig


def build_tau_lookup(
    frame: pl.DataFrame,
    calendar_config: MarketCalendarConfig,
) -> pl.DataFrame:
    """Build a per-root and per-expiration tau lookup for one quote date."""

    quote_dates = frame.select(pl.col("quote_date").unique()).to_series().to_list()
    if len(quote_dates) != 1:
        message = "build_tau_lookup expects a single quote_date per frame."
        raise ValueError(message)
    quote_date = quote_dates[0]
    if not isinstance(quote_date, date):
        message = "quote_date must be parsed as a Polars Date before tau construction."
        raise TypeError(message)

    market_calendar = MarketCalendar(
        calendar_name=calendar_config.calendar_name,
        timezone=calendar_config.timezone,
        decision_time=calendar_config.decision_time,
        decision_snapshot_minutes_before_close=calendar_config.decision_snapshot_minutes_before_close,
        am_settled_roots=calendar_config.am_settled_roots,
    )

    keys = frame.select("root", "expiration").unique().sort(["root", "expiration"])
    rows: list[dict[str, object]] = []
    for row in keys.iter_rows(named=True):
        expiration = row["expiration"]
        root = row["root"]
        if not isinstance(expiration, date) or not isinstance(root, str):
            message = "root/expiration lookup contains invalid types."
            raise TypeError(message)
        tau_years = market_calendar.compute_tau_years(
            quote_date=quote_date,
            expiration=expiration,
            root=root,
        )
        rows.append({"root": root, "expiration": expiration, "tau_years": tau_years})
    return pl.DataFrame(
        rows,
        schema={"root": pl.String, "expiration": pl.Date, "tau_years": pl.Float64},
    )


def add_derived_fields(frame: pl.DataFrame, tau_lookup: pl.DataFrame) -> pl.DataFrame:
    """Join maturity and compute option-level derived fields."""

    enriched = frame.join(tau_lookup, on=["root", "expiration"], how="left", validate="m:1")
    return enriched.with_columns(
        ((pl.col("bid_1545") + pl.col("ask_1545")) / 2.0).alias("mid_1545"),
        (pl.col("ask_1545") - pl.col("bid_1545")).alias("spread_1545"),
        (
            pl.col("strike").log() - pl.col("active_underlying_price_1545").log()
        ).alias("log_moneyness"),
        ((pl.col("implied_volatility_1545") ** 2.0) * pl.col("tau_years")).alias("total_variance"),
    )

```

### src/ivsurf/surfaces/grid.py

```python
"""Fixed-grid helpers for daily surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig


@dataclass(frozen=True, slots=True)
class SurfaceGrid:
    """Fixed maturity x moneyness grid."""

    maturity_days: tuple[int, ...]
    moneyness_points: tuple[float, ...]

    @property
    def maturity_years(self) -> np.ndarray:
        return np.asarray(self.maturity_days, dtype=np.float64) / 365.0

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.maturity_days), len(self.moneyness_points))

    @classmethod
    def from_config(cls, config: SurfaceGridConfig) -> SurfaceGrid:
        return cls(maturity_days=config.maturity_days, moneyness_points=config.moneyness_points)


def assign_grid_indices(frame: pl.DataFrame, grid: SurfaceGrid) -> pl.DataFrame:
    """Assign each option row to the nearest fixed grid point."""

    maturity_edges = np.empty(len(grid.maturity_days) + 1, dtype=np.float64)
    maturity_years = grid.maturity_years
    maturity_edges[0] = -np.inf
    maturity_edges[-1] = np.inf
    maturity_edges[1:-1] = (maturity_years[:-1] + maturity_years[1:]) / 2.0

    moneyness_edges = np.empty(len(grid.moneyness_points) + 1, dtype=np.float64)
    money = np.asarray(grid.moneyness_points, dtype=np.float64)
    moneyness_edges[0] = -np.inf
    moneyness_edges[-1] = np.inf
    moneyness_edges[1:-1] = (money[:-1] + money[1:]) / 2.0

    maturity_index = np.searchsorted(
        maturity_edges[1:-1],
        frame["tau_years"].to_numpy(),
        side="right",
    ).astype(np.int64)
    moneyness_index = np.searchsorted(
        moneyness_edges[1:-1],
        frame["log_moneyness"].to_numpy(),
        side="right",
    ).astype(np.int64)

    return frame.with_columns(
        pl.Series("maturity_index", maturity_index),
        pl.Series("moneyness_index", moneyness_index),
    )

```

### src/ivsurf/surfaces/aggregation.py

```python
"""Vega-weighted daily surface aggregation."""

from __future__ import annotations

import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.grid import SurfaceGrid


def aggregate_daily_surface(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    config: SurfaceGridConfig,
) -> pl.DataFrame:
    """Aggregate valid option rows to daily observed grid cells."""

    grouped = (
        frame.group_by(["quote_date", "maturity_index", "moneyness_index"])
        .agg(
            (
                (pl.col("total_variance") * pl.col("vega_1545")).sum() / pl.col("vega_1545").sum()
            ).alias("observed_total_variance"),
            (
                (pl.col("implied_volatility_1545") * pl.col("vega_1545")).sum()
                / pl.col("vega_1545").sum()
            ).alias("observed_iv"),
            (pl.col("spread_1545") * pl.col("vega_1545")).sum().alias("vega_weighted_spread_sum"),
            pl.col("vega_1545").sum().alias("vega_sum"),
            pl.len().alias("observation_count"),
        )
        .with_columns(
            (pl.col("vega_weighted_spread_sum") / pl.col("vega_sum")).alias("weighted_spread_1545"),
            (pl.col("observation_count") >= config.observed_cell_min_count).alias("observed_mask"),
        )
        .drop("vega_weighted_spread_sum")
    )

    date_values = grouped.select(pl.col("quote_date").unique()).to_series().to_list()
    rows: list[dict[str, object]] = []
    for quote_date in date_values:
        for maturity_index, maturity_day in enumerate(grid.maturity_days):
            for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_day,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                    }
                )
    dense_grid = pl.DataFrame(rows)
    return dense_grid.join(
        grouped,
        on=["quote_date", "maturity_index", "moneyness_index"],
        how="left",
        validate="m:1",
    ).with_columns(
        pl.col("observed_mask").fill_null(False),
        pl.col("observation_count").fill_null(0),
        pl.col("vega_sum").fill_null(0.0),
        pl.col("weighted_spread_1545").fill_null(0.0),
    )

```

### src/ivsurf/surfaces/interpolation.py

```python
"""Deterministic sequential axis-wise surface completion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from ivsurf.exceptions import InterpolationError


@dataclass(frozen=True, slots=True)
class CompletedSurface:
    """Completed daily surface with mask information."""

    completed_total_variance: np.ndarray
    observed_mask: np.ndarray


def _fill_axis(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    result = values.copy()
    finite_mask = np.isfinite(values)
    count = int(finite_mask.sum())
    if count == 0:
        return result
    if count == 1:
        result[~finite_mask] = values[finite_mask][0]
        return result

    observed_x = coordinates[finite_mask]
    observed_y = values[finite_mask]
    interpolator = PchipInterpolator(observed_x, observed_y, extrapolate=False)
    missing_positions = np.flatnonzero(~finite_mask)
    if missing_positions.size == 0:
        return result

    target_x = coordinates[missing_positions]
    predicted = interpolator(target_x)
    predicted = np.where(
        target_x < observed_x.min(),
        observed_y[0],
        np.where(target_x > observed_x.max(), observed_y[-1], predicted),
    )
    result[missing_positions] = predicted
    return result


def complete_surface(
    observed_total_variance: np.ndarray,
    observed_mask: np.ndarray,
    maturity_coordinates: np.ndarray,
    moneyness_coordinates: np.ndarray,
    interpolation_order: tuple[str, ...],
    interpolation_cycles: int,
    total_variance_floor: float,
) -> CompletedSurface:
    """Complete a surface by fixed-order sequential one-dimensional interpolation."""

    completed = observed_total_variance.astype(np.float64, copy=True)
    normalized_observed_mask = np.asarray(observed_mask, dtype=bool)
    if normalized_observed_mask.shape != completed.shape:
        message = (
            "observed_mask must have the same shape as observed_total_variance, "
            f"found {normalized_observed_mask.shape} != {completed.shape}."
        )
        raise ValueError(message)

    for _ in range(interpolation_cycles):
        for axis_name in interpolation_order:
            if axis_name == "maturity":
                for money_idx in range(completed.shape[1]):
                    completed[:, money_idx] = _fill_axis(
                        completed[:, money_idx],
                        maturity_coordinates,
                    )
            elif axis_name == "moneyness":
                for maturity_idx in range(completed.shape[0]):
                    completed[maturity_idx, :] = _fill_axis(
                        completed[maturity_idx, :],
                        moneyness_coordinates,
                    )
            else:
                message = f"Unsupported interpolation axis: {axis_name}"
                raise ValueError(message)

    if not np.isfinite(completed).all():
        message = (
            "Surface completion left NaN or infinite values "
            "after deterministic interpolation."
        )
        raise InterpolationError(message)

    completed = np.maximum(completed, total_variance_floor)
    return CompletedSurface(
        completed_total_variance=completed,
        observed_mask=normalized_observed_mask,
    )

```

### src/ivsurf/surfaces/masks.py

```python
"""Helpers for reshaping long-form surface artifacts into dense grids."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid


def _ordered_surface_frame(frame: pl.DataFrame, grid: SurfaceGrid) -> pl.DataFrame:
    ordered = frame.sort(["maturity_index", "moneyness_index"])
    expected_rows = grid.shape[0] * grid.shape[1]
    if ordered.height != expected_rows:
        message = (
            f"Expected exactly {expected_rows} rows for one dense surface, "
            f"found {ordered.height}."
        )
        raise ValueError(message)
    return ordered


def reshape_surface_column(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    column_name: str,
    *,
    null_fill: float | None = None,
) -> np.ndarray:
    """Reshape one long-form numeric surface column into the fixed grid."""

    ordered = _ordered_surface_frame(frame, grid)
    series = ordered[column_name]
    if series.null_count() > 0:
        if null_fill is None:
            message = f"Column {column_name} contains nulls but no null_fill was provided."
            raise ValueError(message)
        values = series.fill_null(null_fill).to_numpy()
    else:
        values = series.to_numpy()
    return np.asarray(values, dtype=np.float64).reshape(grid.shape)


def reshape_mask_column(frame: pl.DataFrame, grid: SurfaceGrid, column_name: str) -> np.ndarray:
    """Reshape one long-form boolean surface column into the fixed grid."""

    ordered = _ordered_surface_frame(frame, grid)
    values = ordered[column_name].to_numpy()
    return np.asarray(values, dtype=bool).reshape(grid.shape)


def dense_surface_rows(
    frame: pl.DataFrame,
    grid: SurfaceGrid,
    *,
    column_name: str,
    null_fill: float | None = None,
) -> list[dict[str, Any]]:
    """Return a dense surface column as row dictionaries for debugging or reporting."""

    surface = reshape_surface_column(frame, grid, column_name, null_fill=null_fill)
    rows: list[dict[str, Any]] = []
    for maturity_index, maturity_days in enumerate(grid.maturity_days):
        for moneyness_index, moneyness_point in enumerate(grid.moneyness_points):
            rows.append(
                {
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    column_name: float(surface[maturity_index, moneyness_index]),
                }
            )
    return rows

```

### src/ivsurf/surfaces/arbitrage_diagnostics.py

```python
"""Surface-shape diagnostics for calendar monotonicity and butterfly convexity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass(frozen=True, slots=True)
class ArbitrageDiagnostics:
    """Violation counts and magnitudes."""

    calendar_violation_count: int
    calendar_violation_magnitude: float
    convexity_violation_count: int
    convexity_violation_magnitude: float


def calendar_monotonicity_violations(surface: np.ndarray) -> tuple[int, float]:
    """Count decreases across maturity."""

    diffs = np.diff(surface, axis=0)
    violations = np.minimum(diffs, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def _validate_coordinate_vector(
    coordinates: np.ndarray,
    *,
    expected_size: int,
    coordinate_name: str,
) -> np.ndarray:
    vector = np.asarray(coordinates, dtype=np.float64)
    if vector.ndim != 1:
        message = f"{coordinate_name} must be one-dimensional."
        raise ValueError(message)
    if vector.shape[0] != expected_size:
        message = (
            f"{coordinate_name} length {vector.shape[0]} does not match expected size "
            f"{expected_size}."
        )
        raise ValueError(message)
    if not np.isfinite(vector).all():
        message = f"{coordinate_name} must contain only finite values."
        raise ValueError(message)
    if not np.all(np.diff(vector) > 0.0):
        message = f"{coordinate_name} must be strictly increasing."
        raise ValueError(message)
    return vector


def _validate_positive_total_variance(surface: np.ndarray) -> np.ndarray:
    values = np.asarray(surface, dtype=np.float64)
    if values.ndim != 2:
        message = "surface must be a two-dimensional maturity x moneyness array."
        raise ValueError(message)
    if not np.isfinite(values).all():
        message = "surface must contain only finite total-variance values."
        raise ValueError(message)
    if not np.all(values > 0.0):
        message = "surface total variance must be strictly positive for price diagnostics."
        raise ValueError(message)
    return values


def _normalized_call_prices(
    total_variance: np.ndarray,
    *,
    log_moneyness: np.ndarray,
) -> np.ndarray:
    strikes = np.exp(log_moneyness)
    sqrt_variance = np.sqrt(total_variance)
    d1 = ((-log_moneyness)[None, :] + (0.5 * total_variance)) / sqrt_variance
    d2 = d1 - sqrt_variance
    return ndtr(d1) - (strikes[None, :] * ndtr(d2))


def _second_derivative_nonuniform(values: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    left_spacing = coordinates[1:-1] - coordinates[:-2]
    right_spacing = coordinates[2:] - coordinates[1:-1]
    full_spacing = coordinates[2:] - coordinates[:-2]
    left_slope = (values[:, 1:-1] - values[:, :-2]) / left_spacing[None, :]
    right_slope = (values[:, 2:] - values[:, 1:-1]) / right_spacing[None, :]
    return 2.0 * (right_slope - left_slope) / full_spacing[None, :]


def convexity_violations(
    surface: np.ndarray,
    *,
    moneyness_points: np.ndarray,
) -> tuple[int, float]:
    """Count negative strike-space call-price curvature across moneyness."""

    values = _validate_positive_total_variance(surface)
    log_moneyness = _validate_coordinate_vector(
        moneyness_points,
        expected_size=values.shape[1],
        coordinate_name="moneyness_points",
    )
    strikes = np.exp(log_moneyness)
    call_prices = _normalized_call_prices(values, log_moneyness=log_moneyness)
    curvature = _second_derivative_nonuniform(call_prices, strikes)
    violations = np.minimum(curvature, 0.0)
    return int(np.count_nonzero(violations < 0.0)), float(np.abs(violations).sum())


def summarize_diagnostics(
    surface: np.ndarray,
    *,
    moneyness_points: np.ndarray,
) -> ArbitrageDiagnostics:
    """Summarize daily arbitrage-aware diagnostics."""

    calendar_count, calendar_magnitude = calendar_monotonicity_violations(surface)
    convexity_count, convexity_magnitude = convexity_violations(
        surface,
        moneyness_points=moneyness_points,
    )
    return ArbitrageDiagnostics(
        calendar_violation_count=calendar_count,
        calendar_violation_magnitude=calendar_magnitude,
        convexity_violation_count=convexity_count,
        convexity_violation_magnitude=convexity_magnitude,
    )

```

### src/ivsurf/features/lagged_surface.py

```python
"""Lagged-surface feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import polars as pl

from ivsurf.surfaces.grid import SurfaceGrid


@dataclass(frozen=True, slots=True)
class DailySurfaceArrays:
    """Dense daily arrays keyed by quote date."""

    quote_dates: list[date]
    completed_surfaces: np.ndarray
    observed_masks: np.ndarray
    observed_surfaces: np.ndarray
    vega_weights: np.ndarray


def pivot_surface_arrays(surface_frame: pl.DataFrame, grid: SurfaceGrid) -> DailySurfaceArrays:
    """Convert long daily surface rows into dense arrays."""

    sorted_frame = surface_frame.sort(["quote_date", "maturity_index", "moneyness_index"])
    raw_dates = (
        sorted_frame.select(pl.col("quote_date").unique(maintain_order=True)).to_series().to_list()
    )
    if any(not isinstance(value, date) for value in raw_dates):
        message = "Daily surface arrays require quote_date values to be Polars Date values."
        raise TypeError(message)
    dates = [cast(date, value) for value in raw_dates]
    rows_per_day = grid.shape[0] * grid.shape[1]
    observed_values = (
        sorted_frame["observed_total_variance"]
        .fill_null(np.nan)
        .to_numpy()
        .reshape(len(dates), rows_per_day)
    )
    completed_values = (
        sorted_frame["completed_total_variance"].to_numpy().reshape(len(dates), rows_per_day)
    )
    observed_masks = (
        sorted_frame["observed_mask"]
        .to_numpy()
        .reshape(len(dates), rows_per_day)
        .astype(np.float64)
    )
    vega_weights = sorted_frame["vega_sum"].to_numpy().reshape(len(dates), rows_per_day)
    return DailySurfaceArrays(
        quote_dates=dates,
        completed_surfaces=completed_values,
        observed_masks=observed_masks,
        observed_surfaces=observed_values,
        vega_weights=vega_weights,
    )


def summarize_lag_window(array: np.ndarray, index: int, window: int) -> np.ndarray:
    """Compute an equal-weight mean over the previous `window` observations."""

    start_index = index - window + 1
    if start_index < 0:
        message = f"Window {window} is not available at position {index}."
        raise ValueError(message)
    return np.asarray(array[start_index : index + 1].mean(axis=0), dtype=np.float64)

```

### src/ivsurf/features/factors.py

```python
"""Compact factor features for tree and HAR-style models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass(slots=True)
class SurfacePCAFactors:
    """Fit/transform helper for surface factors."""

    n_components: int
    pca: PCA | None = None

    def fit(self, surfaces: np.ndarray) -> SurfacePCAFactors:
        self.pca = PCA(n_components=self.n_components, svd_solver="full")
        self.pca.fit(surfaces)
        return self

    def transform(self, surfaces: np.ndarray) -> np.ndarray:
        if self.pca is None:
            message = "SurfacePCAFactors must be fit before transform."
            raise ValueError(message)
        return np.asarray(self.pca.transform(surfaces), dtype=np.float64)

    def inverse_transform(self, factors: np.ndarray) -> np.ndarray:
        if self.pca is None:
            message = "SurfacePCAFactors must be fit before inverse_transform."
            raise ValueError(message)
        return np.asarray(self.pca.inverse_transform(factors), dtype=np.float64)

```

### src/ivsurf/features/tabular_dataset.py

```python
"""Model-ready daily feature and target dataset builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl

from ivsurf.calendar import MarketCalendar
from ivsurf.config import FeatureConfig, MarketCalendarConfig
from ivsurf.features.lagged_surface import pivot_surface_arrays, summarize_lag_window
from ivsurf.features.liquidity import build_daily_liquidity_features
from ivsurf.qc.timing_checks import (
    assert_next_observed_target_date,
    assert_strictly_increasing_unique_dates,
)
from ivsurf.surfaces.grid import SurfaceGrid


@dataclass(frozen=True, slots=True)
class DailyDatasetBuildResult:
    """Feature/target table plus metadata."""

    feature_frame: pl.DataFrame


def _vector_columns(prefix: str, values: np.ndarray) -> dict[str, float]:
    return {f"{prefix}_{index:04d}": float(value) for index, value in enumerate(values)}


def build_target_training_weights(
    *,
    observed_mask: np.ndarray,
    vega_weights: np.ndarray,
) -> np.ndarray:
    """Build explicit neural training weights for the completed target surface.

    Completed targets are the supervised object for every model. Observed cells retain
    their positive target-day vega weighting, while completed-only cells receive a
    unit training weight so `imputed_loss_weight` changes the neural objective.
    """

    observed_mask_array = np.asarray(observed_mask, dtype=np.float64)
    vega_weight_array = np.asarray(vega_weights, dtype=np.float64)
    if observed_mask_array.shape != vega_weight_array.shape:
        message = (
            "observed_mask and vega_weights must share the same shape when building "
            "target training weights."
        )
        raise ValueError(message)
    if not np.isfinite(vega_weight_array).all():
        message = "target-day vega_weights must be finite when building training weights."
        raise ValueError(message)

    observed_cells = observed_mask_array > 0.5
    observed_vega = np.maximum(vega_weight_array, 0.0)
    invalid_observed_cells = observed_cells & (observed_vega <= 0.0)
    if invalid_observed_cells.any():
        message = (
            "Observed target cells must retain strictly positive target-day vega when "
            "building neural training weights."
        )
        raise ValueError(message)

    return np.where(observed_cells, observed_vega, 1.0).astype(np.float64, copy=False)


def _count_intervening_trading_sessions(
    calendar: MarketCalendar,
    *,
    quote_date: date,
    target_date: date,
) -> int:
    """Count trading sessions strictly between quote_date and target_date."""

    if target_date <= quote_date:
        message = "target_date must be after quote_date when counting target-gap sessions."
        raise ValueError(message)

    intervening_sessions = 0
    current_date = quote_date
    while True:
        current_date = calendar.next_trading_session(current_date)
        if current_date == target_date:
            return intervening_sessions
        if current_date > target_date:
            message = (
                "target_date must be a trading session reachable from quote_date: "
                f"quote_date={quote_date.isoformat()} "
                f"target_date={target_date.isoformat()}."
            )
            raise ValueError(message)
        intervening_sessions += 1


def build_daily_feature_dataset(
    surface_frame: pl.DataFrame,
    grid: SurfaceGrid,
    feature_config: FeatureConfig,
    calendar_config: MarketCalendarConfig,
) -> DailyDatasetBuildResult:
    """Build one model-ready row per quote date."""

    surface_arrays = pivot_surface_arrays(surface_frame=surface_frame, grid=grid)
    liquidity_frame = build_daily_liquidity_features(surface_frame)
    liquidity_by_date = {row["quote_date"]: row for row in liquidity_frame.iter_rows(named=True)}
    calendar = MarketCalendar(
        calendar_name=calendar_config.calendar_name,
        timezone=calendar_config.timezone,
        decision_time=calendar_config.decision_time,
        decision_snapshot_minutes_before_close=calendar_config.decision_snapshot_minutes_before_close,
        am_settled_roots=calendar_config.am_settled_roots,
    )

    max_window = max(feature_config.lag_windows)
    required_surface_dates = max_window + 1
    if feature_config.include_daily_change:
        required_surface_dates = max(required_surface_dates, 3)
    if len(surface_arrays.quote_dates) < required_surface_dates:
        message = (
            "Not enough daily surfaces to build a supervised feature dataset. "
            f"Need at least {required_surface_dates} decision-time-aligned dates, "
            f"received {len(surface_arrays.quote_dates)}."
        )
        raise ValueError(message)

    assert_strictly_increasing_unique_dates(
        surface_arrays.quote_dates,
        context="Observed gold-surface quote dates",
    )

    rows: list[dict[str, object]] = []
    start_position = max_window - 1
    if feature_config.include_daily_change:
        start_position = max(start_position, 1)
    for position in range(start_position, len(surface_arrays.quote_dates) - 1):
        quote_date = surface_arrays.quote_dates[position]
        target_date = surface_arrays.quote_dates[position + 1]
        assert_next_observed_target_date(
            surface_arrays.quote_dates,
            position=position,
            quote_date=quote_date,
            target_date=target_date,
        )

        row: dict[str, object] = {
            "quote_date": quote_date,
            "target_date": target_date,
            "target_gap_sessions": _count_intervening_trading_sessions(
                calendar,
                quote_date=quote_date,
                target_date=target_date,
            ),
        }
        for window in feature_config.lag_windows:
            lag_surface = summarize_lag_window(surface_arrays.completed_surfaces, position, window)
            row.update(_vector_columns(f"feature_surface_mean_{window:02d}", lag_surface))
            if feature_config.include_mask:
                lag_mask = summarize_lag_window(surface_arrays.observed_masks, position, window)
                row.update(_vector_columns(f"feature_mask_mean_{window:02d}", lag_mask))

        if feature_config.include_daily_change:
            change_vector = (
                surface_arrays.completed_surfaces[position]
                - surface_arrays.completed_surfaces[position - 1]
            )
            row.update(_vector_columns("feature_surface_change_01", change_vector))

        if feature_config.include_liquidity:
            if quote_date not in liquidity_by_date:
                message = f"Missing liquidity features for quote_date={quote_date.isoformat()}."
                raise KeyError(message)
            daily_liquidity = liquidity_by_date[quote_date]
            row["feature_coverage_ratio"] = float(daily_liquidity["coverage_ratio"])
            row["feature_daily_vega_sum"] = float(daily_liquidity["daily_vega_sum"])
            row["feature_daily_option_count"] = int(daily_liquidity["daily_option_count"])
            row["feature_daily_weighted_spread_1545"] = float(
                daily_liquidity["daily_weighted_spread_1545"]
            )

        target_surface = surface_arrays.completed_surfaces[position + 1]
        target_mask = surface_arrays.observed_masks[position + 1]
        target_vega = surface_arrays.vega_weights[position + 1]
        target_training_weights = build_target_training_weights(
            observed_mask=target_mask,
            vega_weights=target_vega,
        )
        row.update(_vector_columns("target_total_variance", target_surface))
        row.update(_vector_columns("target_observed_mask", target_mask))
        row.update(_vector_columns("target_vega_weight", target_vega))
        row.update(_vector_columns("target_training_weight", target_training_weights))
        rows.append(row)

    return DailyDatasetBuildResult(feature_frame=pl.DataFrame(rows).sort("quote_date"))

```

### src/ivsurf/models/naive.py

```python
"""Naive persistence benchmark."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ivsurf.models.base import SurfaceForecastModel


def validate_naive_feature_layout(
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    lag_prefix: str = "feature_surface_mean_01_",
    target_prefix: str = "target_total_variance_",
) -> None:
    """Fail fast if the naive benchmark inputs do not exactly match the target grid layout."""

    target_suffixes = tuple(
        column.removeprefix(target_prefix)
        for column in target_columns
        if column.startswith(target_prefix)
    )
    if len(target_suffixes) != len(target_columns):
        message = (
            "Naive baseline requires target_total_variance columns with the expected "
            f"prefix {target_prefix!r}."
        )
        raise ValueError(message)
    expected_feature_columns = tuple(f"{lag_prefix}{suffix}" for suffix in target_suffixes)
    leading_feature_columns = tuple(feature_columns[: len(target_columns)])
    if leading_feature_columns != expected_feature_columns:
        message = (
            "Naive baseline requires lag-1 surface features to be present first and aligned "
            "one-to-one with the target surface columns. "
            f"Expected leading columns {expected_feature_columns!r}, "
            f"found {leading_feature_columns!r}."
        )
        raise ValueError(message)


class NaiveSurfaceModel(SurfaceForecastModel):
    """Forecast tomorrow's surface as today's completed surface."""

    def __init__(self, lag_prefix: str = "feature_surface_mean_01_") -> None:
        self.lag_prefix = lag_prefix
        self._column_count: int | None = None

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
    ) -> NaiveSurfaceModel:
        self._column_count = targets.shape[1]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self._column_count is None:
            message = "NaiveSurfaceModel must be fit before predict."
            raise ValueError(message)
        return features[:, : self._column_count].copy()

```

### src/ivsurf/models/har_factor.py

```python
"""HAR-style factor surface model."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ivsurf.features.factors import SurfacePCAFactors
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.positive_target import LogPositiveTargetAdapter


class HarFactorSurfaceModel(SurfaceForecastModel):
    """Project surfaces to factors, run HAR-style regression, then reconstruct."""

    def __init__(self, n_factors: int, alpha: float, target_dim: int) -> None:
        self.n_factors = n_factors
        self.alpha = alpha
        self.target_dim = target_dim
        self.feature_scaler = StandardScaler()
        self.factor_model = Ridge(alpha=alpha)
        self.factorizer = SurfacePCAFactors(n_components=n_factors)
        self.target_adapter = LogPositiveTargetAdapter("HarFactorSurfaceModel")

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
    ) -> HarFactorSurfaceModel:
        lag_1 = features[:, : self.target_dim]
        lag_5 = features[:, self.target_dim : (2 * self.target_dim)]
        lag_22 = features[:, (2 * self.target_dim) : (3 * self.target_dim)]
        transformed_targets = self.target_adapter.transform_targets(
            targets,
            array_name="training targets",
        )
        transformed_lag_1 = self.target_adapter.transform_targets(
            lag_1,
            array_name="lag_1 features",
        )
        transformed_lag_5 = self.target_adapter.transform_targets(
            lag_5,
            array_name="lag_5 features",
        )
        transformed_lag_22 = self.target_adapter.transform_targets(
            lag_22,
            array_name="lag_22 features",
        )

        self.factorizer.fit(transformed_targets)
        target_factors = self.factorizer.transform(transformed_targets)
        har_features = np.concatenate(
            [
                self.factorizer.transform(transformed_lag_1),
                self.factorizer.transform(transformed_lag_5),
                self.factorizer.transform(transformed_lag_22),
            ],
            axis=1,
        )
        scaled_features = self.feature_scaler.fit_transform(har_features)
        self.factor_model.fit(scaled_features, target_factors)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        lag_1 = features[:, : self.target_dim]
        lag_5 = features[:, self.target_dim : (2 * self.target_dim)]
        lag_22 = features[:, (2 * self.target_dim) : (3 * self.target_dim)]
        har_features = np.concatenate(
            [
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_1, array_name="lag_1 features")
                ),
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_5, array_name="lag_5 features")
                ),
                self.factorizer.transform(
                    self.target_adapter.transform_targets(lag_22, array_name="lag_22 features")
                ),
            ],
            axis=1,
        )
        predicted_factors = self.factor_model.predict(self.feature_scaler.transform(har_features))
        return self.target_adapter.inverse_predictions(
            self.factorizer.inverse_transform(predicted_factors)
        )

```

### src/ivsurf/models/neural_surface.py

```python
"""Arbitrage-aware neural surface model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import optuna
import torch
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset

from ivsurf.config import NeuralModelConfig, TrainingProfileConfig
from ivsurf.evaluation.loss_panels import mean_daily_loss_metric
from ivsurf.models.base import SurfaceForecastModel
from ivsurf.models.losses import weighted_surface_mse
from ivsurf.models.penalties import (
    calendar_monotonicity_penalty,
    convexity_penalty,
    roughness_penalty,
)

NEURAL_GRADIENT_CLIP_NORM = 5.0


class NeuralSurfaceMLP(nn.Module):
    """Compact MLP for joint surface prediction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_width: int,
        depth: int,
        dropout: float,
        output_total_variance_floor: float,
    ) -> None:
        super().__init__()
        if output_total_variance_floor <= 0.0:
            message = "output_total_variance_floor must be strictly positive."
            raise ValueError(message)
        self.output_total_variance_floor = output_total_variance_floor
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(depth):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_width),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_width
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw_predictions = cast(torch.Tensor, self.network(features))
        return cast(
            torch.Tensor,
            functional.softplus(raw_predictions.to(dtype=torch.float64))
            + self.output_total_variance_floor,
        )


def _resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        message = "NeuralSurfaceRegressor requested CUDA, but torch.cuda.is_available() is False."
        raise RuntimeError(message)
    return device


def _clone_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _feature_standardization_stats(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(features.mean(axis=0), dtype=np.float64)
    scale = np.asarray(features.std(axis=0), dtype=np.float64)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    return mean, safe_scale


def _standardize_features(
    features: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return np.asarray((features - mean) / scale, dtype=np.float64)


def _validated_total_variance(
    total_variance_predictions: torch.Tensor,
    *,
    context: str,
) -> torch.Tensor:
    if not bool(torch.isfinite(total_variance_predictions).all().detach().item()):
        message = f"{context} produced non-finite total-variance predictions."
        raise RuntimeError(message)
    if not bool((total_variance_predictions >= 0.0).all().detach().item()):
        message = f"{context} produced negative total-variance predictions."
        raise RuntimeError(message)
    return total_variance_predictions


@dataclass(frozen=True, slots=True)
class NeuralValidationDiagnostics:
    """Validation diagnostics persisted for the selected neural checkpoint."""

    metric_name: str
    metric_value: float
    prediction_mean: float
    target_mean: float
    prediction_target_ratio: float
    prediction_below_1e_6_share: float
    calendar_penalty: float
    convexity_penalty: float
    roughness_penalty: float


def _predict_total_variance_array(
    model: NeuralSurfaceMLP,
    *,
    device: torch.device,
    standardized_features: np.ndarray,
) -> np.ndarray:
    with torch.inference_mode():
        feature_tensor = torch.as_tensor(
            standardized_features,
            dtype=torch.float32,
            device=device,
        )
        predictions = _validated_total_variance(
            model(feature_tensor),
            context="NeuralSurfaceRegressor inference",
        )
    return np.asarray(predictions.cpu().numpy(), dtype=np.float64)


def _validation_score_and_diagnostics(
    model: NeuralSurfaceMLP,
    *,
    device: torch.device,
    standardized_features: np.ndarray,
    targets: np.ndarray,
    observed_masks: np.ndarray,
    vega_weights: np.ndarray,
    grid_shape: tuple[int, int],
    moneyness_points: tuple[float, ...],
    metric_name: str,
    positive_floor: float,
) -> NeuralValidationDiagnostics:
    predictions = _predict_total_variance_array(
        model,
        device=device,
        standardized_features=standardized_features,
    )
    metric_value = mean_daily_loss_metric(
        metric_name=metric_name,
        y_true=targets,
        y_pred=predictions,
        observed_masks=observed_masks,
        vega_weights=vega_weights,
        positive_floor=positive_floor,
    )
    prediction_tensor = torch.as_tensor(predictions, dtype=torch.float32, device=device)
    target_mean = float(np.mean(targets))
    prediction_mean = float(np.mean(predictions))
    return NeuralValidationDiagnostics(
        metric_name=metric_name,
        metric_value=metric_value,
        prediction_mean=prediction_mean,
        target_mean=target_mean,
        prediction_target_ratio=(
            prediction_mean / target_mean if target_mean > 0.0 else float("nan")
        ),
        prediction_below_1e_6_share=float(np.mean(predictions < 1.0e-6)),
        calendar_penalty=float(
            calendar_monotonicity_penalty(prediction_tensor, grid_shape).detach().cpu().item()
        ),
        convexity_penalty=float(
            convexity_penalty(
                prediction_tensor,
                grid_shape,
                moneyness_points=moneyness_points,
            )
            .detach()
            .cpu()
            .item()
        ),
        roughness_penalty=float(
            roughness_penalty(prediction_tensor, grid_shape).detach().cpu().item()
        ),
    )


@dataclass(slots=True)
class NeuralSurfaceRegressor(SurfaceForecastModel):
    """Torch regressor wrapper with explicit penalty weights."""

    config: NeuralModelConfig
    grid_shape: tuple[int, int]
    moneyness_points: tuple[float, ...]
    model: NeuralSurfaceMLP | None = None
    best_epoch: int | None = None
    epochs_completed: int = 0
    best_validation_score: float | None = None
    feature_mean: np.ndarray | None = None
    feature_scale: np.ndarray | None = None
    validation_diagnostics: NeuralValidationDiagnostics | None = None

    def __post_init__(self) -> None:
        if len(self.moneyness_points) != self.grid_shape[1]:
            message = (
                "NeuralSurfaceRegressor moneyness_points length must match the "
                f"moneyness grid size: {len(self.moneyness_points)} != {self.grid_shape[1]}."
            )
            raise ValueError(message)
        moneyness_array = np.asarray(self.moneyness_points, dtype=np.float64)
        if not np.isfinite(moneyness_array).all():
            message = "NeuralSurfaceRegressor moneyness_points must be finite."
            raise ValueError(message)
        if not np.all(np.diff(moneyness_array) > 0.0):
            message = "NeuralSurfaceRegressor moneyness_points must be strictly increasing."
            raise ValueError(message)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        observed_masks: np.ndarray | None = None,
        vega_weights: np.ndarray | None = None,
        training_weights: np.ndarray | None = None,
        *,
        validation_features: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        validation_observed_masks: np.ndarray | None = None,
        validation_vega_weights: np.ndarray | None = None,
        training_profile: TrainingProfileConfig | None = None,
        trial: optuna.Trial | None = None,
        trial_step_offset: int = 0,
        validation_metric_name: str = "observed_mse_total_variance",
        validation_positive_floor: float = 1.0e-8,
    ) -> NeuralSurfaceRegressor:
        if observed_masks is None:
            message = "NeuralSurfaceRegressor requires observed_masks."
            raise ValueError(message)
        if training_weights is None:
            message = "NeuralSurfaceRegressor requires training_weights."
            raise ValueError(message)
        if training_profile is not None and (
            validation_features is None
            or validation_targets is None
            or validation_observed_masks is None
            or validation_vega_weights is None
        ):
            message = "Validation arrays are required when a training_profile is provided."
            raise ValueError(message)

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

        device = _resolve_device(self.config.device)
        input_dim = features.shape[1]
        output_dim = targets.shape[1]
        feature_mean, feature_scale = _feature_standardization_stats(features)
        standardized_features = _standardize_features(
            features,
            mean=feature_mean,
            scale=feature_scale,
        )
        standardized_validation_features = (
            None
            if validation_features is None
            else _standardize_features(
                validation_features,
                mean=feature_mean,
                scale=feature_scale,
            )
        )
        model = NeuralSurfaceMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_width=self.config.hidden_width,
            depth=self.config.depth,
            dropout=self.config.dropout,
            output_total_variance_floor=self.config.output_total_variance_floor,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        use_cuda = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_cuda else None
        dataset = TensorDataset(
            torch.as_tensor(standardized_features, dtype=torch.float32),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(observed_masks, dtype=torch.float32),
            torch.as_tensor(training_weights, dtype=torch.float32),
        )
        generator = torch.Generator().manual_seed(self.config.seed)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            pin_memory=use_cuda,
        )

        best_state_dict: dict[str, torch.Tensor] | None = None
        best_epoch: int | None = None
        best_validation_score = float("inf")
        best_validation_diagnostics: NeuralValidationDiagnostics | None = None
        epochs_without_improvement = 0
        max_epochs = training_profile.epochs if training_profile is not None else self.config.epochs
        min_delta = (
            training_profile.neural_early_stopping_min_delta
            if training_profile is not None
            else 0.0
        )
        min_epochs_before_early_stop = (
            training_profile.neural_min_epochs_before_early_stop
            if training_profile is not None
            else 1
        )
        patience = (
            training_profile.neural_early_stopping_patience
            if training_profile is not None
            else max_epochs
        )

        model.train()
        for epoch in range(max_epochs):
            for (
                batch_features,
                batch_targets,
                batch_masks,
                batch_training_weights,
            ) in loader:
                batch_features = batch_features.to(device, non_blocking=use_cuda)
                batch_targets = batch_targets.to(device, non_blocking=use_cuda)
                batch_masks = batch_masks.to(device, non_blocking=use_cuda)
                batch_training_weights = batch_training_weights.to(
                    device,
                    non_blocking=use_cuda,
                )
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_cuda):
                    predictions = _validated_total_variance(
                        model(batch_features),
                        context="NeuralSurfaceRegressor training",
                    )
                    loss = weighted_surface_mse(
                        predictions=predictions,
                        targets=batch_targets,
                        observed_mask=batch_masks,
                        training_weights=batch_training_weights,
                        observed_loss_weight=self.config.observed_loss_weight,
                        imputed_loss_weight=self.config.imputed_loss_weight,
                    )
                    loss = loss + (
                        self.config.calendar_penalty_weight
                        * calendar_monotonicity_penalty(predictions, self.grid_shape)
                    )
                    loss = loss + (
                        self.config.convexity_penalty_weight
                        * convexity_penalty(
                            predictions,
                            self.grid_shape,
                            moneyness_points=self.moneyness_points,
                        )
                    )
                    loss = loss + (
                        self.config.roughness_penalty_weight
                        * roughness_penalty(predictions, self.grid_shape)
                    )

                if use_cuda:
                    if scaler is None:
                        message = "CUDA training requires an initialized GradScaler."
                        raise RuntimeError(message)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=NEURAL_GRADIENT_CLIP_NORM,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=NEURAL_GRADIENT_CLIP_NORM,
                    )
                    optimizer.step()

            self.epochs_completed = epoch + 1
            if training_profile is None:
                continue

            if (
                validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable during training."
                raise RuntimeError(message)
            validation_diagnostics = _validation_score_and_diagnostics(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                grid_shape=self.grid_shape,
                moneyness_points=self.moneyness_points,
                metric_name=validation_metric_name,
                positive_floor=validation_positive_floor,
            )
            validation_score = validation_diagnostics.metric_value
            if validation_score < (best_validation_score - min_delta):
                best_validation_score = validation_score
                best_epoch = epoch + 1
                best_state_dict = _clone_state_dict(model)
                best_validation_diagnostics = validation_diagnostics
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if trial is not None:
                trial.report(validation_score, trial_step_offset + epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if (
                (epoch + 1) >= min_epochs_before_early_stop
                and epochs_without_improvement >= patience
            ):
                break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        elif training_profile is not None:
            if (
                validation_targets is None
                or validation_observed_masks is None
                or validation_vega_weights is None
                or standardized_validation_features is None
            ):
                message = "Validation arrays unexpectedly became unavailable after training."
                raise RuntimeError(message)
            fallback_diagnostics = _validation_score_and_diagnostics(
                model,
                device=device,
                standardized_features=standardized_validation_features,
                targets=validation_targets,
                observed_masks=validation_observed_masks,
                vega_weights=validation_vega_weights,
                grid_shape=self.grid_shape,
                moneyness_points=self.moneyness_points,
                metric_name=validation_metric_name,
                positive_floor=validation_positive_floor,
            )
            best_validation_score = fallback_diagnostics.metric_value
            best_epoch = self.epochs_completed
            best_validation_diagnostics = fallback_diagnostics

        self.best_epoch = best_epoch
        self.best_validation_score = (
            None if best_validation_score == float("inf") else float(best_validation_score)
        )
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.validation_diagnostics = best_validation_diagnostics
        self.model = model.eval()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if (
            self.model is None
            or self.feature_mean is None
            or self.feature_scale is None
        ):
            message = "NeuralSurfaceRegressor must be fit before predict."
            raise ValueError(message)
        device = _resolve_device(self.config.device)
        standardized_features = _standardize_features(
            features,
            mean=self.feature_mean,
            scale=self.feature_scale,
        )
        return _predict_total_variance_array(
            self.model,
            device=device,
            standardized_features=standardized_features,
        )

```

### src/ivsurf/models/losses.py

```python
"""Forecast losses for the neural surface model."""

from __future__ import annotations

import torch


def weighted_surface_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    observed_mask: torch.Tensor,
    training_weights: torch.Tensor,
    observed_loss_weight: float,
    imputed_loss_weight: float,
) -> torch.Tensor:
    """Weighted MSE for completed-surface supervision in the neural model."""

    observation_weights = torch.where(
        observed_mask > 0.5,
        torch.full_like(predictions, observed_loss_weight),
        torch.full_like(predictions, imputed_loss_weight),
    )
    positive_training_weights = torch.clamp_min(training_weights, 0.0)
    weights = observation_weights * positive_training_weights
    imputed_cells = observed_mask <= 0.5
    if (
        imputed_loss_weight > 0.0
        and bool(torch.any(imputed_cells).detach().item())
        and float(weights[imputed_cells].sum().detach().item()) <= 0.0
    ):
        message = (
            "weighted_surface_mse requires positive training weight on at least one "
            "completed-only target cell when imputed_loss_weight > 0."
        )
        raise ValueError(message)
    weight_sum = weights.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        message = "weighted_surface_mse requires at least one positive supervised target cell."
        raise ValueError(message)
    return ((predictions - targets).square() * weights).sum() / weight_sum

```

### src/ivsurf/models/penalties.py

```python
"""Arbitrage-aware neural penalties."""

from __future__ import annotations

import math

import torch


def _reshape_surface(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    return predictions.view(predictions.shape[0], grid_shape[0], grid_shape[1])


def _coordinate_tensor(
    coordinates: tuple[float, ...],
    *,
    expected_size: int,
    device: torch.device,
    dtype: torch.dtype,
    coordinate_name: str,
) -> torch.Tensor:
    if len(coordinates) != expected_size:
        message = (
            f"{coordinate_name} length {len(coordinates)} does not match expected size "
            f"{expected_size}."
        )
        raise ValueError(message)
    tensor = torch.as_tensor(coordinates, dtype=dtype, device=device)
    if not bool(torch.isfinite(tensor).all().detach().item()):
        message = f"{coordinate_name} must contain only finite values."
        raise ValueError(message)
    if not bool((tensor[1:] > tensor[:-1]).all().detach().item()):
        message = f"{coordinate_name} must be strictly increasing."
        raise ValueError(message)
    return tensor


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _normalized_call_prices(
    total_variance: torch.Tensor,
    *,
    log_moneyness: torch.Tensor,
) -> torch.Tensor:
    if not bool(torch.isfinite(total_variance).all().detach().item()):
        message = "total_variance must contain only finite values."
        raise ValueError(message)
    if not bool((total_variance > 0.0).all().detach().item()):
        message = "total_variance must be strictly positive for price convexity penalties."
        raise ValueError(message)
    strikes = torch.exp(log_moneyness)
    sqrt_variance = torch.sqrt(total_variance)
    d1 = ((-log_moneyness).view(1, 1, -1) + (0.5 * total_variance)) / sqrt_variance
    d2 = d1 - sqrt_variance
    return _normal_cdf(d1) - (strikes.view(1, 1, -1) * _normal_cdf(d2))


def _second_derivative_nonuniform(
    values: torch.Tensor,
    coordinates: torch.Tensor,
) -> torch.Tensor:
    left_spacing = coordinates[1:-1] - coordinates[:-2]
    right_spacing = coordinates[2:] - coordinates[1:-1]
    full_spacing = coordinates[2:] - coordinates[:-2]
    left_slope = (values[:, :, 1:-1] - values[:, :, :-2]) / left_spacing.view(1, 1, -1)
    right_slope = (values[:, :, 2:] - values[:, :, 1:-1]) / right_spacing.view(1, 1, -1)
    return 2.0 * (right_slope - left_slope) / full_spacing.view(1, 1, -1)


def calendar_monotonicity_penalty(
    predictions: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    """Penalize decreases across maturity."""

    surface = _reshape_surface(predictions, grid_shape)
    diffs = surface[:, 1:, :] - surface[:, :-1, :]
    return torch.relu(-diffs).mean()


def convexity_penalty(
    predictions: torch.Tensor,
    grid_shape: tuple[int, int],
    *,
    moneyness_points: tuple[float, ...],
) -> torch.Tensor:
    """Penalize negative strike-space call-price curvature across moneyness."""

    if grid_shape[1] < 3:
        return predictions.new_zeros(())
    surface = _reshape_surface(predictions, grid_shape)
    log_moneyness = _coordinate_tensor(
        moneyness_points,
        expected_size=grid_shape[1],
        device=surface.device,
        dtype=surface.dtype,
        coordinate_name="moneyness_points",
    )
    strikes = torch.exp(log_moneyness)
    call_prices = _normalized_call_prices(surface, log_moneyness=log_moneyness)
    curvature = _second_derivative_nonuniform(call_prices, strikes)
    return torch.relu(-curvature).mean()


def roughness_penalty(predictions: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """Discourage large local curvature in either axis."""

    surface = _reshape_surface(predictions, grid_shape)
    maturity_curvature = surface[:, 2:, :] - (2.0 * surface[:, 1:-1, :]) + surface[:, :-2, :]
    moneyness_curvature = surface[:, :, 2:] - (2.0 * surface[:, :, 1:-1]) + surface[:, :, :-2]
    return maturity_curvature.square().mean() + moneyness_curvature.square().mean()

```

### src/ivsurf/splits/walkforward.py

```python
"""Blocked walk-forward split generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.manifests import WalkforwardSplit


@dataclass(frozen=True, slots=True)
class CleanEvaluationBoundary:
    """Boundary between HPO-used validation dates and clean evaluation test dates."""

    max_hpo_validation_date: date
    first_clean_test_split_id: str


def build_walkforward_splits(
    dates: list[date],
    config: WalkforwardConfig,
) -> list[WalkforwardSplit]:
    """Build explicit non-overlapping blocked time-series splits."""

    if len(dates) < (config.train_size + config.validation_size + config.test_size):
        message = "Not enough dates to build one walk-forward split."
        raise ValueError(message)

    splits: list[WalkforwardSplit] = []
    test_start = config.train_size + config.validation_size
    split_number = 0
    while (test_start + config.test_size) <= len(dates):
        validation_start = test_start - config.validation_size
        train_end = validation_start
        train_start = 0 if config.expanding_train else max(0, train_end - config.train_size)

        split = WalkforwardSplit(
            split_id=f"split_{split_number:04d}",
            train_dates=tuple(day.isoformat() for day in dates[train_start:train_end]),
            validation_dates=tuple(day.isoformat() for day in dates[validation_start:test_start]),
            test_dates=tuple(
                day.isoformat()
                for day in dates[test_start : test_start + config.test_size]
            ),
        )
        splits.append(split)
        split_number += 1
        test_start += config.step_size
    return splits


def clean_evaluation_boundary(
    splits: list[WalkforwardSplit],
    *,
    tuning_splits_count: int,
) -> CleanEvaluationBoundary:
    """Return the first uncontaminated evaluation split after HPO validation windows."""

    if not splits:
        message = "At least one walk-forward split is required."
        raise ValueError(message)
    if tuning_splits_count > len(splits):
        message = (
            "tuning_splits_count cannot exceed the number of available splits: "
            f"{tuning_splits_count} > {len(splits)}."
        )
        raise ValueError(message)

    hpo_validation_dates = [
        date.fromisoformat(day)
        for split in splits[:tuning_splits_count]
        for day in split.validation_dates
    ]
    if not hpo_validation_dates:
        message = "HPO tuning splits must contain at least one validation date."
        raise ValueError(message)
    max_hpo_validation_date = max(hpo_validation_dates)

    for split in splits:
        if not split.test_dates:
            message = f"Walk-forward split {split.split_id} has no test dates."
            raise ValueError(message)
        test_start_date = date.fromisoformat(split.test_dates[0])
        if test_start_date > max_hpo_validation_date:
            return CleanEvaluationBoundary(
                max_hpo_validation_date=max_hpo_validation_date,
                first_clean_test_split_id=split.split_id,
            )

    message = (
        "No clean evaluation split remains after excluding all test windows that overlap the "
        "HPO-used validation sample."
    )
    raise ValueError(message)


def clean_evaluation_splits(
    splits: list[WalkforwardSplit],
    *,
    tuning_splits_count: int,
) -> tuple[CleanEvaluationBoundary, list[WalkforwardSplit]]:
    """Return the boundary metadata and the clean evaluation splits only."""

    boundary = clean_evaluation_boundary(splits, tuning_splits_count=tuning_splits_count)
    clean_splits = [
        split
        for split in splits
        if date.fromisoformat(split.test_dates[0]) > boundary.max_hpo_validation_date
    ]
    if not clean_splits:
        message = "Expected at least one clean evaluation split after HPO boundary filtering."
        raise ValueError(message)
    return boundary, clean_splits

```

### src/ivsurf/splits/manifests.py

```python
"""Split manifest serialization and hashing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import orjson

from ivsurf.io.atomic import write_bytes_atomic


@dataclass(frozen=True, slots=True)
class WalkforwardSplit:
    """Explicit walk-forward split."""

    split_id: str
    train_dates: tuple[str, ...]
    validation_dates: tuple[str, ...]
    test_dates: tuple[str, ...]


def serialize_splits(splits: list[WalkforwardSplit], output_path: Path) -> str:
    """Write split manifest and return its SHA256 hash."""

    payload = [asdict(split) for split in splits]
    raw = orjson.dumps(payload, option=orjson.OPT_INDENT_2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(output_path, raw)
    return sha256(raw).hexdigest()


def load_splits(path: Path) -> list[WalkforwardSplit]:
    """Load a split manifest from disk."""

    payload = orjson.loads(path.read_bytes())
    return [WalkforwardSplit(**item) for item in payload]

```

### src/ivsurf/evaluation/alignment.py

```python
"""Forecast-artifact alignment against realized surfaces and spot states."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import total_variance_to_iv, validate_total_variance_array
from ivsurf.io.parquet import scan_parquet_files
from ivsurf.io.paths import sorted_artifact_files


def _require_files(paths: list[Path], description: str) -> None:
    if not paths:
        message = f"No {description} files found."
        raise FileNotFoundError(message)


def _require_non_null_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        if frame[column].null_count() > 0:
            message = f"Aligned evaluation panel contains nulls in required column {column}."
            raise ValueError(message)


def _require_finite_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        if not np.isfinite(values).all():
            message = f"Aligned evaluation panel contains non-finite values in column {column}."
            raise ValueError(message)


def _require_non_negative_float_columns(frame: pl.DataFrame, *, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = frame[column].to_numpy().astype(np.float64, copy=False)
        validate_total_variance_array(
            values,
            context=f"Aligned evaluation panel column {column}",
            allow_zero=True,
        )


def _format_spot_contract_violations(
    frame: pl.DataFrame,
    *,
    columns: tuple[str, ...],
) -> str:
    violations: list[str] = []
    for row in frame.select("quote_date", *columns).iter_rows(named=True):
        quote_date = cast(date, row["quote_date"])
        details = ", ".join(f"{column}={row[column]!r}" for column in columns)
        violations.append(f"{quote_date.isoformat()} ({details})")
    return "; ".join(violations)


def load_actual_surface_frame(gold_dir: Path) -> pl.DataFrame:
    """Load persisted daily surface artifacts."""

    gold_files = sorted_artifact_files(gold_dir, "year=*/*.parquet")
    _require_files(gold_files, "gold surface")
    return (
        scan_parquet_files(gold_files)
        .select(
            "quote_date",
            "maturity_index",
            "maturity_days",
            "moneyness_index",
            "moneyness_point",
            "observed_total_variance",
            "observed_iv",
            "completed_total_variance",
            "completed_iv",
            "observed_mask",
            "vega_sum",
        )
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )


def load_forecast_frame(forecast_dir: Path) -> pl.DataFrame:
    """Load persisted forecast artifacts."""

    forecast_files = sorted_artifact_files(forecast_dir, "*.parquet")
    _require_files(forecast_files, "forecast")
    return (
        scan_parquet_files(forecast_files)
        .collect(engine="streaming")
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )


def load_daily_spot_frame(silver_dir: Path) -> pl.DataFrame:
    """Load the official stage-08 daily spot from valid active_underlying_price_1545 rows."""

    silver_files = sorted_artifact_files(silver_dir, "year=*/*.parquet")
    _require_files(silver_files, "silver")
    lazy_frame = scan_parquet_files(silver_files)
    spot_frame = (
        lazy_frame.select("quote_date", "active_underlying_price_1545", "is_valid_observation")
        .filter(pl.col("is_valid_observation"))
        .group_by("quote_date")
        .agg(
            pl.len().alias("valid_spot_row_count"),
            pl.col("active_underlying_price_1545").n_unique().alias("active_spot_n_unique"),
            pl.col("active_underlying_price_1545").median().alias("spot_1545"),
            pl.col("active_underlying_price_1545").min().alias("active_spot_min"),
            pl.col("active_underlying_price_1545").max().alias("active_spot_max"),
        )
        .collect(engine="streaming")
        .sort("quote_date")
    )
    if spot_frame.height == 0:
        message = "No valid silver rows available to derive stage-08 daily spot states."
        raise ValueError(message)
    invalid_spot_frames = [
        spot_frame.filter(pl.col("valid_spot_row_count") <= 0),
        spot_frame.filter(
            pl.col("spot_1545").is_null()
            | (~pl.col("spot_1545").is_finite())
            | (pl.col("spot_1545") <= 0.0)
        ),
    ]
    invalid_spots = pl.concat(invalid_spot_frames).unique(subset=["quote_date"]).sort("quote_date")
    if invalid_spots.height > 0:
        message = (
            "Expected strictly positive finite stage-08 daily spot values derived from the "
            "median active_underlying_price_1545 across valid silver rows. Violations: "
            f"{_format_spot_contract_violations(invalid_spots, columns=('spot_1545',))}."
        )
        raise ValueError(message)
    return spot_frame.select("quote_date", "spot_1545")


def assert_forecast_origins_after_hpo_boundary(
    forecast_frame: pl.DataFrame,
    *,
    max_hpo_validation_date: date,
) -> None:
    """Fail fast if forecast artifacts include HPO-contaminated origin dates."""

    contaminated = forecast_frame.filter(pl.col("quote_date") <= pl.lit(max_hpo_validation_date))
    if contaminated.is_empty():
        return
    earliest_quote_date = cast(date, contaminated["quote_date"].min())
    latest_quote_date = cast(date, contaminated["quote_date"].max())
    earliest_target_date = cast(date, contaminated["target_date"].min())
    latest_target_date = cast(date, contaminated["target_date"].max())
    message = (
        "Forecast artifacts include quote_date values that were inside the HPO-used validation "
        f"sample. Found {contaminated.height} contaminated rows with quote_date range "
        f"[{earliest_quote_date.isoformat()}, {latest_quote_date.isoformat()}], "
        f"target_date range "
        f"[{earliest_target_date.isoformat()}, {latest_target_date.isoformat()}], and boundary "
        f"{max_hpo_validation_date.isoformat()}."
    )
    raise ValueError(message)


def build_forecast_realization_panel(
    actual_surface_frame: pl.DataFrame,
    forecast_frame: pl.DataFrame,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Align forecast artifacts with realized target-day surfaces and origin-day references."""

    actual_target = actual_surface_frame.rename(
        {
            "quote_date": "target_date",
            "observed_total_variance": "actual_observed_total_variance",
            "observed_iv": "actual_observed_iv",
            "completed_total_variance": "actual_completed_total_variance",
            "completed_iv": "actual_completed_iv",
            "observed_mask": "actual_observed_mask",
            "vega_sum": "actual_vega_sum",
        }
    )
    origin_surface = actual_surface_frame.rename(
        {
            "completed_total_variance": "origin_completed_total_variance",
            "completed_iv": "origin_completed_iv",
        }
    ).select(
        "quote_date",
        "maturity_index",
        "moneyness_index",
        "origin_completed_total_variance",
        "origin_completed_iv",
    )

    joined_panel = (
        forecast_frame.join(
            actual_target,
            on=["target_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
        .join(
            origin_surface,
            on=["quote_date", "maturity_index", "moneyness_index"],
            how="left",
            validate="m:1",
        )
    )

    _require_non_null_columns(
        joined_panel,
        columns=(
            "actual_completed_total_variance",
            "actual_completed_iv",
            "origin_completed_iv",
            "predicted_total_variance",
        ),
    )
    _require_finite_float_columns(joined_panel, columns=("predicted_total_variance",))
    _require_non_negative_float_columns(joined_panel, columns=("predicted_total_variance",))

    panel = (
        panel_with_completed_iv(
            joined_panel,
            maturity_days_column="maturity_days",
            total_variance_column="predicted_total_variance",
            output_iv_column="predicted_iv",
            total_variance_floor=total_variance_floor,
        )
        .with_columns(
            (
                pl.col("actual_completed_iv") - pl.col("origin_completed_iv")
            ).alias("actual_iv_change"),
            (
                pl.col("predicted_iv") - pl.col("origin_completed_iv")
            ).alias("predicted_iv_change"),
            # Completed surfaces are the forecast target, while observed-mask x vega defines
            # the official observed-cell evaluation slice used for headline metrics.
            pl.when(pl.col("actual_observed_mask"))
            .then(pl.col("actual_vega_sum"))
            .otherwise(0.0)
            .alias("observed_weight"),
            pl.lit(1.0).alias("full_grid_weight"),
        )
        .sort(["model_name", "quote_date", "target_date", "maturity_index", "moneyness_index"])
    )

    _require_non_null_columns(
        panel,
        columns=(
            "predicted_iv",
            "predicted_iv_change",
        ),
    )
    _require_finite_float_columns(
        panel,
        columns=("predicted_total_variance", "predicted_iv", "predicted_iv_change"),
    )
    return panel


def panel_with_completed_iv(
    frame: pl.DataFrame,
    maturity_days_column: str,
    total_variance_column: str,
    output_iv_column: str,
    *,
    total_variance_floor: float,
) -> pl.DataFrame:
    """Add an IV column derived from total variance and maturity days."""

    maturity_years = (
        frame[maturity_days_column].to_numpy().astype(np.float64) / 365.0
    ).reshape(-1, 1)
    total_variance = validate_total_variance_array(
        frame[total_variance_column].to_numpy().astype(np.float64),
        context=f"Aligned panel column {total_variance_column}",
        allow_zero=True,
    ).reshape(-1, 1)
    iv = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
        total_variance_floor=total_variance_floor,
    ).reshape(-1)
    return frame.with_columns(pl.Series(output_iv_column, iv))

```

### src/ivsurf/evaluation/metrics.py

```python
"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = weights.sum()
    if total <= 0.0:
        return np.asarray(np.full_like(weights, 1.0 / weights.size, dtype=np.float64))
    return np.asarray(weights / total, dtype=np.float64)


def weighted_rmse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = _normalize_weights(weights.astype(np.float64, copy=False))
    return float(np.sqrt(np.sum(normalized * np.square(y_pred - y_true))))


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = _normalize_weights(weights.astype(np.float64, copy=False))
    return float(np.sum(normalized * np.abs(y_pred - y_true)))


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    normalized = weights.astype(np.float64, copy=False)
    total = normalized.sum()
    if total <= 0.0:
        return float(np.mean(np.square(y_true - y_pred)))
    normalized = normalized / total
    return float(np.sum(normalized * np.square(y_true - y_pred)))


def validate_total_variance_array(
    total_variance: np.ndarray,
    *,
    context: str,
    allow_zero: bool = True,
) -> np.ndarray:
    values = np.asarray(total_variance, dtype=np.float64)
    if not np.isfinite(values).all():
        message = f"{context} contains non-finite total variance values."
        raise ValueError(message)
    invalid = values < 0.0 if allow_zero else values <= 0.0
    if invalid.any():
        relation = "negative" if allow_zero else "non-positive"
        minimum = float(values[invalid].min())
        message = (
            f"{context} contains {relation} total variance values; "
            f"minimum_invalid_value={minimum}."
        )
        raise ValueError(message)
    return values


def qlike(y_true: np.ndarray, y_pred: np.ndarray, positive_floor: float) -> float:
    true_values = validate_total_variance_array(
        y_true,
        context="QLIKE y_true",
        allow_zero=True,
    )
    pred_values = validate_total_variance_array(
        y_pred,
        context="QLIKE y_pred",
        allow_zero=True,
    )
    true_clipped = np.maximum(true_values, positive_floor)
    pred_clipped = np.maximum(pred_values, positive_floor)
    return float(np.mean((true_clipped / pred_clipped) - np.log(true_clipped / pred_clipped) - 1.0))


def total_variance_to_iv(
    total_variance: np.ndarray,
    maturity_years: np.ndarray,
    *,
    total_variance_floor: float = 0.0,
) -> np.ndarray:
    validated_total_variance = validate_total_variance_array(
        total_variance,
        context="IV conversion input",
        allow_zero=True,
    )
    maturity = np.maximum(maturity_years, 1.0e-12)
    floored_total_variance = np.maximum(validated_total_variance, total_variance_floor)
    return np.asarray(np.sqrt(floored_total_variance / maturity), dtype=np.float64)

```

### src/ivsurf/evaluation/interpolation_sensitivity.py

```python
"""Interpolation-order sensitivity checks using saved observed surfaces only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import cast

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface
from ivsurf.surfaces.masks import reshape_mask_column, reshape_surface_column


@dataclass(frozen=True, slots=True)
class InterpolationSensitivityRow:
    """Difference between canonical and alternate interpolation orders."""

    quote_date: date
    observed_cell_count: int
    total_cell_count: int
    mean_abs_diff: float
    rmse_diff: float
    max_abs_diff: float


def build_interpolation_sensitivity_frame(
    actual_surface_frame: pl.DataFrame,
    *,
    grid: SurfaceGrid,
    surface_config: SurfaceGridConfig,
    alternate_order: tuple[str, ...],
    interpolation_cycles: int | None = None,
) -> pl.DataFrame:
    """Compare stored completed surfaces against a fixed alternate interpolation order."""

    cycles = (
        surface_config.interpolation_cycles
        if interpolation_cycles is None
        else interpolation_cycles
    )
    rows: list[InterpolationSensitivityRow] = []
    for group in actual_surface_frame.partition_by("quote_date", maintain_order=True):
        observed_surface = reshape_surface_column(
            group,
            grid,
            "observed_total_variance",
            null_fill=float("nan"),
        )
        observed_mask = reshape_mask_column(group, grid, "observed_mask")
        reference_surface = reshape_surface_column(group, grid, "completed_total_variance")
        alternate_surface = complete_surface(
            observed_total_variance=observed_surface,
            observed_mask=observed_mask,
            maturity_coordinates=grid.maturity_years,
            moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
            interpolation_order=alternate_order,
            interpolation_cycles=cycles,
            total_variance_floor=surface_config.total_variance_floor,
        ).completed_total_variance
        diff = alternate_surface - reference_surface
        rows.append(
            InterpolationSensitivityRow(
                quote_date=group["quote_date"][0],
                observed_cell_count=int(observed_mask.sum()),
                total_cell_count=int(observed_mask.size),
                mean_abs_diff=float(np.mean(np.abs(diff))),
                rmse_diff=float(np.sqrt(np.mean(np.square(diff)))),
                max_abs_diff=float(np.max(np.abs(diff))),
            )
        )
    return pl.DataFrame(rows).sort("quote_date")


def summarize_interpolation_sensitivity(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate interpolation sensitivity to one-row summary statistics."""

    mean_mean_abs_diff = float(cast(float, frame["mean_abs_diff"].mean()))
    mean_rmse_diff = float(cast(float, frame["rmse_diff"].mean()))
    max_max_abs_diff = float(cast(float, frame["max_abs_diff"].max()))
    return pl.DataFrame(
        {
            "mean_mean_abs_diff": [mean_mean_abs_diff],
            "mean_rmse_diff": [mean_rmse_diff],
            "max_max_abs_diff": [max_max_abs_diff],
            "n_quote_dates": [frame.height],
        }
    )

```

### src/ivsurf/evaluation/forecast_store.py

```python
"""Persist model forecasts to parquet."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from ivsurf.evaluation.metrics import validate_total_variance_array
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.surfaces.grid import SurfaceGrid


def _normalize_python_date(value: object) -> date:
    normalized = value.item() if isinstance(value, np.generic) else value
    if isinstance(normalized, datetime):
        return normalized.date()
    if isinstance(normalized, date):
        return normalized
    message = f"Expected forecast date value to be date-like, found {type(value).__name__}."
    raise TypeError(message)


def _normalize_python_dates(values: np.ndarray) -> list[date]:
    return [_normalize_python_date(value) for value in values]


def write_forecasts(
    output_path: Path,
    model_name: str,
    quote_dates: np.ndarray,
    target_dates: np.ndarray,
    predictions: np.ndarray,
    grid: SurfaceGrid,
) -> None:
    """Write long-form forecast artifacts."""

    surface_cell_count = len(grid.maturity_days) * len(grid.moneyness_points)
    if quote_dates.shape[0] != target_dates.shape[0]:
        message = "quote_dates and target_dates must contain the same number of forecast rows."
        raise ValueError(message)
    if predictions.ndim != 2:
        message = f"predictions must be rank-2, found ndim={predictions.ndim}."
        raise ValueError(message)
    expected_shape = (quote_dates.shape[0], surface_cell_count)
    if predictions.shape != expected_shape:
        message = (
            "predictions shape does not match the forecast-date count and grid size: "
            f"expected {expected_shape}, found {predictions.shape}."
        )
        raise ValueError(message)

    flat_predictions = validate_total_variance_array(
        predictions.reshape(-1),
        context=f"Forecast artifact for model {model_name}",
        allow_zero=True,
    )
    normalized_quote_dates = _normalize_python_dates(quote_dates)
    normalized_target_dates = _normalize_python_dates(target_dates)

    per_surface_maturity_index = np.repeat(
        np.arange(len(grid.maturity_days), dtype=np.int64),
        len(grid.moneyness_points),
    )
    per_surface_maturity_days = np.repeat(
        np.asarray(grid.maturity_days, dtype=np.int64),
        len(grid.moneyness_points),
    )
    per_surface_moneyness_index = np.tile(
        np.arange(len(grid.moneyness_points), dtype=np.int64),
        len(grid.maturity_days),
    )
    per_surface_moneyness_points = np.tile(
        np.asarray(grid.moneyness_points, dtype=np.float64),
        len(grid.maturity_days),
    )

    repeated_quote_dates = np.repeat(
        np.asarray(normalized_quote_dates, dtype=object),
        surface_cell_count,
    ).tolist()
    repeated_target_dates = np.repeat(
        np.asarray(normalized_target_dates, dtype=object),
        surface_cell_count,
    ).tolist()
    repeated_model_names = np.full(
        flat_predictions.shape[0],
        model_name,
        dtype=object,
    )
    frame = pl.DataFrame(
        {
            "model_name": repeated_model_names,
            "quote_date": pl.Series("quote_date", repeated_quote_dates, dtype=pl.Date),
            "target_date": pl.Series("target_date", repeated_target_dates, dtype=pl.Date),
            "maturity_index": np.tile(per_surface_maturity_index, quote_dates.shape[0]),
            "maturity_days": np.tile(per_surface_maturity_days, quote_dates.shape[0]),
            "moneyness_index": np.tile(per_surface_moneyness_index, quote_dates.shape[0]),
            "moneyness_point": np.tile(per_surface_moneyness_points, quote_dates.shape[0]),
            "predicted_total_variance": flat_predictions,
        },
    )
    frame = frame.cast(
        {
            "model_name": pl.String,
            "quote_date": pl.Date,
            "target_date": pl.Date,
            "maturity_index": pl.Int64,
            "maturity_days": pl.Int64,
            "moneyness_index": pl.Int64,
            "moneyness_point": pl.Float64,
            "predicted_total_variance": pl.Float64,
        }
    )
    write_parquet_frame(frame, output_path)

```

### scripts/01_ingest_cboe.py

```python
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import typer

from ivsurf.config import RawDataConfig, load_yaml_config
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.ingest_cboe import ingest_one_zip, list_raw_zip_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import sample_window_label
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    zip_paths = list_raw_zip_files(config)
    if limit is not None:
        zip_paths = zip_paths[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(config.manifests_dir, "01_ingest_cboe"),
        stage_name="01_ingest_cboe",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path],
            input_artifact_paths=zip_paths,
        ),
    )

    result_rows: list[dict[str, object]] = []
    with create_progress() as progress:
        for zip_path in iter_with_progress(
            progress,
            zip_paths,
            description="Stage 01 ingesting raw Cboe daily zips",
        ):
            item_id = str(zip_path.resolve())
            if resumer.item_complete(item_id):
                result_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            result = ingest_one_zip(zip_path=zip_path, config=config)
            result_row = {
                "source_zip": str(result.source_zip),
                "quote_date": result.quote_date.isoformat(),
                "status": result.status,
                "bronze_path": str(result.bronze_path) if result.bronze_path is not None else None,
                "row_count": result.row_count,
            }
            resumer.mark_complete(
                item_id,
                output_paths=[] if result.bronze_path is None else [result.bronze_path],
                metadata=result_row,
            )
            result_rows.append(result_row)
    written_results = [row for row in result_rows if row["status"] == "written"]
    written_output_paths = [
        Path(str(row["bronze_path"]))
        for row in written_results
        if row["bronze_path"] is not None
    ]
    skipped_results = [
        row for row in result_rows if row["status"] == "skipped_out_of_sample_window"
    ]
    payload = {
        "files_seen": len(result_rows),
        "files_written": len(written_results),
        "files_skipped_out_of_sample_window": len(skipped_results),
        "rows_parsed": sum(int(row["row_count"]) for row in result_rows),
        "rows_written": sum(int(row["row_count"]) for row in written_results),
        "sample_window": {
            "start_date": config.sample_start_date.isoformat(),
            "end_date": config.sample_end_date.isoformat(),
        },
        "resume_context_hash": resumer.context_hash,
        "results": result_rows,
    }
    manifest_path = config.manifests_dir / "bronze_ingestion_summary.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(manifest_path, orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    run_manifest_path = write_run_manifest(
        manifests_dir=config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="01_ingest_cboe",
        started_at=started_at,
        config_paths=[raw_config_path],
        input_artifact_paths=zip_paths,
        output_artifact_paths=[manifest_path, *written_output_paths],
        data_manifest_paths=zip_paths,
        extra_metadata={
            "limit": limit,
            "files_processed": len(result_rows),
            "files_written": len(written_results),
            "files_skipped_out_of_sample_window": len(skipped_results),
            "sample_window": sample_window_label(config),
            "resume_context_hash": resumer.context_hash,
            "source_zips": [str(zip_path) for zip_path in zip_paths],
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved bronze ingestion summary to {manifest_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()

```

### scripts/02_build_option_panel.py

```python
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import orjson
import polars as pl
import typer

from ivsurf.cleaning.derived_fields import add_derived_fields, build_tau_lookup
from ivsurf.cleaning.option_filters import apply_option_quality_flags
from ivsurf.config import (
    CleaningConfig,
    RawDataConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import require_quote_date_in_sample_window, sample_window_label
from ivsurf.qc.schema_checks import assert_required_columns
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path

app = typer.Typer(add_completion=False)


def _silver_path(bronze_path: Path, raw_config: RawDataConfig) -> Path:
    year_partition = bronze_path.parent.name
    output_dir = raw_config.silver_dir / year_partition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / bronze_path.name


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    cleaning_config_path: Path = Path("configs/data/cleaning.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    cleaning_config = CleaningConfig.model_validate(load_yaml_config(cleaning_config_path))
    calendar_config = calendar_config_from_raw(raw_config)

    bronze_files = sorted_artifact_files(raw_config.bronze_dir, "year=*/*.parquet")
    if limit is not None:
        bronze_files = bronze_files[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "02_build_option_panel"),
        stage_name="02_build_option_panel",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path, cleaning_config_path],
            input_artifact_paths=bronze_files,
        ),
    )

    summary_rows: list[dict[str, object]] = []
    with create_progress() as progress:
        for bronze_path in iter_with_progress(
            progress,
            bronze_files,
            description="Stage 02 building silver option panel",
        ):
            item_id = str(bronze_path.resolve())
            if resumer.item_complete(item_id):
                summary_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            frame = pl.read_parquet(bronze_path)
            assert_required_columns(
                frame.columns,
                required_columns=(
                    "quote_date",
                    "expiration",
                    "root",
                    "strike",
                    "option_type",
                    "bid_1545",
                    "ask_1545",
                    "vega_1545",
                    "active_underlying_price_1545",
                    "implied_volatility_1545",
                ),
                dataset_name=str(bronze_path),
            )
            quote_date = frame["quote_date"][0]
            if not isinstance(quote_date, date):
                message = f"Unexpected quote_date type in {bronze_path}"
                raise TypeError(message)
            require_quote_date_in_sample_window(
                quote_date,
                raw_config,
                context=f"Stage 02 bronze artifact {bronze_path}",
            )

            tau_lookup = build_tau_lookup(frame=frame, calendar_config=calendar_config)
            enriched = add_derived_fields(frame=frame, tau_lookup=tau_lookup)
            enriched = apply_option_quality_flags(frame=enriched, config=cleaning_config)
            summary_status = "built"

            output_path = _silver_path(bronze_path=bronze_path, raw_config=raw_config)
            write_parquet_frame(enriched, output_path)
            summary_row = {
                "silver_path": str(output_path),
                "quote_date": quote_date.isoformat(),
                "status": summary_status,
                "rows": enriched.height,
                "valid_rows": enriched.filter(pl.col("is_valid_observation")).height,
            }
            resumer.mark_complete(
                item_id,
                output_paths=[output_path],
                metadata=summary_row,
            )
            summary_rows.append(summary_row)

    summary_path = raw_config.manifests_dir / "silver_build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(summary_path, orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))
    silver_output_paths = [Path(str(row["silver_path"])) for row in summary_rows]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="02_build_option_panel",
        started_at=started_at,
        config_paths=[raw_config_path, cleaning_config_path],
        input_artifact_paths=[
            raw_config.manifests_dir / "bronze_ingestion_summary.json",
            *bronze_files,
        ],
        output_artifact_paths=[summary_path, *silver_output_paths],
        data_manifest_paths=bronze_files,
        extra_metadata={
            "limit": limit,
            "bronze_files_processed": len(bronze_files),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved silver build summary to {summary_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()

```

### scripts/03_build_surfaces.py

```python
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.cleaning.option_filters import valid_option_rows
from ivsurf.config import RawDataConfig, SurfaceGridConfig, load_yaml_config
from ivsurf.evaluation.metrics import total_variance_to_iv
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import require_quote_date_in_sample_window, sample_window_label
from ivsurf.qc.schema_checks import assert_required_columns
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.arbitrage_diagnostics import summarize_diagnostics
from ivsurf.surfaces.grid import SurfaceGrid, assign_grid_indices
from ivsurf.surfaces.interpolation import complete_surface

app = typer.Typer(add_completion=False)


def _gold_path(silver_path: Path, raw_config: RawDataConfig) -> Path:
    year_partition = silver_path.parent.name
    output_dir = raw_config.gold_dir / year_partition
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / silver_path.name


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    limit: int | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    silver_files = sorted_artifact_files(raw_config.silver_dir, "year=*/*.parquet")
    if limit is not None:
        silver_files = silver_files[:limit]
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "03_build_surfaces"),
        stage_name="03_build_surfaces",
        context_hash=build_resume_context_hash(
            config_paths=[raw_config_path, surface_config_path],
            input_artifact_paths=silver_files,
            extra_tokens={"artifact_schema_version": 2},
        ),
    )

    summary_rows: list[dict[str, object]] = []
    maturity_years = grid.maturity_years
    with create_progress() as progress:
        for silver_path in iter_with_progress(
            progress,
            silver_files,
            description="Stage 03 constructing gold surfaces",
        ):
            item_id = str(silver_path.resolve())
            if resumer.item_complete(item_id):
                summary_rows.append(resumer.metadata_for(item_id))
                continue
            stale_output_paths = resumer.output_paths_for(item_id)
            if stale_output_paths:
                resumer.clear_item(item_id, output_paths=stale_output_paths)
            silver_frame = pl.read_parquet(silver_path)
            assert_required_columns(
                silver_frame.columns,
                required_columns=(
                    "quote_date",
                    "tau_years",
                    "log_moneyness",
                    "total_variance",
                    "implied_volatility_1545",
                    "vega_1545",
                    "is_valid_observation",
                ),
                dataset_name=str(silver_path),
            )
            quote_date = silver_frame["quote_date"][0]
            if not isinstance(quote_date, date):
                message = f"Unexpected quote_date type in {silver_path}"
                raise TypeError(message)
            require_quote_date_in_sample_window(
                quote_date,
                raw_config,
                context=f"Stage 03 silver artifact {silver_path}",
            )
            valid_frame = valid_option_rows(silver_frame)
            if valid_frame.is_empty():
                summary_row = {
                    "silver_path": str(silver_path),
                    "quote_date": quote_date.isoformat(),
                    "status": "skipped_no_valid_rows",
                    "reason": "NO_VALID_ROWS_AFTER_CLEANING",
                }
                resumer.mark_complete(
                    item_id,
                    output_paths=[],
                    metadata=summary_row,
                )
                summary_rows.append(summary_row)
                continue

            assigned = assign_grid_indices(frame=valid_frame, grid=grid)
            observed = aggregate_daily_surface(
                frame=assigned,
                grid=grid,
                config=surface_config,
            ).sort(["quote_date", "maturity_index", "moneyness_index"])
            observed_matrix = (
                observed["observed_total_variance"]
                .fill_null(np.nan)
                .to_numpy()
                .reshape(grid.shape)
            )
            observed_mask = observed["observed_mask"].to_numpy().reshape(grid.shape)
            completed = complete_surface(
                observed_total_variance=observed_matrix,
                observed_mask=observed_mask,
                maturity_coordinates=maturity_years,
                moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
                interpolation_order=surface_config.interpolation_order,
                interpolation_cycles=surface_config.interpolation_cycles,
                total_variance_floor=surface_config.total_variance_floor,
            )
            diagnostics = summarize_diagnostics(
                completed.completed_total_variance,
                moneyness_points=np.asarray(grid.moneyness_points, dtype=np.float64),
            )
            completed_flat = completed.completed_total_variance.reshape(-1)
            completed_iv = total_variance_to_iv(
                total_variance=completed.completed_total_variance,
                maturity_years=maturity_years[:, None],
            ).reshape(-1)

            output_frame = observed.with_columns(
                pl.Series("completed_total_variance", completed_flat),
                pl.Series("completed_iv", completed_iv),
                pl.lit(diagnostics.calendar_violation_count).alias("calendar_violation_count"),
                pl.lit(diagnostics.calendar_violation_magnitude).alias(
                    "calendar_violation_magnitude"
                ),
                pl.lit(diagnostics.convexity_violation_count).alias("convexity_violation_count"),
                pl.lit(diagnostics.convexity_violation_magnitude).alias(
                    "convexity_violation_magnitude"
                ),
            )
            output_path = _gold_path(silver_path=silver_path, raw_config=raw_config)
            write_parquet_frame(output_frame, output_path)
            summary_row = {
                "gold_path": str(output_path),
                "quote_date": str(output_frame["quote_date"][0]),
                "status": "built",
                "observed_cells": int(output_frame["observed_mask"].sum()),
            }
            resumer.mark_complete(
                item_id,
                output_paths=[output_path],
                metadata=summary_row,
            )
            summary_rows.append(summary_row)

    summary_path = raw_config.manifests_dir / "gold_surface_summary.json"
    skipped_dates_path = raw_config.manifests_dir / "gold_surface_skipped_dates.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(summary_path, orjson.dumps(summary_rows, option=orjson.OPT_INDENT_2))
    skipped_date_rows = [
        {
            "quote_date": row["quote_date"],
            "reason": row["reason"],
            "silver_path": row["silver_path"],
        }
        for row in summary_rows
        if row.get("status") == "skipped_no_valid_rows"
    ]
    write_bytes_atomic(
        skipped_dates_path,
        orjson.dumps(skipped_date_rows, option=orjson.OPT_INDENT_2),
    )
    gold_output_paths = [
        Path(str(row["gold_path"]))
        for row in summary_rows
        if row.get("status") == "built" and "gold_path" in row
    ]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="03_build_surfaces",
        started_at=started_at,
        config_paths=[raw_config_path, surface_config_path],
        input_artifact_paths=[
            raw_config.manifests_dir / "silver_build_summary.json",
            *silver_files,
        ],
        output_artifact_paths=[summary_path, skipped_dates_path, *gold_output_paths],
        data_manifest_paths=silver_files,
        extra_metadata={
            "limit": limit,
            "silver_files_processed": len(silver_files),
            "gold_files_written": len(gold_output_paths),
            "skipped_dates_count": len(skipped_date_rows),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved gold surface summary to {summary_path}")
    typer.echo(f"Saved skipped-date manifest to {skipped_dates_path}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()

```

### scripts/04_build_features.py

```python
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import typer

from ivsurf.config import (
    FeatureConfig,
    RawDataConfig,
    SurfaceGridConfig,
    WalkforwardConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivsurf.exceptions import DataValidationError
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.io.parquet import scan_parquet_files, write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress, iter_with_progress
from ivsurf.qc.sample_window import (
    assert_frame_dates_in_sample_window,
    sample_window_expr,
    sample_window_label,
)
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.splits.manifests import serialize_splits
from ivsurf.splits.walkforward import build_walkforward_splits
from ivsurf.surfaces.grid import SurfaceGrid

app = typer.Typer(add_completion=False)


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    surface_config_path: Path = Path("configs/data/surface.yaml"),
    feature_config_path: Path = Path("configs/data/features.yaml"),
    walkforward_config_path: Path = Path("configs/eval/walkforward.yaml"),
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    surface_config = SurfaceGridConfig.model_validate(load_yaml_config(surface_config_path))
    feature_config = FeatureConfig.model_validate(load_yaml_config(feature_config_path))
    calendar_config = calendar_config_from_raw(raw_config)
    walkforward_config = WalkforwardConfig.model_validate(load_yaml_config(walkforward_config_path))
    grid = SurfaceGrid.from_config(surface_config)

    gold_files = sorted_artifact_files(raw_config.gold_dir, "year=*/*.parquet")
    if not gold_files:
        message = "No gold surface files found. Run scripts/03_build_surfaces.py first."
        raise FileNotFoundError(message)
    output_path = raw_config.gold_dir / "daily_features.parquet"
    split_manifest_path = raw_config.manifests_dir / "walkforward_splits.json"
    skipped_dates_manifest_path = raw_config.manifests_dir / "gold_surface_skipped_dates.json"
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "04_build_features"),
        stage_name="04_build_features",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                surface_config_path,
                feature_config_path,
                walkforward_config_path,
            ],
            input_artifact_paths=gold_files,
            extra_tokens={"artifact_schema_version": 2},
        ),
    )
    resume_item_id = "daily_feature_dataset"
    if resumer.item_complete(
        resume_item_id,
        required_output_paths=[output_path, split_manifest_path],
    ):
        typer.echo(
            "Stage 04 resume: existing daily_features.parquet and walkforward_splits.json "
            "match the current context; skipping rebuild."
        )
        return
    resumer.clear_item(resume_item_id, output_paths=[output_path, split_manifest_path])
    with create_progress() as progress:
        for _gold_path in iter_with_progress(
            progress,
            gold_files,
            description="Stage 04 validating gold surface scope",
        ):
            continue
    gold_scan = scan_parquet_files(gold_files)
    out_of_window_dates = (
        gold_scan.select("quote_date")
        .filter(~sample_window_expr(raw_config, column="quote_date"))
        .unique()
        .sort("quote_date")
        .collect(engine="streaming")
    )
    if not out_of_window_dates.is_empty():
        offending_dates = ", ".join(
            value.isoformat()
            for value in out_of_window_dates["quote_date"].to_list()
            if isinstance(value, date)
        )
        message = (
            "Stage 04 found gold surfaces outside the configured sample window "
            f"{sample_window_label(raw_config)}: {offending_dates}."
        )
        raise DataValidationError(message)
    surface_frame = (
        gold_scan.filter(sample_window_expr(raw_config, column="quote_date"))
        .collect(engine="streaming")
        .sort(["quote_date", "maturity_index", "moneyness_index"])
    )
    if surface_frame.is_empty():
        message = (
            "No in-window gold surfaces remain after applying the configured sample window "
            f"{sample_window_label(raw_config)}."
        )
        raise DataValidationError(message)

    daily_dataset = build_daily_feature_dataset(
        surface_frame=surface_frame,
        grid=grid,
        feature_config=feature_config,
        calendar_config=calendar_config,
    ).feature_frame
    assert_frame_dates_in_sample_window(
        daily_dataset,
        raw_config,
        context="Stage 04 daily feature dataset",
    )
    assert_frame_dates_in_sample_window(
        daily_dataset,
        raw_config,
        column="target_date",
        context="Stage 04 daily feature dataset",
    )
    write_parquet_frame(daily_dataset, output_path)

    dates = daily_dataset["quote_date"].to_list()
    if any(not isinstance(value, date) for value in dates):
        message = "daily_features.parquet must contain Polars Date quote_date values."
        raise TypeError(message)
    split_hash = serialize_splits(
        splits=build_walkforward_splits(dates=dates, config=walkforward_config),
        output_path=split_manifest_path,
    )
    resumer.mark_complete(
        resume_item_id,
        output_paths=[output_path, split_manifest_path],
        metadata={
            "feature_rows": daily_dataset.height,
            "split_hash": split_hash,
        },
    )
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="04_build_features",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            surface_config_path,
            feature_config_path,
            walkforward_config_path,
        ],
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            skipped_dates_manifest_path,
            *gold_files,
        ],
        output_artifact_paths=[output_path, split_manifest_path],
        data_manifest_paths=gold_files,
        split_manifest_path=split_manifest_path,
        extra_metadata={
            "feature_rows": daily_dataset.height,
            "split_hash": split_hash,
            "gold_input_file_count": len(gold_files),
            "sample_window": sample_window_label(raw_config),
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved feature dataset to {output_path} and split hash {split_hash}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()

```

### scripts/07_run_stats.py

```python
from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import orjson
import polars as pl
import typer

from ivsurf.config import (
    EvaluationMetricsConfig,
    HpoProfileConfig,
    RawDataConfig,
    StatsTestConfig,
    TrainingProfileConfig,
    load_yaml_config,
)
from ivsurf.evaluation.alignment import (
    assert_forecast_origins_after_hpo_boundary,
    build_forecast_realization_panel,
    load_actual_surface_frame,
    load_forecast_frame,
)
from ivsurf.evaluation.loss_panels import build_daily_loss_frame
from ivsurf.io.atomic import write_bytes_atomic
from ivsurf.io.parquet import write_parquet_frame
from ivsurf.io.paths import sorted_artifact_files
from ivsurf.progress import create_progress
from ivsurf.reproducibility import write_run_manifest
from ivsurf.resume import StageResumer, build_resume_context_hash, resume_state_path
from ivsurf.stats.diebold_mariano import diebold_mariano_test
from ivsurf.stats.mcs import model_confidence_set
from ivsurf.stats.spa import superior_predictive_ability_test
from ivsurf.training.model_factory import TUNABLE_MODEL_NAMES
from ivsurf.training.tuning import (
    load_required_tuning_results,
    require_consistent_clean_evaluation_policy,
    require_matching_primary_loss_metric,
)
from ivsurf.workflow import resolve_workflow_run_paths

app = typer.Typer(add_completion=False)


def _daily_loss_matrix(
    loss_frame: pl.DataFrame,
    metric_column: str,
) -> tuple[np.ndarray, tuple[str, ...], tuple[object, ...]]:
    pivoted = (
        loss_frame.select("model_name", "target_date", metric_column)
        .pivot(on="model_name", index="target_date", values=metric_column)
        .sort("target_date")
    )
    model_columns = tuple(column for column in pivoted.columns if column != "target_date")
    matrix = pivoted.select(model_columns).to_numpy().astype(np.float64)
    return matrix, model_columns, tuple(pivoted["target_date"].to_list())


def _loss_summary_frame(
    daily_loss_frame: pl.DataFrame,
    *,
    loss_metrics: tuple[str, ...],
) -> pl.DataFrame:
    aggregation_expressions: list[pl.Expr] = []
    for metric_column in loss_metrics:
        aggregation_expressions.extend(
            (
                pl.col(metric_column).mean().alias(f"mean_{metric_column}"),
                pl.col(metric_column).std(ddof=1).alias(f"std_{metric_column}"),
            )
        )
    return (
        daily_loss_frame.group_by("model_name")
        .agg(*aggregation_expressions, pl.len().alias("n_target_dates"))
        .sort(f"mean_{loss_metrics[0]}")
    )


@app.command()
def main(
    raw_config_path: Path = Path("configs/data/raw.yaml"),
    metrics_config_path: Path = Path("configs/eval/metrics.yaml"),
    stats_config_path: Path = Path("configs/eval/stats_tests.yaml"),
    hpo_profile_config_path: Path = Path("configs/workflow/hpo_30_trials.yaml"),
    training_profile_config_path: Path = Path("configs/workflow/train_30_epochs.yaml"),
    run_profile_name: str | None = None,
    mlflow_tracking_uri: str | None = None,
    mlflow_experiment_name: str = "ivsurf",
) -> None:
    started_at = datetime.now(UTC)
    raw_config = RawDataConfig.model_validate(load_yaml_config(raw_config_path))
    hpo_profile = HpoProfileConfig.model_validate(load_yaml_config(hpo_profile_config_path))
    training_profile = TrainingProfileConfig.model_validate(
        load_yaml_config(training_profile_config_path)
    )
    metrics_config = EvaluationMetricsConfig.model_validate(load_yaml_config(metrics_config_path))
    stats_config = StatsTestConfig.model_validate(load_yaml_config(stats_config_path))
    workflow_paths = resolve_workflow_run_paths(
        raw_config,
        hpo_profile_name=hpo_profile.profile_name,
        training_profile_name=training_profile.profile_name,
        run_profile_name=run_profile_name,
    )

    output_dir = workflow_paths.stats_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_paths = sorted_artifact_files(workflow_paths.forecast_dir, "*.parquet")
    forecast_reuse_manifest_paths = []
    if run_profile_name is not None:
        forecast_reuse_manifest_path = (
            raw_config.manifests_dir / "forecast_profile_reuse" / f"{run_profile_name}.json"
        )
        if forecast_reuse_manifest_path.exists():
            forecast_reuse_manifest_paths.append(forecast_reuse_manifest_path)
    tuning_manifest_paths = [
        raw_config.manifests_dir / "tuning" / hpo_profile.profile_name / f"{model_name}.json"
        for model_name in TUNABLE_MODEL_NAMES
    ]
    panel_path = output_dir / "forecast_realization_panel.parquet"
    daily_loss_path = output_dir / "daily_loss_frame.parquet"
    dm_results_path = output_dir / "dm_results.json"
    spa_result_path = output_dir / "spa_result.json"
    mcs_result_path = output_dir / "mcs_result.json"
    loss_summary_path = output_dir / "loss_summary.parquet"
    resumer = StageResumer(
        state_path=resume_state_path(raw_config.manifests_dir, "07_run_stats"),
        stage_name="07_run_stats",
        context_hash=build_resume_context_hash(
            config_paths=[
                raw_config_path,
                metrics_config_path,
                stats_config_path,
                hpo_profile_config_path,
                training_profile_config_path,
            ],
            input_artifact_paths=[
                raw_config.manifests_dir / "gold_surface_summary.json",
                *forecast_reuse_manifest_paths,
                *tuning_manifest_paths,
                *forecast_paths,
            ],
            extra_tokens={
                "run_profile_name": run_profile_name,
                "workflow_run_label": workflow_paths.run_label,
                "artifact_schema_version": 2,
            },
        ),
    )
    tuning_results = load_required_tuning_results(
        raw_config.manifests_dir,
        hpo_profile_name=hpo_profile.profile_name,
        model_names=TUNABLE_MODEL_NAMES,
    )
    require_matching_primary_loss_metric(
        tuning_results.values(),
        expected_primary_loss_metric=metrics_config.primary_loss_metric,
    )
    clean_evaluation_policy = require_consistent_clean_evaluation_policy(tuning_results.values())

    with create_progress() as progress:
        task_id = progress.add_task("Stage 07 statistical evaluation", total=None)

        if resumer.item_complete("forecast_realization_panel", required_output_paths=[panel_path]):
            progress.update(task_id, description="Stage 07 resume: loading saved panel")
            panel = pl.read_parquet(panel_path)
        else:
            resumer.clear_item("forecast_realization_panel", output_paths=[panel_path])
            progress.update(task_id, description="Stage 07 loading realized surfaces and forecasts")
            actual_surface_frame = load_actual_surface_frame(raw_config.gold_dir)
            forecast_frame = load_forecast_frame(workflow_paths.forecast_dir)
            assert_forecast_origins_after_hpo_boundary(
                forecast_frame,
                max_hpo_validation_date=clean_evaluation_policy.max_hpo_validation_date,
            )
            panel = build_forecast_realization_panel(
                actual_surface_frame=actual_surface_frame,
                forecast_frame=forecast_frame,
                total_variance_floor=metrics_config.positive_floor,
            )
            write_parquet_frame(panel, panel_path)
            resumer.mark_complete(
                "forecast_realization_panel",
                output_paths=[panel_path],
                metadata={"rows": panel.height},
            )

        if resumer.item_complete("daily_loss_frame", required_output_paths=[daily_loss_path]):
            progress.update(task_id, description="Stage 07 resume: loading saved daily loss frame")
            daily_loss_frame = pl.read_parquet(daily_loss_path)
        else:
            resumer.clear_item("daily_loss_frame", output_paths=[daily_loss_path])
            progress.update(task_id, description="Stage 07 computing daily loss panel")
            daily_loss_frame = build_daily_loss_frame(
                panel=panel,
                positive_floor=metrics_config.positive_floor,
                full_grid_weighting=stats_config.full_grid_weighting,
            )
            write_parquet_frame(daily_loss_frame, daily_loss_path)
            resumer.mark_complete(
                "daily_loss_frame",
                output_paths=[daily_loss_path],
                metadata={"rows": daily_loss_frame.height},
            )

        primary_metric_column = stats_config.loss_metrics[0]
        model_columns = _daily_loss_matrix(
            loss_frame=daily_loss_frame,
            metric_column=primary_metric_column,
        )[1]
        benchmark_model = stats_config.benchmark_model
        if benchmark_model not in model_columns:
            message = f"Benchmark model {benchmark_model} not found in loss frame."
            raise ValueError(message)

        dm_model_count = max(len(model_columns) - 1, 0) * len(stats_config.loss_metrics)
        progress.update(
            task_id,
            total=4 + dm_model_count + (2 * len(stats_config.loss_metrics)),
            completed=2,
            description="Stage 07 running Diebold-Mariano tests",
        )
        if not resumer.item_complete("dm_results", required_output_paths=[dm_results_path]):
            resumer.clear_item("dm_results", output_paths=[dm_results_path])
            dm_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                if metric_model_columns != model_columns:
                    message = (
                        "All loss metrics must produce the same model ordering in the daily "
                        f"loss frame, found {metric_model_columns!r} != {model_columns!r} "
                        f"for {metric_column}."
                    )
                    raise ValueError(message)
                benchmark_index = metric_model_columns.index(benchmark_model)
                benchmark_losses = loss_matrix[:, benchmark_index]
                for model_name in metric_model_columns:
                    if model_name == benchmark_model:
                        continue
                    model_index = metric_model_columns.index(model_name)
                    result_row = asdict(
                        diebold_mariano_test(
                            loss_a=benchmark_losses,
                            loss_b=loss_matrix[:, model_index],
                            model_a=benchmark_model,
                            model_b=model_name,
                            alternative=stats_config.dm_alternative,
                            max_lag=stats_config.dm_max_lag,
                        )
                    )
                    result_row["loss_metric"] = metric_column
                    dm_results.append(result_row)
                    progress.advance(task_id)
            write_bytes_atomic(
                dm_results_path,
                orjson.dumps(dm_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "dm_results",
                output_paths=[dm_results_path],
                metadata={
                    "model_count": len(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                },
            )
        else:
            progress.advance(task_id, advance=dm_model_count)

        progress.update(task_id, description="Stage 07 running SPA bootstrap")
        if not resumer.item_complete("spa_result", required_output_paths=[spa_result_path]):
            resumer.clear_item("spa_result", output_paths=[spa_result_path])
            spa_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                candidate_models = tuple(
                    model for model in metric_model_columns if model != benchmark_model
                )
                candidate_losses = np.column_stack(
                    [
                        loss_matrix[:, metric_model_columns.index(model)]
                        for model in candidate_models
                    ]
                )
                benchmark_index = metric_model_columns.index(benchmark_model)
                benchmark_losses = loss_matrix[:, benchmark_index]
                spa_result = asdict(
                    superior_predictive_ability_test(
                        benchmark_losses=benchmark_losses,
                        candidate_losses=candidate_losses,
                        benchmark_model=benchmark_model,
                        candidate_models=candidate_models,
                        alpha=stats_config.spa_alpha,
                        block_size=stats_config.spa_block_size,
                        bootstrap_reps=stats_config.spa_bootstrap_reps,
                        seed=stats_config.bootstrap_seed,
                    )
                )
                spa_result["loss_metric"] = metric_column
                spa_results.append(spa_result)
            write_bytes_atomic(
                spa_result_path,
                orjson.dumps(spa_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "spa_result",
                output_paths=[spa_result_path],
                metadata={"loss_metrics": list(stats_config.loss_metrics)},
            )
        progress.advance(task_id, advance=len(stats_config.loss_metrics))

        progress.update(task_id, description="Stage 07 running simplified Tmax bootstrap")
        if not resumer.item_complete("mcs_result", required_output_paths=[mcs_result_path]):
            resumer.clear_item("mcs_result", output_paths=[mcs_result_path])
            mcs_results: list[dict[str, object]] = []
            for metric_column in stats_config.loss_metrics:
                loss_matrix, metric_model_columns, _ = _daily_loss_matrix(
                    loss_frame=daily_loss_frame,
                    metric_column=metric_column,
                )
                mcs_result = asdict(
                    model_confidence_set(
                        losses=loss_matrix,
                        model_names=metric_model_columns,
                        alpha=stats_config.mcs_alpha,
                        block_size=stats_config.mcs_block_size,
                        bootstrap_reps=stats_config.mcs_bootstrap_reps,
                        seed=stats_config.bootstrap_seed,
                    )
                )
                mcs_result["loss_metric"] = metric_column
                mcs_results.append(mcs_result)
            write_bytes_atomic(
                mcs_result_path,
                orjson.dumps(mcs_results, option=orjson.OPT_INDENT_2),
            )
            resumer.mark_complete(
                "mcs_result",
                output_paths=[mcs_result_path],
                metadata={
                    "model_names": list(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                    "procedure_name": "simplified_tmax_elimination",
                },
            )
        progress.advance(task_id, advance=len(stats_config.loss_metrics))

        progress.update(task_id, description="Stage 07 writing summary tables")
        if not resumer.item_complete("loss_summary", required_output_paths=[loss_summary_path]):
            resumer.clear_item("loss_summary", output_paths=[loss_summary_path])
            summary_frame = _loss_summary_frame(
                daily_loss_frame,
                loss_metrics=stats_config.loss_metrics,
            )
            write_parquet_frame(summary_frame, loss_summary_path)
            resumer.mark_complete(
                "loss_summary",
                output_paths=[loss_summary_path],
                metadata={
                    "model_count": len(model_columns),
                    "loss_metrics": list(stats_config.loss_metrics),
                },
            )
        progress.advance(task_id)

    output_paths = [
        panel_path,
        daily_loss_path,
        dm_results_path,
        spa_result_path,
        mcs_result_path,
        loss_summary_path,
    ]
    run_manifest_path = write_run_manifest(
        manifests_dir=raw_config.manifests_dir,
        repo_root=Path.cwd(),
        script_name="07_run_stats",
        started_at=started_at,
        config_paths=[
            raw_config_path,
            metrics_config_path,
            stats_config_path,
            hpo_profile_config_path,
            training_profile_config_path,
        ],
        input_artifact_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_reuse_manifest_paths,
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        output_artifact_paths=output_paths,
        data_manifest_paths=[
            raw_config.manifests_dir / "gold_surface_summary.json",
            *forecast_reuse_manifest_paths,
            *tuning_manifest_paths,
            *forecast_paths,
        ],
        random_seed=stats_config.bootstrap_seed,
        extra_metadata={
            "loss_metrics": list(stats_config.loss_metrics),
            "benchmark_model": benchmark_model,
            "mcs_procedure_name": "simplified_tmax_elimination",
            "max_hpo_validation_date": clean_evaluation_policy.max_hpo_validation_date.isoformat(),
            "first_clean_test_split_id": clean_evaluation_policy.first_clean_test_split_id,
            "run_profile_name": run_profile_name,
            "workflow_run_label": workflow_paths.run_label,
            "resume_context_hash": resumer.context_hash,
        },
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )
    typer.echo(f"Saved stats outputs to {output_dir}")
    typer.echo(f"Saved run manifest to {run_manifest_path}")


if __name__ == "__main__":
    app()

```

## CONFIG_AND_TEST_FILES


### configs/data/surface.yaml

```yaml
moneyness_points:
  - -0.30
  - -0.20
  - -0.10
  - -0.05
  - 0.00
  - 0.05
  - 0.10
  - 0.20
  - 0.30
maturity_days:
  - 1
  - 7
  - 14
  - 30
  - 60
  - 90
  - 180
  - 365
  - 730
interpolation_order:
  - "maturity"
  - "moneyness"
interpolation_cycles: 2
total_variance_floor: 1.0e-8
observed_cell_min_count: 1


```

### configs/data/features.yaml

```yaml
lag_windows:
  - 1
  - 5
  - 22
include_daily_change: true
include_mask: true
include_liquidity: true


```

### configs/eval/metrics.yaml

```yaml
positive_floor: 1.0e-8
primary_loss_metric: "observed_mse_total_variance"

```

### configs/eval/walkforward.yaml

```yaml
train_size: 504
validation_size: 126
test_size: 21
step_size: 21
expanding_train: true


```

### configs/models/neural_surface.yaml

```yaml
model_name: "neural_surface"
hidden_width: 256
depth: 3
dropout: 0.10
learning_rate: 0.001
weight_decay: 0.0001
epochs: 80
batch_size: 64
seed: 7
observed_loss_weight: 1.0
imputed_loss_weight: 0.25
calendar_penalty_weight: 0.05
convexity_penalty_weight: 0.05
roughness_penalty_weight: 0.005
output_total_variance_floor: 1.0e-8
device: "cuda"

```

### configs/models/har_factor.yaml

```yaml
model_name: "har_factor"
n_factors: 5
alpha: 1.0


```

### configs/models/ridge.yaml

```yaml
model_name: "ridge"
alpha: 1.0

```

### configs/models/elasticnet.yaml

```yaml
model_name: "elasticnet"
alpha: 0.01
l1_ratio: 0.1
max_iter: 50000
tol: 0.0001

```

### tests/unit/test_option_filters.py

```python
from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.cleaning.option_filters import apply_option_quality_flags
from ivsurf.config import CleaningConfig


def test_zero_iv_is_flagged_explicitly() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4)],
            "option_type": ["C"],
            "bid_1545": [1.0],
            "ask_1545": [1.2],
            "implied_volatility_1545": [0.0],
            "vega_1545": [1.0],
            "active_underlying_price_1545": [100.0],
            "mid_1545": [1.1],
            "tau_years": [0.1],
            "log_moneyness": [0.0],
        }
    )
    flagged = apply_option_quality_flags(frame=frame, config=CleaningConfig())
    assert flagged["invalid_reason"][0] == "NON_POSITIVE_IV"
    assert flagged["is_valid_observation"][0] is False


def test_cleaning_threshold_fields_are_explicitly_exclusive() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "option_type": ["C", "C"],
            "bid_1545": [0.0, 0.1],
            "ask_1545": [0.0, 0.2],
            "implied_volatility_1545": [0.2, 0.2],
            "vega_1545": [1.0, 1.0],
            "active_underlying_price_1545": [100.0, 100.0],
            "mid_1545": [0.0, 0.15],
            "tau_years": [0.1, 0.1],
            "log_moneyness": [0.0, 0.0],
        }
    )
    flagged = apply_option_quality_flags(
        frame=frame,
        config=CleaningConfig(
            min_valid_bid_exclusive=0.0,
            min_valid_ask_exclusive=0.0,
            min_valid_mid_price_exclusive=0.0,
        ),
    )

    assert flagged["invalid_reason"].to_list() == ["NON_POSITIVE_BID", None]
    assert flagged["is_valid_observation"].to_list() == [False, True]

```

### tests/unit/test_interpolation.py

```python
from __future__ import annotations

import numpy as np

from ivsurf.surfaces.interpolation import complete_surface


def test_surface_completion_fills_all_cells() -> None:
    observed = np.asarray([[0.04, np.nan, 0.09], [np.nan, 0.07, np.nan]], dtype=np.float64)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=np.isfinite(observed),
        maturity_coordinates=np.asarray([7.0 / 365.0, 30.0 / 365.0]),
        moneyness_coordinates=np.asarray([-0.1, 0.0, 0.1]),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    )
    assert np.isfinite(completed.completed_total_variance).all()
    assert completed.completed_total_variance[0, 0] == observed[0, 0]

```

### tests/unit/test_metrics.py

```python
from __future__ import annotations

import numpy as np
import pytest

from ivsurf.evaluation.metrics import total_variance_to_iv, weighted_mse


def test_weighted_mse_matches_manual_weighted_average() -> None:
    y_true = np.asarray([1.0, 3.0, 5.0], dtype=np.float64)
    y_pred = np.asarray([2.0, 1.0, 6.0], dtype=np.float64)
    weights = np.asarray([1.0, 2.0, 1.0], dtype=np.float64)

    result = weighted_mse(y_true, y_pred, weights)

    expected = ((1.0 * 1.0) + (2.0 * 4.0) + (1.0 * 1.0)) / 4.0
    assert result == expected


def test_total_variance_to_iv_rejects_negative_inputs() -> None:
    total_variance = np.asarray([[-1.0e-4], [4.0e-4]], dtype=np.float64)
    maturity_years = np.asarray([[30.0 / 365.0], [30.0 / 365.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="negative total variance"):
        total_variance_to_iv(
            total_variance=total_variance,
            maturity_years=maturity_years,
            total_variance_floor=1.0e-8,
        )


def test_total_variance_to_iv_floors_small_non_negative_inputs_for_metric_stability() -> None:
    total_variance = np.asarray([[0.0], [4.0e-4]], dtype=np.float64)
    maturity_years = np.asarray([[30.0 / 365.0], [30.0 / 365.0]], dtype=np.float64)

    result = total_variance_to_iv(
        total_variance=total_variance,
        maturity_years=maturity_years,
        total_variance_floor=1.0e-8,
    ).reshape(-1)

    assert np.isfinite(result).all()
    assert result[0] > 0.0

```

### tests/unit/test_alignment.py

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from ivsurf.evaluation.alignment import build_forecast_realization_panel, load_daily_spot_frame


def _actual_surface_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for quote_date, sigma in ((date(2021, 1, 4), 0.20), (date(2021, 1, 5), 0.22)):
        for maturity_index, maturity_days in enumerate((30, 90)):
            maturity_years = maturity_days / 365.0
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                total_variance = (sigma * sigma) * maturity_years
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                        "observed_total_variance": total_variance,
                        "observed_iv": sigma,
                        "completed_total_variance": total_variance,
                        "completed_iv": sigma,
                        "observed_mask": True,
                        "vega_sum": 1.0,
                    }
                )
    return pl.DataFrame(rows)


def test_build_forecast_realization_panel_rejects_negative_predicted_total_variance() -> None:
    forecast_rows = []
    for maturity_index, maturity_days in enumerate((30, 90)):
        for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
            forecast_rows.append(
                {
                    "model_name": "ridge",
                    "quote_date": date(2021, 1, 4),
                    "target_date": date(2021, 1, 5),
                    "maturity_index": maturity_index,
                    "maturity_days": maturity_days,
                    "moneyness_index": moneyness_index,
                    "moneyness_point": moneyness_point,
                    "predicted_total_variance": -1.0e-4 if maturity_index == 0 else 5.0e-3,
                }
            )
    with pytest.raises(ValueError, match="negative total variance"):
        build_forecast_realization_panel(
            actual_surface_frame=_actual_surface_frame(),
            forecast_frame=pl.DataFrame(forecast_rows),
            total_variance_floor=1.0e-8,
        )


def test_load_daily_spot_frame_uses_active_underlying_price_when_bid_ask_are_zero(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 11),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 99.0,
                "is_valid_observation": True,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.0,
        },
        {
            "quote_date": date(2020, 8, 11),
            "spot_1545": 99.0,
        },
    ]


def test_load_daily_spot_frame_uses_valid_row_median_when_spot_varies_across_option_rows(
    tmp_path: Path,
) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.0,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.1,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 97.2,
                "is_valid_observation": True,
            },
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 140.0,
                "is_valid_observation": False,
            },
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    spots = load_daily_spot_frame(tmp_path / "silver")

    assert spots.to_dicts() == [
        {
            "quote_date": date(2020, 8, 10),
            "spot_1545": 97.1,
        }
    ]


def test_load_daily_spot_frame_rejects_nonpositive_active_underlying_price(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver" / "year=2020"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "quote_date": date(2020, 8, 10),
                "underlying_bid_1545": 0.0,
                "underlying_ask_1545": 0.0,
                "active_underlying_price_1545": 0.0,
                "is_valid_observation": True,
            }
        ]
    ).write_parquet(silver_dir / "spots.parquet")

    with pytest.raises(
        ValueError,
        match="strictly positive finite stage-08 daily spot values derived from the median",
    ):
        load_daily_spot_frame(tmp_path / "silver")

```

### tests/unit/test_feature_dataset.py

```python
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from ivsurf.config import FeatureConfig, MarketCalendarConfig, SurfaceGridConfig
from ivsurf.features.tabular_dataset import (
    build_daily_feature_dataset,
    build_target_training_weights,
)
from ivsurf.surfaces.grid import SurfaceGrid


def _surface_frame(quote_dates: list[date]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for offset, quote_date in enumerate(quote_dates, start=1):
        rows.append(
            {
                "quote_date": quote_date,
                "maturity_index": 0,
                "maturity_days": 30,
                "moneyness_index": 0,
                "moneyness_point": 0.0,
                "observed_total_variance": 0.01 * offset,
                "completed_total_variance": 0.01 * offset,
                "observed_mask": True,
                "vega_sum": 1.0,
                "observation_count": 1,
                "weighted_spread_1545": 0.01,
            }
        )
    return pl.DataFrame(rows)


def test_build_daily_feature_dataset_requires_minimum_history() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    surface_frame = _surface_frame([date(2021, 1, 4)])

    with pytest.raises(ValueError, match="Not enough daily surfaces"):
        build_daily_feature_dataset(
            surface_frame=surface_frame,
            grid=grid,
            feature_config=FeatureConfig(lag_windows=(1,)),
            calendar_config=MarketCalendarConfig(),
        )


def test_build_daily_feature_dataset_uses_next_observed_date_and_records_gap_sessions() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 6),
                date(2021, 1, 7),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(
            lag_windows=(1,),
            include_daily_change=False,
            include_liquidity=False,
        ),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame.select("quote_date", "target_date", "target_gap_sessions").to_dict(
        as_series=False
    ) == {
        "quote_date": [date(2021, 1, 4), date(2021, 1, 6)],
        "target_date": [date(2021, 1, 6), date(2021, 1, 7)],
        "target_gap_sessions": [1, 0],
    }


def test_build_daily_feature_dataset_records_two_consecutive_skipped_sessions() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2021, 1, 8),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(
            lag_windows=(1,),
            include_daily_change=False,
            include_liquidity=False,
        ),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame["target_gap_sessions"].to_list() == [0, 2]
    assert all(
        target_date > quote_date
        for quote_date, target_date in zip(
            feature_frame["quote_date"].to_list(),
            feature_frame["target_date"].to_list(),
            strict=True,
        )
    )


def test_build_daily_feature_dataset_does_not_wrap_daily_change_to_the_last_surface() -> None:
    grid = SurfaceGrid.from_config(
        SurfaceGridConfig(
            moneyness_points=(0.0,),
            maturity_days=(30,),
        )
    )
    feature_frame = build_daily_feature_dataset(
        surface_frame=_surface_frame(
            [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2021, 1, 6),
            ]
        ),
        grid=grid,
        feature_config=FeatureConfig(lag_windows=(1,), include_liquidity=False),
        calendar_config=MarketCalendarConfig(),
    ).feature_frame

    assert feature_frame["quote_date"].to_list() == [date(2021, 1, 5)]
    assert feature_frame["target_date"].to_list() == [date(2021, 1, 6)]


def test_build_target_training_weights_keeps_completed_only_cells_supervised() -> None:
    training_weights = build_target_training_weights(
        observed_mask=np.asarray([1.0, 0.0], dtype=np.float64),
        vega_weights=np.asarray([2.0, 0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(training_weights, np.asarray([2.0, 1.0], dtype=np.float64))

```

### tests/unit/test_calendar.py

```python
from __future__ import annotations

from datetime import date, time

from ivsurf.calendar import MarketCalendar


def test_early_close_day_uses_effective_vendor_snapshot_time() -> None:
    calendar = MarketCalendar()
    session_date = date(2019, 11, 29)

    assert calendar.session_has_decision_time(session_date) is True
    assert calendar.effective_decision_datetime(session_date).time() == time(12, 45)
    assert calendar.compute_tau_years(
        quote_date=session_date,
        expiration=date(2019, 12, 20),
        root="SPXW",
    ) > 0.0


def test_next_trading_session_skips_holiday() -> None:
    calendar = MarketCalendar()
    assert calendar.next_trading_session(date(2021, 4, 1)) == date(2021, 4, 5)


def test_next_decision_session_includes_early_close() -> None:
    calendar = MarketCalendar()
    assert calendar.next_decision_session(date(2019, 11, 27)) == date(2019, 11, 29)


def test_am_settled_tau_uses_previous_session_before_settlement() -> None:
    calendar = MarketCalendar(am_settled_roots=("SPX",))
    tau_years = calendar.compute_tau_years(
        quote_date=date(2021, 4, 15),
        expiration=date(2021, 4, 16),
        root="SPX",
    )
    assert 0.0 < tau_years < (1.0 / 365.0)


def test_calendar_supports_pre_2006_history() -> None:
    calendar = MarketCalendar()
    session_date = date(2004, 1, 2)

    assert calendar.is_session(session_date) is True
    assert calendar.session_has_decision_time(session_date) is True
    assert calendar.next_trading_session(session_date) == date(2004, 1, 5)

```

### tests/unit/test_schema.py

```python
from __future__ import annotations

import pytest

from ivsurf.exceptions import SchemaDriftError
from ivsurf.schemas import RAW_ALLOWED_EXTRA_COLUMNS, RAW_COLUMNS, validate_raw_columns


def test_validate_raw_columns_accepts_expected_schema() -> None:
    validate_raw_columns(RAW_COLUMNS)


def test_validate_raw_columns_accepts_audited_extra_columns() -> None:
    validate_raw_columns((*RAW_COLUMNS, *RAW_ALLOWED_EXTRA_COLUMNS))


def test_validate_raw_columns_accepts_audited_implied_underlying_price_column() -> None:
    validate_raw_columns((*RAW_COLUMNS, "implied_underlying_price_1545"))


def test_validate_raw_columns_rejects_unexpected_extra_vendor_column() -> None:
    with pytest.raises(
        SchemaDriftError,
        match=r"unexpected=\['vendor_added_column_1545'\]",
    ):
        validate_raw_columns((*RAW_COLUMNS, "vendor_added_column_1545"))


def test_validate_raw_columns_rejects_missing_required_column() -> None:
    with pytest.raises(
        SchemaDriftError,
        match=r"missing=\['open_interest'\] unexpected=\[\]",
    ):
        validate_raw_columns(RAW_COLUMNS[:-1])

```

### tests/unit/test_raw_checks.py

```python
from __future__ import annotations

from datetime import date

import polars as pl

from ivsurf.qc.raw_checks import assert_target_symbol_only


def test_assert_target_symbol_only_accepts_spx_underlying_scope_across_roots() -> None:
    frame = pl.DataFrame(
        {
            "quote_date": [date(2021, 1, 4), date(2021, 1, 4)],
            "underlying_symbol": ["^SPX", "^SPX"],
            "root": ["SPX", "SPXW"],
        }
    )

    assert_target_symbol_only(
        frame,
        symbol_column="underlying_symbol",
        expected_symbol="^SPX",
        dataset_name="synthetic_spx_scope",
    )

```

### tests/unit/test_forecast_store.py

```python
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ivsurf.evaluation.forecast_store import write_forecasts
from ivsurf.surfaces.grid import SurfaceGrid


def _grid() -> SurfaceGrid:
    return SurfaceGrid(maturity_days=(30, 90), moneyness_points=(-0.1, 0.0))


def test_write_forecasts_rejects_negative_total_variance(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="negative total variance"):
        write_forecasts(
            output_path=tmp_path / "ridge.parquet",
            model_name="ridge",
            quote_dates=np.asarray([date(2021, 1, 4)], dtype=object),
            target_dates=np.asarray([date(2021, 1, 5)], dtype=object),
            predictions=np.asarray([[-1.0e-4, 0.01, 0.02, 0.03]], dtype=np.float64),
            grid=_grid(),
        )


def test_write_forecasts_persists_only_finite_non_negative_values(tmp_path: Path) -> None:
    output_path = tmp_path / "ridge.parquet"
    write_forecasts(
        output_path=output_path,
        model_name="ridge",
        quote_dates=np.asarray([date(2021, 1, 4), date(2021, 1, 5)], dtype=object),
        target_dates=np.asarray([date(2021, 1, 5), date(2021, 1, 6)], dtype=object),
        predictions=np.asarray(
            [
                [0.001, 0.002, 0.003, 0.004],
                [0.005, 0.006, 0.007, 0.008],
            ],
            dtype=np.float64,
        ),
        grid=_grid(),
    )

    frame = pl.read_parquet(output_path)
    predicted = frame["predicted_total_variance"].to_numpy().astype(np.float64)

    assert frame.height == 8
    assert np.isfinite(predicted).all()
    assert (predicted >= 0.0).all()

```

### tests/property/test_walkforward.py

```python
from __future__ import annotations

from datetime import date, timedelta

from hypothesis import assume, given
from hypothesis import strategies as st

from ivsurf.config import WalkforwardConfig
from ivsurf.splits.walkforward import build_walkforward_splits, clean_evaluation_splits


@given(st.integers(min_value=12, max_value=40))
def test_walkforward_splits_are_ordered_and_non_overlapping(length: int) -> None:
    dates = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(length)]
    splits = build_walkforward_splits(
        dates=dates,
        config=WalkforwardConfig(
            train_size=6,
            validation_size=3,
            test_size=2,
            step_size=2,
            expanding_train=True,
        ),
    )
    for split in splits:
        train_set = set(split.train_dates)
        validation_set = set(split.validation_dates)
        test_set = set(split.test_dates)
        assert train_set.isdisjoint(validation_set)
        assert train_set.isdisjoint(test_set)
        assert validation_set.isdisjoint(test_set)
        assert max(split.train_dates) < min(split.validation_dates)
        assert max(split.validation_dates) < min(split.test_dates)


@given(st.integers(min_value=14, max_value=60), st.integers(min_value=1, max_value=3))
def test_clean_evaluation_splits_exclude_hpo_validation_dates(
    length: int,
    tuning_splits_count: int,
) -> None:
    dates = [date(2021, 1, 1) + timedelta(days=offset) for offset in range(length)]
    splits = build_walkforward_splits(
        dates=dates,
        config=WalkforwardConfig(
            train_size=6,
            validation_size=3,
            test_size=2,
            step_size=2,
            expanding_train=True,
        ),
    )
    assume(len(splits) > tuning_splits_count)

    boundary, clean_splits = clean_evaluation_splits(
        splits,
        tuning_splits_count=tuning_splits_count,
    )
    hpo_validation_dates = {
        day
        for split in splits[:tuning_splits_count]
        for day in split.validation_dates
    }
    evaluation_test_dates = {day for split in clean_splits for day in split.test_dates}

    assert clean_splits[0].split_id == boundary.first_clean_test_split_id
    assert evaluation_test_dates.isdisjoint(hpo_validation_dates)
    assert all(
        min(split.test_dates) > boundary.max_hpo_validation_date.isoformat()
        for split in clean_splits
    )

```

### tests/property/test_arbitrage_penalties.py

```python
from __future__ import annotations

import torch
from hypothesis import given
from hypothesis import strategies as st

from ivsurf.models.penalties import calendar_monotonicity_penalty, convexity_penalty


@given(
    st.floats(min_value=0.01, max_value=0.20),
    st.floats(min_value=0.001, max_value=0.05),
    st.floats(min_value=0.001, max_value=0.05),
)
def test_quadratic_surface_has_no_penalty(base: float, slope: float, curvature: float) -> None:
    money = torch.tensor([-0.30, -0.05, 0.30], dtype=torch.float64)
    maturity = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32).unsqueeze(1)
    surface = base + (slope * maturity) + (curvature * torch.zeros_like(money))
    flattened = surface.reshape(1, -1)
    assert float(calendar_monotonicity_penalty(flattened, (3, 3))) == 0.0
    assert (
        float(convexity_penalty(flattened, (3, 3), moneyness_points=(-0.30, -0.05, 0.30)))
        == 0.0
    )

```

### tests/regression/test_observed_mask_preservation.py

```python
from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.interpolation import complete_surface


def test_thresholded_observed_mask_survives_surface_completion() -> None:
    surface_config = SurfaceGridConfig(
        moneyness_points=(0.0, 0.1),
        maturity_days=(30, 60),
        observed_cell_min_count=2,
    )
    grid = SurfaceGrid.from_config(surface_config)
    quote_date = date(2021, 1, 4)
    rows = [
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 0,
            "total_variance": 0.010,
            "implied_volatility_1545": 0.20,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 0,
            "total_variance": 0.012,
            "implied_volatility_1545": 0.21,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "moneyness_index": 1,
            "total_variance": 0.013,
            "implied_volatility_1545": 0.22,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 0,
            "total_variance": 0.014,
            "implied_volatility_1545": 0.23,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 0,
            "total_variance": 0.015,
            "implied_volatility_1545": 0.24,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 1,
            "total_variance": 0.016,
            "implied_volatility_1545": 0.25,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
        {
            "quote_date": quote_date,
            "maturity_index": 1,
            "moneyness_index": 1,
            "total_variance": 0.017,
            "implied_volatility_1545": 0.26,
            "spread_1545": 0.01,
            "vega_1545": 1.0,
        },
    ]
    observed = aggregate_daily_surface(pl.DataFrame(rows), grid, surface_config).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )
    observed_matrix = (
        observed["observed_total_variance"]
        .fill_null(np.nan)
        .to_numpy()
        .reshape(grid.shape)
    )
    observed_mask = observed["observed_mask"].to_numpy().reshape(grid.shape)

    completed = complete_surface(
        observed_total_variance=observed_matrix,
        observed_mask=observed_mask,
        maturity_coordinates=grid.maturity_years,
        moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
        interpolation_order=surface_config.interpolation_order,
        interpolation_cycles=surface_config.interpolation_cycles,
        total_variance_floor=surface_config.total_variance_floor,
    )

    assert np.isfinite(completed.completed_total_variance[0, 1])
    assert bool(completed.observed_mask[0, 1]) is False

```

### tests/integration/test_stage03_stage04_target_gap_alignment.py

```python
from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _silver_rows(
    quote_date: date,
    *,
    total_variance: float,
    is_valid: bool,
) -> list[dict[str, object]]:
    return [
        {
            "quote_date": quote_date,
            "tau_years": 30.0 / 365.0,
            "log_moneyness": 0.0,
            "total_variance": total_variance,
            "implied_volatility_1545": float((total_variance / (30.0 / 365.0)) ** 0.5),
            "vega_1545": 1.0,
            "spread_1545": 0.01,
            "is_valid_observation": is_valid,
        }
    ]


def test_stage03_and_stage04_preserve_skipped_day_gaps_and_gold_input_provenance(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    silver_year = tmp_path / "data" / "silver" / "year=2021"
    silver_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    dates_and_validity = [
        (date(2021, 1, 4), True),
        (date(2021, 1, 5), True),
        (date(2021, 1, 6), False),
        (date(2021, 1, 7), True),
        (date(2021, 1, 8), True),
    ]
    silver_summary: list[dict[str, object]] = []
    for offset, (quote_date_value, is_valid) in enumerate(dates_and_validity, start=1):
        silver_path = silver_year / f"{quote_date_value.isoformat()}.parquet"
        pl.DataFrame(
            _silver_rows(
                quote_date_value,
                total_variance=0.001 * offset,
                is_valid=is_valid,
            )
        ).write_parquet(silver_path)
        silver_summary.append(
            {
                "silver_path": str(silver_path),
                "quote_date": quote_date_value.isoformat(),
                "status": "built",
                "rows": 1,
                "valid_rows": 1 if is_valid else 0,
            }
        )
    (manifests_dir / "silver_build_summary.json").write_bytes(
        orjson.dumps(silver_summary, option=orjson.OPT_INDENT_2)
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'sample_start_date: "2021-01-04"\n'
            'sample_end_date: "2021-01-08"\n'
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [0.0]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    feature_config_path = _write_yaml(
        tmp_path / "features.yaml",
        (
            "lag_windows: [1]\n"
            "include_daily_change: false\n"
            "include_mask: true\n"
            "include_liquidity: false\n"
        ),
    )
    walkforward_config_path = _write_yaml(
        tmp_path / "walkforward.yaml",
        (
            "train_size: 1\n"
            "validation_size: 1\n"
            "test_size: 1\n"
            "step_size: 1\n"
            "expanding_train: true\n"
        ),
    )

    stage03 = _load_script_module(
        repo_root / "scripts" / "03_build_surfaces.py",
        "stage03_target_gap_alignment",
    )
    stage04 = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "stage04_target_gap_alignment",
    )

    stage03.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
    )

    skipped_payload = orjson.loads((manifests_dir / "gold_surface_skipped_dates.json").read_bytes())
    assert skipped_payload == [
        {
            "quote_date": "2021-01-06",
            "reason": "NO_VALID_ROWS_AFTER_CLEANING",
            "silver_path": str(silver_year / "2021-01-06.parquet"),
        }
    ]

    stage03_run_manifest = sorted((manifests_dir / "runs" / "03_build_surfaces").glob("*.json"))[-1]
    stage03_manifest_payload = orjson.loads(stage03_run_manifest.read_bytes())
    stage03_output_paths = {
        artifact["path"] for artifact in stage03_manifest_payload["output_artifacts"]
    }
    assert (
        str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage03_output_paths
    )

    stage04.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        feature_config_path=feature_config_path,
        walkforward_config_path=walkforward_config_path,
    )

    feature_frame = pl.read_parquet(tmp_path / "data" / "gold" / "daily_features.parquet")
    target_gap_row = feature_frame.filter(pl.col("quote_date") == date(2021, 1, 5))
    assert target_gap_row.height == 1
    assert target_gap_row["target_date"].to_list() == [date(2021, 1, 7)]
    assert target_gap_row["target_gap_sessions"].to_list() == [1]

    gold_files = sorted((tmp_path / "data" / "gold").glob("year=*/*.parquet"))
    stage04_run_manifest = sorted((manifests_dir / "runs" / "04_build_features").glob("*.json"))[-1]
    stage04_manifest_payload = orjson.loads(stage04_run_manifest.read_bytes())
    stage04_input_paths = {
        artifact["path"] for artifact in stage04_manifest_payload["input_artifacts"]
    }
    assert str((manifests_dir / "gold_surface_skipped_dates.json").resolve()) in stage04_input_paths
    for gold_file in gold_files:
        assert str(gold_file.resolve()) in stage04_input_paths

```

### tests/integration/test_stage04_early_close_alignment.py

```python
from __future__ import annotations

from datetime import date
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import orjson
import polars as pl


def _load_script_module(script_path: Path, module_name: str) -> ModuleType:
    spec = spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_yaml(path: Path, payload: str) -> Path:
    path.write_text(payload, encoding="utf-8")
    return path


def _gold_rows(quote_date: date, total_variance: float) -> list[dict[str, object]]:
    maturity_days = 30
    maturity_years = maturity_days / 365.0
    return [
        {
            "quote_date": quote_date,
            "maturity_index": 0,
            "maturity_days": maturity_days,
            "moneyness_index": 0,
            "moneyness_point": 0.0,
            "observed_total_variance": total_variance,
            "observed_iv": float((total_variance / maturity_years) ** 0.5),
            "completed_total_variance": total_variance,
            "completed_iv": float((total_variance / maturity_years) ** 0.5),
            "observed_mask": True,
            "vega_sum": 1.0,
            "observation_count": 1,
            "weighted_spread_1545": 0.01,
        }
    ]


def test_stage04_aligns_pre_early_close_features_to_the_early_close_target(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    gold_year = tmp_path / "data" / "gold" / "year=2019"
    gold_year.mkdir(parents=True)
    manifests_dir = tmp_path / "data" / "manifests"
    manifests_dir.mkdir(parents=True)

    quote_dates = [
        date(2019, 11, 26),
        date(2019, 11, 27),
        date(2019, 11, 29),
        date(2019, 12, 2),
        date(2019, 12, 3),
    ]
    gold_summary: list[dict[str, object]] = []
    for offset, quote_date_value in enumerate(quote_dates, start=1):
        output_path = gold_year / f"{quote_date_value.isoformat()}.parquet"
        pl.DataFrame(_gold_rows(quote_date_value, total_variance=0.001 * offset)).write_parquet(
            output_path
        )
        gold_summary.append(
            {
                "gold_path": str(output_path),
                "quote_date": quote_date_value.isoformat(),
                "observed_cells": 1,
            }
        )
    (manifests_dir / "gold_surface_summary.json").write_bytes(
        orjson.dumps(gold_summary, option=orjson.OPT_INDENT_2)
    )
    (manifests_dir / "gold_surface_skipped_dates.json").write_bytes(
        orjson.dumps([], option=orjson.OPT_INDENT_2)
    )

    raw_config_path = _write_yaml(
        tmp_path / "raw.yaml",
        (
            f"raw_options_dir: '{(tmp_path / 'raw').as_posix()}'\n"
            f"bronze_dir: '{(tmp_path / 'data' / 'bronze').as_posix()}'\n"
            f"silver_dir: '{(tmp_path / 'data' / 'silver').as_posix()}'\n"
            f"gold_dir: '{(tmp_path / 'data' / 'gold').as_posix()}'\n"
            f"manifests_dir: '{manifests_dir.as_posix()}'\n"
            'decision_time: "15:45:00"\n'
            "decision_snapshot_minutes_before_close: 15\n"
            'sample_start_date: "2019-11-26"\n'
            'sample_end_date: "2019-12-03"\n'
        ),
    )
    surface_config_path = _write_yaml(
        tmp_path / "surface.yaml",
        (
            "moneyness_points: [0.0]\n"
            "maturity_days: [30]\n"
            'interpolation_order: ["maturity", "moneyness"]\n'
            "interpolation_cycles: 2\n"
            "total_variance_floor: 1.0e-8\n"
            "observed_cell_min_count: 1\n"
        ),
    )
    feature_config_path = _write_yaml(
        tmp_path / "features.yaml",
        (
            "lag_windows: [1]\n"
            "include_daily_change: false\n"
            "include_mask: true\n"
            "include_liquidity: false\n"
        ),
    )
    walkforward_config_path = _write_yaml(
        tmp_path / "walkforward.yaml",
        (
            "train_size: 1\n"
            "validation_size: 1\n"
            "test_size: 1\n"
            "step_size: 1\n"
            "expanding_train: true\n"
        ),
    )

    script_module = _load_script_module(
        repo_root / "scripts" / "04_build_features.py",
        "stage04_early_close_alignment_script",
    )
    script_module.main(
        raw_config_path=raw_config_path,
        surface_config_path=surface_config_path,
        feature_config_path=feature_config_path,
        walkforward_config_path=walkforward_config_path,
    )

    feature_frame = pl.read_parquet(tmp_path / "data" / "gold" / "daily_features.parquet")
    aligned_row = feature_frame.filter(pl.col("quote_date") == date(2019, 11, 27))

    assert aligned_row.height == 1
    assert aligned_row["target_date"].to_list() == [date(2019, 11, 29)]
    assert aligned_row["target_gap_sessions"].to_list() == [0]

```
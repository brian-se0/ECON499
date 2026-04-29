from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
from pytest import MonkeyPatch

from ivsurf.cleaning.derived_fields import (
    add_derived_fields,
    add_effective_decision_timestamp,
    build_tau_lookup,
)
from ivsurf.cleaning.option_filters import apply_option_quality_flags, valid_option_rows
from ivsurf.config import (
    CleaningConfig,
    FeatureConfig,
    MarketCalendarConfig,
    NeuralModelConfig,
    SurfaceGridConfig,
    TrainingProfileConfig,
)
from ivsurf.features.tabular_dataset import build_daily_feature_dataset
from ivsurf.models import neural_surface as neural_surface_module
from ivsurf.models.base import dataset_to_matrices
from ivsurf.surfaces.aggregation import aggregate_daily_surface
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
    assign_grid_indices,
)
from ivsurf.surfaces.interpolation import COMPLETED_SURFACE_SCHEMA_VERSION, complete_surface
from ivsurf.training.fit_torch import fit_and_predict_neural

SURFACE_CONFIG_HASH = "surface-hash"


def _make_daily_option_rows(
    quote_date: date,
    spot: float,
    *,
    omit_target_cell: bool,
) -> pl.DataFrame:
    moneyness_points = (-0.1, 0.0, 0.1)
    expiries = [date(2021, 1, 11), date(2021, 2, 5)]
    rows: list[dict[str, object]] = []
    for expiry in expiries:
        for option_type in ("C", "P"):
            for point in moneyness_points:
                if omit_target_cell and expiry == expiries[1] and point == 0.1:
                    continue
                strike = spot * np.exp(point)
                iv = 0.18 + (0.02 * abs(point)) + (0.01 if expiry == expiries[1] else 0.0)
                rows.append(
                    {
                        "underlying_symbol": "^SPX",
                        "quote_date": quote_date,
                        "root": "SPXW",
                        "expiration": expiry,
                        "strike": float(strike),
                        "option_type": option_type,
                        "trade_volume": 10,
                        "bid_size_1545": 5,
                        "bid_1545": 1.0,
                        "ask_1545": 1.2,
                        "ask_size_1545": 5,
                        "underlying_bid_1545": spot - 0.1,
                        "underlying_ask_1545": spot + 0.1,
                        "implied_underlying_price_1545": spot,
                        "active_underlying_price_1545": spot,
                        "implied_volatility_1545": iv,
                        "delta_1545": 0.5,
                        "gamma_1545": 0.1,
                        "theta_1545": -0.01,
                        "vega_1545": 1.0,
                        "rho_1545": 0.01,
                        "open_interest": 100,
                    }
                )
    return pl.DataFrame(rows)


def _completed_daily_surface(
    quote_date: date,
    spot: float,
    *,
    omit_target_cell: bool,
    grid: SurfaceGrid,
    calendar_config: MarketCalendarConfig,
    cleaning_config: CleaningConfig,
    surface_config: SurfaceGridConfig,
) -> pl.DataFrame:
    bronze = _make_daily_option_rows(
        quote_date=quote_date,
        spot=spot,
        omit_target_cell=omit_target_cell,
    )
    tau_lookup = build_tau_lookup(frame=bronze, calendar_config=calendar_config)
    silver = add_derived_fields(frame=bronze, tau_lookup=tau_lookup)
    silver = add_effective_decision_timestamp(frame=silver, calendar_config=calendar_config)
    silver = apply_option_quality_flags(frame=silver, config=cleaning_config)
    valid = valid_option_rows(silver)
    assigned = assign_grid_indices(frame=valid, grid=grid).filter(pl.col("inside_grid_domain"))
    observed = aggregate_daily_surface(frame=assigned, grid=grid, config=surface_config).sort(
        ["quote_date", "maturity_index", "moneyness_index"]
    )
    observed_matrix = observed["observed_total_variance"].fill_null(np.nan).to_numpy().reshape(
        grid.shape
    )
    completed = complete_surface(
        observed_total_variance=observed_matrix,
        observed_mask=np.isfinite(observed_matrix),
        maturity_coordinates=grid.maturity_years,
        moneyness_coordinates=np.asarray(grid.moneyness_points, dtype=np.float64),
        interpolation_order=("maturity", "moneyness"),
        interpolation_cycles=2,
        total_variance_floor=1.0e-8,
    )
    return observed.with_columns(
        pl.Series(
            "completed_total_variance",
            completed.completed_total_variance.reshape(-1),
        ),
        pl.lit(silver["effective_decision_timestamp"][0]).alias("effective_decision_timestamp"),
        pl.lit(SURFACE_GRID_SCHEMA_VERSION).alias("surface_grid_schema_version"),
        pl.lit(grid.grid_hash).alias("surface_grid_hash"),
        pl.lit(MATURITY_COORDINATE).alias("maturity_coordinate"),
        pl.lit(MONEYNESS_COORDINATE).alias("moneyness_coordinate"),
        pl.lit(COMPLETED_SURFACE_SCHEMA_VERSION).alias("target_surface_version"),
        pl.lit(SURFACE_CONFIG_HASH).alias("surface_config_hash"),
    )


def test_stage03_stage04_emit_positive_training_weight_for_completed_only_target_cell() -> None:
    calendar_config = MarketCalendarConfig()
    cleaning_config = CleaningConfig()
    surface_config = SurfaceGridConfig(
        moneyness_points=(-0.1, 0.0, 0.1),
        maturity_days=(5, 40),
    )
    grid = SurfaceGrid.from_config(surface_config)
    target_column_index = len(grid.moneyness_points) + 2

    surface_frame = pl.concat(
        [
            _completed_daily_surface(
                quote_date=date(2021, 1, 4),
                spot=100.0,
                omit_target_cell=False,
                grid=grid,
                calendar_config=calendar_config,
                cleaning_config=cleaning_config,
                surface_config=surface_config,
            ),
            _completed_daily_surface(
                quote_date=date(2021, 1, 5),
                spot=101.0,
                omit_target_cell=True,
                grid=grid,
                calendar_config=calendar_config,
                cleaning_config=cleaning_config,
                surface_config=surface_config,
            ),
        ]
    )

    feature_frame = build_daily_feature_dataset(
        surface_frame=surface_frame,
        grid=grid,
        feature_config=FeatureConfig(
            lag_windows=(1,),
            include_daily_change=False,
            include_liquidity=False,
        ),
        calendar_config=calendar_config,
    ).feature_frame

    assert feature_frame.height == 1
    assert feature_frame.item(0, f"target_observed_mask_{target_column_index:04d}") == 0.0
    assert feature_frame.item(0, f"target_vega_weight_{target_column_index:04d}") == 0.0
    assert feature_frame.item(0, f"target_training_weight_{target_column_index:04d}") == 1.0


def test_fit_and_predict_neural_uses_positive_imputed_training_weight(
    monkeypatch: MonkeyPatch,
) -> None:
    feature_frame = pl.DataFrame(
        {
            "quote_date": [
                date(2021, 1, 4),
                date(2021, 1, 5),
                date(2021, 1, 6),
                date(2021, 1, 7),
            ],
            "target_date": [
                date(2021, 1, 5),
                date(2021, 1, 6),
                date(2021, 1, 7),
                date(2021, 1, 8),
            ],
            "effective_decision_timestamp": [
                "2021-01-04T15:45:00-05:00",
                "2021-01-05T15:45:00-05:00",
                "2021-01-06T15:45:00-05:00",
                "2021-01-07T15:45:00-05:00",
            ],
            "target_effective_decision_timestamp": [
                "2021-01-05T15:45:00-05:00",
                "2021-01-06T15:45:00-05:00",
                "2021-01-07T15:45:00-05:00",
                "2021-01-08T15:45:00-05:00",
            ],
            "feature_surface_mean_01_0000": [0.10, 0.11, 0.12, 0.13],
            "feature_surface_mean_01_0001": [0.20, 0.21, 0.22, 0.23],
            "target_total_variance_0000": [0.11, 0.12, 0.13, 0.14],
            "target_total_variance_0001": [0.21, 0.22, 0.23, 0.24],
            "target_observed_mask_0000": [1.0, 1.0, 1.0, 1.0],
            "target_observed_mask_0001": [0.0, 0.0, 0.0, 0.0],
            "target_vega_weight_0000": [2.0, 2.0, 2.0, 2.0],
            "target_vega_weight_0001": [0.0, 0.0, 0.0, 0.0],
            "target_training_weight_0000": [2.0, 2.0, 2.0, 2.0],
            "target_training_weight_0001": [1.0, 1.0, 1.0, 1.0],
        }
    )
    matrices = dataset_to_matrices(feature_frame)
    model = neural_surface_module.NeuralSurfaceRegressor(
        config=NeuralModelConfig(
            hidden_width=4,
            depth=1,
            dropout=0.0,
            learning_rate=1.0e-3,
            weight_decay=0.0,
            epochs=2,
            batch_size=2,
            seed=7,
            device="cpu",
            observed_loss_weight=1.0,
            imputed_loss_weight=0.25,
            calendar_penalty_weight=0.0,
            convexity_penalty_weight=0.0,
            roughness_penalty_weight=0.0,
        ),
        grid_shape=(1, 2),
        moneyness_points=(-0.1, 0.0),
    )
    training_profile = TrainingProfileConfig(
        profile_name="test_train_profile",
        epochs=1,
        neural_early_stopping_patience=1,
        neural_early_stopping_min_delta=0.0,
        lightgbm_early_stopping_rounds=1,
        lightgbm_early_stopping_min_delta=0.0,
        lightgbm_first_metric_only=True,
    )

    original_loss = neural_surface_module.weighted_surface_mse
    captured_training_weights: list[np.ndarray] = []
    captured_masks: list[np.ndarray] = []

    def spy_weighted_surface_mse(
        predictions: neural_surface_module.torch.Tensor,
        targets: neural_surface_module.torch.Tensor,
        observed_mask: neural_surface_module.torch.Tensor,
        training_weights: neural_surface_module.torch.Tensor,
        observed_loss_weight: float,
        imputed_loss_weight: float,
    ) -> neural_surface_module.torch.Tensor:
        captured_training_weights.append(training_weights.detach().cpu().numpy())
        captured_masks.append(observed_mask.detach().cpu().numpy())
        return original_loss(
            predictions=predictions,
            targets=targets,
            observed_mask=observed_mask,
            training_weights=training_weights,
            observed_loss_weight=observed_loss_weight,
            imputed_loss_weight=imputed_loss_weight,
        )

    monkeypatch.setattr(neural_surface_module, "weighted_surface_mse", spy_weighted_surface_mse)

    predictions = fit_and_predict_neural(
        model=model,
        train_index=np.asarray([0, 1], dtype=np.int64),
        validation_index=np.asarray([2], dtype=np.int64),
        predict_index=np.asarray([3], dtype=np.int64),
        matrices=matrices,
        training_profile=training_profile,
    )

    assert predictions.shape == (1, 2)
    assert captured_training_weights
    assert any(
        np.any((mask <= 0.5) & (weights > 0.0))
        for mask, weights in zip(captured_masks, captured_training_weights, strict=True)
    )

"""Tests for vendor maturity, cleaning, surface completion, neural scaling, and inference."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from arch.bootstrap import MCS

from ivcast.calendar import MarketCalendar
from ivcast.cleaning.derived_fields import (
    DAILY_SPOT_COLUMN,
    add_derived_fields,
    build_tau_lookup,
)
from ivcast.cleaning.option_filters import apply_option_quality_flags
from ivcast.config import (
    CleaningConfig,
    MarketCalendarConfig,
    NeuralModelConfig,
    RawDataConfig,
    calendar_config_from_raw,
    load_yaml_config,
)
from ivcast.models.neural_surface import NeuralSurfaceMLP, NeuralSurfaceRegressor
from ivcast.reproducibility import collect_execution_identity
from ivcast.stats.mcs import model_confidence_set
from ivcast.surfaces.interpolation import complete_surface

REPO_ROOT = Path(__file__).resolve().parents[2]
DAY = 24.0 * 60.0 * 60.0 / (365.0 * 24.0 * 60.0 * 60.0)


def _official_calendar() -> MarketCalendar:
    raw_config = RawDataConfig.model_validate(
        load_yaml_config(REPO_ROOT / "configs" / "data" / "raw.yaml")
    )
    calendar_config = calendar_config_from_raw(raw_config)
    return MarketCalendar(
        calendar_name=calendar_config.calendar_name,
        timezone=calendar_config.timezone,
        decision_time=calendar_config.decision_time,
        decision_snapshot_minutes_before_close=calendar_config.decision_snapshot_minutes_before_close,
        am_settled_roots=calendar_config.am_settled_roots,
        am_settlement_time=calendar_config.am_settlement_time,
        vendor_am_iv_roots=calendar_config.vendor_am_iv_roots,
        vendor_am_iv_previous_session_before=calendar_config.vendor_am_iv_previous_session_before,
    )


# --- maturity timing ------------------------------------------------------------------


def test_am_settled_maturity_runs_to_the_settlement_open() -> None:
    calendar = _official_calendar()
    # Wednesday 15:45 to the Friday 09:30 opening settlement of a Saturday expiration.
    tau = calendar.compute_tau_years(date(2009, 3, 18), date(2009, 3, 21), "SPX")
    assert tau == pytest.approx((1.0 + 17.75 / 24.0) * DAY)


def test_pre_2010_alternate_standard_roots_are_am_settled() -> None:
    calendar = _official_calendar()
    for root in ("SPZ", "SPT", "SXZ", "SZP", "JXA"):
        assert root in calendar.am_settled_roots
    spx = calendar.compute_tau_years(date(2009, 3, 18), date(2009, 3, 21), "SPX")
    spz = calendar.compute_tau_years(date(2009, 3, 18), date(2009, 3, 21), "SPZ")
    assert spz == pytest.approx(spx)
    quarterly = calendar.compute_tau_years(date(2009, 3, 18), date(2009, 3, 31), "QZQ")
    assert quarterly == pytest.approx((13.0 + 0.25 / 24.0) * DAY)


def test_vendor_iv_maturity_matches_the_vendor_greeks() -> None:
    calendar = _official_calendar()
    # Values inferred from vega / (gamma * S^2 * iv) on these archive dates.
    spx = calendar.compute_vendor_iv_tau_years(date(2009, 3, 18), date(2009, 3, 21), "SPX")
    spz = calendar.compute_vendor_iv_tau_years(date(2009, 3, 18), date(2009, 3, 21), "SPZ")
    weekly_2010 = calendar.compute_vendor_iv_tau_years(date(2010, 5, 26), date(2010, 5, 28), "SPX")
    friday_2019 = calendar.compute_vendor_iv_tau_years(date(2019, 6, 11), date(2019, 6, 21), "SPX")
    assert spx / DAY == pytest.approx(1.7404, abs=0.01)
    assert spz / DAY == pytest.approx(2.0308, abs=0.03)
    assert weekly_2010 / DAY == pytest.approx(0.7393, abs=0.01)
    assert friday_2019 / DAY == pytest.approx(9.7427, abs=0.01)
    economic_weekly = calendar.compute_tau_years(date(2010, 5, 26), date(2010, 5, 28), "SPX")
    assert economic_weekly / DAY == pytest.approx(1.7396, abs=1e-3)


def _bronze_rows() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for expiration, spot in ((date(2020, 8, 14), 3383.99), (date(2021, 12, 17), 3248.30)):
        rows.append(
            {
                "quote_date": date(2020, 8, 12),
                "root": "SPXW",
                "expiration": expiration,
                "strike": 3400.0,
                "option_type": "C",
                "bid_1545": 10.0,
                "ask_1545": 11.0,
                "implied_volatility_1545": 0.2,
                "vega_1545": 1.0,
                "active_underlying_price_1545": spot,
            }
        )
    return pl.DataFrame(rows)


def test_total_variance_uses_vendor_maturity_and_one_daily_spot() -> None:
    frame = _bronze_rows()
    calendar_config = MarketCalendarConfig()
    enriched = add_derived_fields(frame, build_tau_lookup(frame, calendar_config))
    assert enriched[DAILY_SPOT_COLUMN].unique().to_list() == [3383.99]
    np.testing.assert_allclose(
        enriched["log_moneyness"].to_numpy(),
        np.log(3400.0) - np.log(3383.99),
    )
    np.testing.assert_allclose(
        enriched["total_variance"].to_numpy(),
        0.04 * enriched["tau_vendor_years"].to_numpy(),
    )


# --- cleaning ---------------------------------------------------------------------------


def _cleaning_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "quote_date": date(2005, 11, 4),
        "option_type": "C",
        "bid_1545": 1.0,
        "ask_1545": 1.2,
        "implied_volatility_1545": 0.2,
        "vega_1545": 1.0,
        "active_underlying_price_1545": 100.0,
        "strike": 100.0,
        "mid_1545": 1.1,
        "tau_years": 0.1,
        "log_moneyness": 0.0,
        "total_variance": 0.004,
    }
    row.update(overrides)
    return row


def test_vendor_placeholder_quotes_are_flagged() -> None:
    frame = pl.DataFrame(
        [
            _cleaning_row(),
            _cleaning_row(bid_1545=998.0, ask_1545=999.0, mid_1545=998.5),
            _cleaning_row(implied_volatility_1545=0.02),
            _cleaning_row(implied_volatility_1545=0.001),
        ]
    )
    config = CleaningConfig(
        placeholder_bid_ask_pairs=((998.0, 999.0),),
        placeholder_implied_volatilities=(0.02, 0.001),
    )
    flagged = apply_option_quality_flags(frame, config)
    assert flagged["invalid_reason"].to_list() == [
        None,
        "PLACEHOLDER_BID_ASK_QUOTE",
        "PLACEHOLDER_IMPLIED_VOLATILITY",
        "PLACEHOLDER_IMPLIED_VOLATILITY",
    ]
    assert apply_option_quality_flags(frame, CleaningConfig())["is_valid_observation"].all()


def test_official_cleaning_config_rejects_vendor_placeholders() -> None:
    config = CleaningConfig.model_validate(
        load_yaml_config(REPO_ROOT / "configs" / "data" / "cleaning.yaml")
    )
    assert (998.0, 999.0) in config.placeholder_bid_ask_pairs
    assert set(config.placeholder_implied_volatilities) == {0.02, 0.001}


# --- surface completion -----------------------------------------------------------------


def test_maturity_boundary_fill_holds_implied_volatility_constant() -> None:
    maturities = np.asarray([1.0, 7.0, 14.0, 30.0]) / 365.0
    sigma = 0.2
    observed = np.full((4, 1), np.nan)
    observed[1:3, 0] = sigma * sigma * maturities[1:3]
    mask = np.isfinite(observed)
    completed = complete_surface(
        observed_total_variance=observed,
        observed_mask=mask,
        maturity_coordinates=maturities,
        moneyness_coordinates=np.asarray([0.0]),
        interpolation_order=("maturity",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
    ).completed_total_variance[:, 0]
    np.testing.assert_allclose(np.sqrt(completed / maturities), sigma)

    flat_total_variance = complete_surface(
        observed_total_variance=observed,
        observed_mask=mask,
        maturity_coordinates=maturities,
        moneyness_coordinates=np.asarray([0.0]),
        interpolation_order=("maturity",),
        interpolation_cycles=1,
        total_variance_floor=1.0e-8,
        maturity_boundary_fill="flat_total_variance",
    ).completed_total_variance[:, 0]
    assert flat_total_variance[0] == pytest.approx(observed[1, 0])


# --- neural output scale ----------------------------------------------------------------


def test_neural_head_is_expressed_in_target_scale_units() -> None:
    network = NeuralSurfaceMLP(3, 2, 4, 1, 0.0, 1.0e-8, output_scale=0.01)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    output = network(torch.zeros((1, 3)))
    np.testing.assert_allclose(output.detach().numpy(), 0.01 * np.log(2.0) + 1.0e-8)
    assert "output_scale" in network.state_dict()


def test_neural_regressor_does_not_collapse_on_total_variance_scale_targets() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(256, 6))
    base = 0.004 * np.exp(0.2 * features[:, :1])
    targets = np.repeat(base, 4, axis=1) * np.asarray([1.0, 1.2, 1.4, 1.6])
    config = NeuralModelConfig(
        hidden_width=32,
        depth=2,
        dropout=0.0,
        learning_rate=0.005,
        weight_decay=0.0,
        epochs=40,
        batch_size=32,
        seed=7,
        calendar_penalty_weight=0.0,
        convexity_penalty_weight=0.0,
        roughness_penalty_weight=0.0,
        device="cpu",
    )
    model = NeuralSurfaceRegressor(config=config, grid_shape=(2, 2), moneyness_points=(0.0, 0.1))
    ones = np.ones_like(targets)
    model.fit(features, targets, ones, ones, ones)
    predictions = model.predict(features)
    assert model.target_scale == pytest.approx(float(targets.mean()))
    assert float(np.mean(predictions < 1.0e-6)) == 0.0
    assert float(predictions.mean() / targets.mean()) == pytest.approx(1.0, abs=0.15)


# --- model confidence set ---------------------------------------------------------------


def test_mcs_statistic_is_not_inflated_by_sqrt_n_and_eliminates_by_e_r() -> None:
    rng = np.random.default_rng(11)
    n_obs = 600
    base = rng.normal(1.0, 0.3, size=n_obs)
    losses = np.column_stack(
        [
            base + rng.normal(0.0, 0.05, size=n_obs),
            base + 0.01 + rng.normal(0.0, 0.05, size=n_obs),
            base + 0.30 + rng.normal(0.0, 0.05, size=n_obs),
            # Large mean loss driven by one outlier but noisy: not the most standardized.
            base + np.where(np.arange(n_obs) == 5, 400.0, 0.0),
        ]
    )
    names = ("a", "b", "clearly_worse", "outlier")
    result = model_confidence_set(
        losses, names, alpha=0.10, block_size=5, bootstrap_reps=500, seed=7
    )
    first = result.iterations[0]
    assert first.included_models == names
    assert first.eliminated_model == "clearly_worse"
    assert first.test_statistic < 200.0
    p_values = [iteration.mcs_p_value for iteration in result.iterations]
    assert p_values == sorted(p_values)
    reference = MCS(
        losses, size=0.10, reps=500, block_size=5, method="R", bootstrap="circular", seed=7
    )
    reference.compute()
    included = [int(str(index)) for index in reference.included]
    assert set(result.superior_models) == {names[index] for index in included}


def test_mcs_rejects_nonzero_constant_contrasts() -> None:
    losses = np.column_stack([np.linspace(1.0, 2.0, 20), np.linspace(1.0, 2.0, 20) + 0.5])
    with pytest.raises(ValueError, match="zero bootstrap variance"):
        model_confidence_set(
            losses, ("a", "b"), alpha=0.10, block_size=2, bootstrap_reps=50, seed=7
        )


# --- provenance on Mac-touched drives ---------------------------------------------------


def test_execution_identity_ignores_appledouble_metadata(tmp_path: Path) -> None:
    package_dir = tmp_path / "src" / "ivcast"
    package_dir.mkdir(parents=True)
    (package_dir / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package_dir / "._module.py").write_bytes(b"\x00\x05\x16\x07Mac OS X\xb0")
    identity = collect_execution_identity(tmp_path)
    assert [record["path"] for record in identity["source_files"]] == ["src/ivcast/module.py"]

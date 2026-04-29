from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from ivsurf.config import SurfaceGridConfig
from ivsurf.surfaces.grid import (
    MATURITY_COORDINATE,
    MONEYNESS_COORDINATE,
    SURFACE_GRID_SCHEMA_VERSION,
    SurfaceGrid,
    assign_grid_indices,
    infer_surface_grid_from_frame,
    require_complete_unique_surface_grid,
    require_surface_grid_metadata,
)


def test_assign_grid_indices_flags_outside_grid_domain_without_boundary_bucket() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0, 0.1))
    frame = pl.DataFrame(
        {
            "tau_years": [30.0 / 365.0, 90.0 / 365.0, 30.0 / 365.0],
            "log_moneyness": [0.0, 0.0, 0.2],
        }
    )

    assigned = assign_grid_indices(frame=frame, grid=grid)

    assert assigned["inside_grid_domain"].to_list() == [True, False, False]
    assert assigned["grid_domain_reason"].to_list() == [
        "VALID_GRID_DOMAIN",
        "OUTSIDE_MATURITY_GRID_DOMAIN",
        "OUTSIDE_MONEYNESS_GRID_DOMAIN",
    ]
    assert assigned["maturity_index"].to_list() == [0, None, None]
    assert assigned["moneyness_index"].to_list() == [1, None, None]


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"moneyness_points": (), "maturity_days": (30,)}, "moneyness_points"),
        ({"moneyness_points": (-0.1, -0.1), "maturity_days": (30,)}, "unique"),
        ({"moneyness_points": (0.0, -0.1), "maturity_days": (30,)}, "strictly increasing"),
        ({"moneyness_points": (float("nan"),), "maturity_days": (30,)}, "finite"),
        ({"moneyness_points": (False, True), "maturity_days": (30,)}, "numeric"),
        ({"moneyness_points": ("0.0", "0.1"), "maturity_days": (30,)}, "numeric"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": ()}, "maturity_days"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": (30, 30)}, "unique"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": (60, 30)}, "increasing"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": (0, 30)}, "positive"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": (-1, 30)}, "positive"),
        ({"moneyness_points": (-0.1, 0.0), "maturity_days": (True, 30)}, "integer"),
    ],
)
def test_surface_grid_config_rejects_invalid_coordinates(
    payload: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SurfaceGridConfig.model_validate(
            {
                **payload,
                "interpolation_order": ("maturity", "moneyness"),
                "interpolation_cycles": 2,
                "total_variance_floor": 1.0e-8,
                "observed_cell_min_count": 1,
            }
        )


@pytest.mark.parametrize(
    ("maturity_days", "moneyness_points", "match"),
    [
        ((30, 30), (-0.1, 0.0), "unique"),
        ((30,), (False, True), "numeric"),
        ((30,), ("0.0", "0.1"), "numeric"),
        ((30,), (object(), 0.1), "numeric"),
        ((30,), (False, 0.0), "numeric"),
        ((30,), (float("nan"), 0.1), "finite"),
    ],
)
def test_surface_grid_dataclass_rejects_programmatic_invalid_coordinates(
    maturity_days: tuple[object, ...],
    moneyness_points: tuple[object, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        SurfaceGrid(maturity_days=maturity_days, moneyness_points=moneyness_points)  # type: ignore[arg-type]


def test_surface_grid_metadata_is_explicit_and_validated() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0, 0.1))
    frame = pl.DataFrame(
        {
            "surface_grid_schema_version": [SURFACE_GRID_SCHEMA_VERSION],
            "surface_grid_hash": [grid.grid_hash],
            "maturity_coordinate": [MATURITY_COORDINATE],
            "moneyness_coordinate": [MONEYNESS_COORDINATE],
        }
    )

    require_surface_grid_metadata(frame, grid, dataset_name="unit grid")

    assert grid.metadata["moneyness_coordinate"] == "log_spot_moneyness"


def _complete_grid_frame() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for quote_date in (date(2021, 1, 4), date(2021, 1, 5)):
        for maturity_index, maturity_days in enumerate((30, 60)):
            for moneyness_index, moneyness_point in enumerate((-0.1, 0.0)):
                rows.append(
                    {
                        "quote_date": quote_date,
                        "maturity_index": maturity_index,
                        "maturity_days": maturity_days,
                        "moneyness_index": moneyness_index,
                        "moneyness_point": moneyness_point,
                    }
                )
    return pl.DataFrame(rows)


def test_complete_unique_surface_grid_contract_accepts_full_daily_grid() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0))

    require_complete_unique_surface_grid(
        _complete_grid_frame(),
        grid,
        dataset_name="unit surface",
        group_columns=("quote_date",),
    )


def test_complete_unique_surface_grid_contract_rejects_duplicate_cell() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0))
    frame = pl.concat([_complete_grid_frame(), _complete_grid_frame().head(1)])

    with pytest.raises(ValueError, match="duplicate surface-grid cells"):
        require_complete_unique_surface_grid(
            frame,
            grid,
            dataset_name="unit surface",
            group_columns=("quote_date",),
        )


def test_complete_unique_surface_grid_contract_rejects_missing_cell() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0))
    frame = _complete_grid_frame().filter(
        ~(
            (pl.col("quote_date") == date(2021, 1, 5))
            & (pl.col("maturity_index") == 1)
            & (pl.col("moneyness_index") == 1)
        )
    )

    with pytest.raises(ValueError, match="exactly 4 grid cells"):
        require_complete_unique_surface_grid(
            frame,
            grid,
            dataset_name="unit surface",
            group_columns=("quote_date",),
        )


def test_complete_unique_surface_grid_contract_rejects_coordinate_mismatch() -> None:
    grid = SurfaceGrid(maturity_days=(30, 60), moneyness_points=(-0.1, 0.0))
    frame = _complete_grid_frame().with_columns(
        pl.when((pl.col("maturity_index") == 1) & (pl.col("moneyness_index") == 1))
        .then(61)
        .otherwise(pl.col("maturity_days"))
        .alias("maturity_days")
    )

    with pytest.raises(ValueError, match="coordinates do not match"):
        require_complete_unique_surface_grid(
            frame,
            grid,
            dataset_name="unit surface",
            group_columns=("quote_date",),
        )


def test_infer_surface_grid_from_frame_rejects_noncontiguous_indices() -> None:
    frame = pl.DataFrame(
        {
            "maturity_index": [1],
            "maturity_days": [30],
            "moneyness_index": [0],
            "moneyness_point": [0.0],
        }
    )

    with pytest.raises(ValueError, match="contiguous from zero"):
        infer_surface_grid_from_frame(frame, dataset_name="unit surface")

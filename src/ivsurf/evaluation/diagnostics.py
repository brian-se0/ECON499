"""Arbitrage-aware diagnostic reporting for realized and predicted surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl

from ivsurf.surfaces.arbitrage_diagnostics import summarize_diagnostics
from ivsurf.surfaces.grid import SurfaceGrid
from ivsurf.surfaces.masks import reshape_surface_column


@dataclass(frozen=True, slots=True)
class DiagnosticRow:
    """Daily arbitrage diagnostic for one model surface."""

    model_name: str
    quote_date: date
    target_date: date
    calendar_violation_count: int
    calendar_violation_magnitude: float
    convexity_violation_count: int
    convexity_violation_magnitude: float


def build_forecast_diagnostic_frame(
    forecast_frame: pl.DataFrame,
    grid: SurfaceGrid,
) -> pl.DataFrame:
    """Compute arbitrage diagnostics for each saved forecast surface."""

    rows: list[DiagnosticRow] = []
    for group in forecast_frame.partition_by(
        ["model_name", "quote_date", "target_date"],
        maintain_order=True,
    ):
        diagnostics = summarize_diagnostics(
            reshape_surface_column(
                group,
                grid,
                "predicted_total_variance",
            )
        )
        rows.append(
            DiagnosticRow(
                model_name=str(group["model_name"][0]),
                quote_date=group["quote_date"][0],
                target_date=group["target_date"][0],
                calendar_violation_count=diagnostics.calendar_violation_count,
                calendar_violation_magnitude=diagnostics.calendar_violation_magnitude,
                convexity_violation_count=diagnostics.convexity_violation_count,
                convexity_violation_magnitude=diagnostics.convexity_violation_magnitude,
            )
        )
    return pl.DataFrame(rows).sort(["model_name", "quote_date", "target_date"])


def build_actual_diagnostic_frame(
    actual_surface_frame: pl.DataFrame,
    grid: SurfaceGrid,
) -> pl.DataFrame:
    """Compute the same diagnostics for realized completed surfaces."""

    rows: list[DiagnosticRow] = []
    for group in actual_surface_frame.partition_by("quote_date", maintain_order=True):
        quote_date = group["quote_date"][0]
        diagnostics = summarize_diagnostics(
            reshape_surface_column(
                group,
                grid,
                "completed_total_variance",
            )
        )
        rows.append(
            DiagnosticRow(
                model_name="actual_surface",
                quote_date=quote_date,
                target_date=quote_date,
                calendar_violation_count=diagnostics.calendar_violation_count,
                calendar_violation_magnitude=diagnostics.calendar_violation_magnitude,
                convexity_violation_count=diagnostics.convexity_violation_count,
                convexity_violation_magnitude=diagnostics.convexity_violation_magnitude,
            )
        )
    return pl.DataFrame(rows).sort(["model_name", "quote_date", "target_date"])


def summarize_diagnostic_frame(diagnostic_frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate daily diagnostics into per-model summaries."""

    return (
        diagnostic_frame.group_by("model_name")
        .agg(
            pl.col("calendar_violation_count").mean().alias("mean_calendar_violation_count"),
            pl.col("calendar_violation_magnitude").mean().alias(
                "mean_calendar_violation_magnitude"
            ),
            pl.col("convexity_violation_count").mean().alias("mean_convexity_violation_count"),
            pl.col("convexity_violation_magnitude").mean().alias(
                "mean_convexity_violation_magnitude"
            ),
            pl.len().alias("n_surfaces"),
        )
        .sort(["mean_calendar_violation_magnitude", "mean_convexity_violation_magnitude"])
    )

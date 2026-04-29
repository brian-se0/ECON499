"""Feature-artifact field availability metadata."""

from __future__ import annotations

from typing import Literal

import polars as pl

from ivsurf.cleaning.derived_fields import DECISION_TIMESTAMP_COLUMN

AvailabilityRole = Literal[
    "alignment_key",
    "decision_timestamp",
    "model_feature",
    "target",
    "target_alignment",
    "target_weight",
    "surface_metadata",
]

TARGET_DECISION_TIMESTAMP_COLUMN = f"target_{DECISION_TIMESTAMP_COLUMN}"


def _record(
    column_name: str,
    *,
    role: AvailabilityRole,
    availability_reference_column: str,
) -> dict[str, str]:
    return {
        "column_name": column_name,
        "role": role,
        "availability_reference_column": availability_reference_column,
    }


def build_feature_availability_manifest(feature_frame: pl.DataFrame) -> list[dict[str, str]]:
    """Declare the availability reference for every daily-feature artifact column."""

    records: list[dict[str, str]] = []
    for column_name in feature_frame.columns:
        if column_name == "quote_date":
            records.append(
                _record(
                    column_name,
                    role="alignment_key",
                    availability_reference_column=DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name == "target_date":
            records.append(
                _record(
                    column_name,
                    role="target_alignment",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name == "target_gap_sessions":
            records.append(
                _record(
                    column_name,
                    role="target_alignment",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name == DECISION_TIMESTAMP_COLUMN:
            records.append(
                _record(
                    column_name,
                    role="decision_timestamp",
                    availability_reference_column=DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name == TARGET_DECISION_TIMESTAMP_COLUMN:
            records.append(
                _record(
                    column_name,
                    role="decision_timestamp",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name.startswith("feature_"):
            records.append(
                _record(
                    column_name,
                    role="model_feature",
                    availability_reference_column=DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name.startswith("target_total_variance_"):
            records.append(
                _record(
                    column_name,
                    role="target",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name.startswith("target_observed_mask_"):
            records.append(
                _record(
                    column_name,
                    role="target",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name.startswith("target_vega_weight_") or column_name.startswith(
            "target_training_weight_"
        ):
            records.append(
                _record(
                    column_name,
                    role="target_weight",
                    availability_reference_column=TARGET_DECISION_TIMESTAMP_COLUMN,
                )
            )
        elif column_name in {
            "surface_grid_schema_version",
            "surface_grid_hash",
            "maturity_coordinate",
            "moneyness_coordinate",
            "target_surface_version",
            "surface_config_hash",
        }:
            records.append(
                _record(
                    column_name,
                    role="surface_metadata",
                    availability_reference_column=DECISION_TIMESTAMP_COLUMN,
                )
            )
        else:
            message = (
                "Daily feature artifact contains a column without declared availability "
                f"metadata: {column_name!r}."
            )
            raise ValueError(message)
    return records

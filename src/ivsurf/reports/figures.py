"""Lightweight SVG figure generation for report artifacts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import escape

import polars as pl

PALETTE = (
    "#1b3a4b",
    "#9b2226",
    "#005f73",
    "#ca6702",
    "#6c757d",
    "#2a9d8f",
    "#bb3e03",
    "#386641",
)


def _svg_wrapper(width: int, height: int, body: Sequence[str]) -> str:
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#fcfbf7"/>',
            *body,
            "</svg>",
        ]
    )


def _write_svg(path: Path, svg: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path


def write_ranked_bar_chart(
    frame: pl.DataFrame,
    *,
    label_column: str,
    value_column: str,
    output_path: Path,
    title: str,
    top_n: int | None = None,
    lower_is_better: bool = True,
) -> Path:
    """Write a ranked horizontal bar chart as SVG."""

    chart_frame = frame.sort(value_column, descending=not lower_is_better)
    if top_n is not None:
        chart_frame = chart_frame.head(top_n)
    labels = [str(value) for value in chart_frame[label_column].to_list()]
    values = [float(value) for value in chart_frame[value_column].to_list()]
    if not values:
        return _write_svg(
            output_path,
            _svg_wrapper(800, 160, [f'<text x="24" y="48" font-size="18">{escape(title)}</text>']),
        )

    width = 960
    height = 110 + (len(values) * 38)
    margin_left = 220
    bar_width = 600
    max_value = max(values)
    scale = bar_width / max_value if max_value > 0.0 else 1.0

    body = [
        f'<text x="24" y="38" font-size="22" font-family="Segoe UI">{escape(title)}</text>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        y = 70 + (index * 38)
        width_px = max(value * scale, 1.0)
        color = PALETTE[index % len(PALETTE)]
        body.extend(
            [
                (
                    f'<text x="24" y="{y + 16}" font-size="14" font-family="Segoe UI">'
                    f"{escape(label)}</text>"
                ),
                (
                    f'<rect x="{margin_left}" y="{y}" width="{width_px:.2f}" height="22" '
                    f'fill="{color}" rx="4" />'
                ),
                (
                    f'<text x="{margin_left + width_px + 8:.2f}" y="{y + 16}" '
                    f'font-size="13" font-family="Consolas">{value:.6f}</text>'
                ),
            ]
        )
    return _write_svg(output_path, _svg_wrapper(width, height, body))


def write_multi_line_chart(
    frame: pl.DataFrame,
    *,
    x_column: str,
    y_column: str,
    series_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    include_series: tuple[str, ...] | None = None,
) -> Path:
    """Write a multi-series line chart as SVG."""

    chart_frame = frame
    if include_series is not None:
        chart_frame = chart_frame.filter(pl.col(series_column).is_in(include_series))
    if chart_frame.is_empty():
        return _write_svg(
            output_path,
            _svg_wrapper(800, 180, [f'<text x="24" y="48" font-size="18">{escape(title)}</text>']),
        )

    width = 960
    height = 520
    margin_left = 88
    margin_right = 180
    margin_top = 64
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [float(value) for value in chart_frame[x_column].to_list()]
    y_values = [float(value) for value in chart_frame[y_column].to_list()]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    x_span = max(max_x - min_x, 1.0)
    y_span = max(max_y - min_y, 1.0e-12)

    def scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / x_span) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - (((value - min_y) / y_span) * plot_height)

    body = [
        f'<text x="24" y="34" font-size="22" font-family="Segoe UI">{escape(title)}</text>',
        (
            f'<text x="24" y="{height - 18}" font-size="14" '
            f'font-family="Segoe UI">{escape(x_label)}</text>'
        ),
        (
            f'<text x="24" y="{margin_top - 18}" font-size="14" '
            f'font-family="Segoe UI">{escape(y_label)}</text>'
        ),
        (
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            'fill="#ffffff" stroke="#d9d3c7" />'
        ),
        (
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
            f'x2="{margin_left + plot_width}" '
            f'y2="{margin_top + plot_height}" stroke="#7a6f5a" />'
        ),
        (
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" '
            f'y2="{margin_top + plot_height}" stroke="#7a6f5a" />'
        ),
    ]

    unique_x = sorted({float(value) for value in x_values})
    for x_value in unique_x:
        x_pos = scale_x(x_value)
        body.extend(
            [
                (
                    f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" '
                    f'y2="{margin_top + plot_height}" stroke="#f0ece3" />'
                ),
                (
                    f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 20}" '
                    f'text-anchor="middle" font-size="12" font-family="Consolas">{x_value:g}</text>'
                ),
            ]
        )

    y_ticks = 5
    for tick_index in range(y_ticks + 1):
        y_value = min_y + ((y_span / y_ticks) * tick_index)
        y_pos = scale_y(y_value)
        body.extend(
            [
                (
                    f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" '
                    f'y2="{y_pos:.2f}" stroke="#f0ece3" />'
                ),
                (
                    f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
                    f'font-size="12" font-family="Consolas">{y_value:.4f}</text>'
                ),
            ]
        )

    series_names = chart_frame[series_column].unique().sort().to_list()
    for series_index, series_name in enumerate(series_names):
        series_rows = chart_frame.filter(pl.col(series_column) == series_name).sort(x_column)
        points = [
            (scale_x(float(row[x_column])), scale_y(float(row[y_column])))
            for row in series_rows.iter_rows(named=True)
        ]
        if len(points) == 1:
            x_pos, y_pos = points[0]
            color = PALETTE[series_index % len(PALETTE)]
            points_text = (
                f'<circle cx="{x_pos:.2f}" cy="{y_pos:.2f}" '
                f'r="3.5" fill="{color}" />'
            )
        else:
            polyline = " ".join(f"{x_pos:.2f},{y_pos:.2f}" for x_pos, y_pos in points)
            color = PALETTE[series_index % len(PALETTE)]
            points_text = (
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{polyline}" />'
            )
        body.append(points_text)
        legend_y = margin_top + 20 + (series_index * 24)
        color = PALETTE[series_index % len(PALETTE)]
        body.extend(
            [
                (
                    f'<line x1="{width - margin_right + 12}" y1="{legend_y}" '
                    f'x2="{width - margin_right + 36}" y2="{legend_y}" '
                    f'stroke="{color}" stroke-width="3" />'
                ),
                (
                    f'<text x="{width - margin_right + 44}" y="{legend_y + 4}" '
                    f'font-size="13" font-family="Segoe UI">{escape(str(series_name))}</text>'
                ),
            ]
        )

    return _write_svg(output_path, _svg_wrapper(width, height, body))

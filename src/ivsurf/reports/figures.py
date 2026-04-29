"""Lightweight SVG figure generation for report artifacts."""

from __future__ import annotations

import math
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


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    stripped = color.lstrip("#")
    if len(stripped) != 6:
        message = f"Expected 6-digit hex color, found {color!r}."
        raise ValueError(message)
    return tuple(int(stripped[index : index + 2], 16) for index in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _interpolate_hex_color(start: str, end: str, fraction: float) -> str:
    bounded_fraction = _clamp(fraction, 0.0, 1.0)
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    blended = tuple(
        round(start_channel + ((end_channel - start_channel) * bounded_fraction))
        for start_channel, end_channel in zip(start_rgb, end_rgb, strict=True)
    )
    return _rgb_to_hex(blended)


def _text_color_for_fill(fill_color: str) -> str:
    red, green, blue = _hex_to_rgb(fill_color)
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    return "#fcfbf7" if luminance < 135.0 else "#1f2428"


def _format_axis_tick(value: float) -> str:
    if value >= 1000.0:
        return f"{int(value):,}"
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.1f}"


def _model_abbreviation(model_name: str) -> str:
    overrides = {
        "naive": "NAI",
        "har_factor": "HAR",
        "random_forest": "RF",
        "lightgbm": "LGBM",
        "elasticnet": "EN",
        "neural_surface": "NN",
        "ridge": "RID",
        "actual_surface": "ACT",
    }
    return overrides.get(model_name, model_name[:4].upper())


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


def write_normalized_log_ranking_chart(
    frame: pl.DataFrame,
    *,
    label_column: str,
    value_column: str,
    benchmark_label: str,
    output_path: Path,
    title: str,
    lower_is_better: bool = True,
) -> Path:
    """Write a normalized ranking chart with a log-scaled horizontal axis."""

    if label_column not in frame.columns or value_column not in frame.columns:
        message = (
            "Ranking chart requires both the label and value columns, found "
            f"{frame.columns!r}."
        )
        raise ValueError(message)
    if benchmark_label not in frame[label_column].to_list():
        message = f"Benchmark label {benchmark_label!r} not found in ranking frame."
        raise ValueError(message)
    if not lower_is_better:
        message = "Normalized log ranking charts currently support only lower-is-better metrics."
        raise ValueError(message)

    chart_frame = frame.sort(value_column)
    labels = [str(value) for value in chart_frame[label_column].to_list()]
    raw_values = [float(value) for value in chart_frame[value_column].to_list()]
    if not raw_values:
        return _write_svg(
            output_path,
            _svg_wrapper(800, 160, [f'<text x="24" y="48" font-size="18">{escape(title)}</text>']),
        )
    if any((not math.isfinite(value)) or value < 0.0 for value in raw_values):
        message = "Normalized log ranking charts require finite non-negative values."
        raise ValueError(message)
    positive_values = [value for value in raw_values if value > 0.0]
    display_floor = min(positive_values) / 10.0 if positive_values else 1.0e-12

    benchmark_value = float(
        frame.filter(pl.col(label_column) == benchmark_label)[value_column][0]
    )
    if not math.isfinite(benchmark_value) or benchmark_value < 0.0:
        message = f"Benchmark value must be finite and non-negative, found {benchmark_value!r}."
        raise ValueError(message)
    display_benchmark_value = benchmark_value if benchmark_value > 0.0 else display_floor
    ratios = [
        (value if value > 0.0 else display_floor) / display_benchmark_value
        for value in raw_values
    ]
    min_ratio = min(ratios)
    max_ratio = max(ratios)
    min_exp = math.floor(math.log10(min_ratio))
    max_exp = math.ceil(math.log10(max_ratio))
    if min_exp == max_exp:
        min_exp -= 1
        max_exp += 1
    axis_min = 10.0**min_exp
    axis_max = 10.0**max_exp
    log_span = math.log10(axis_max) - math.log10(axis_min)

    width = 1040
    height = 120 + (len(labels) * 44)
    margin_left = 240
    margin_right = 160
    margin_top = 88
    margin_bottom = 58
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    def scale_x(value: float) -> float:
        return margin_left + (
            (math.log10(value) - math.log10(axis_min)) / log_span
        ) * plot_width

    benchmark_x = scale_x(1.0)
    body = [
        f'<text x="24" y="36" font-size="22" font-family="Segoe UI">{escape(title)}</text>',
        (
            f'<text x="24" y="60" font-size="13" font-family="Segoe UI" fill="#5c5140">'
            f"{escape(f'Relative mean loss on log scale with {benchmark_label} = 1.')}</text>"
        ),
        (
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            'fill="#ffffff" stroke="#d9d3c7" />'
        ),
        (
            f'<line x1="{benchmark_x:.2f}" y1="{margin_top}" x2="{benchmark_x:.2f}" '
            f'y2="{margin_top + plot_height}" stroke="#9b2226" stroke-width="1.5" '
            'stroke-dasharray="5 4" />'
        ),
        (
            f'<text x="{benchmark_x:.2f}" y="{margin_top - 12}" text-anchor="middle" '
            'font-size="12" font-family="Consolas" fill="#9b2226">benchmark</text>'
        ),
        (
            f'<text x="{margin_left}" y="{height - 18}" font-size="14" '
            'font-family="Segoe UI">Relative mean loss</text>'
        ),
    ]

    for exponent in range(min_exp, max_exp + 1):
        tick_value = 10.0**exponent
        tick_x = scale_x(tick_value)
        body.extend(
            [
                (
                    f'<line x1="{tick_x:.2f}" y1="{margin_top}" x2="{tick_x:.2f}" '
                    f'y2="{margin_top + plot_height}" stroke="#eee6d8" />'
                ),
                (
                    f'<text x="{tick_x:.2f}" y="{margin_top + plot_height + 18}" '
                    f'text-anchor="middle" font-size="12" font-family="Consolas">'
                    f"{escape(_format_axis_tick(tick_value))}</text>"
                ),
            ]
        )

    for index, (label, ratio) in enumerate(zip(labels, ratios, strict=True)):
        y = margin_top + 18 + (index * 44)
        dot_x = scale_x(ratio)
        color = PALETTE[index % len(PALETTE)]
        body.extend(
            [
                (
                    f'<text x="24" y="{y + 4}" font-size="14" font-family="Segoe UI">'
                    f"{escape(label)}</text>"
                ),
                (
                    f'<line x1="{benchmark_x:.2f}" y1="{y:.2f}" x2="{dot_x:.2f}" y2="{y:.2f}" '
                    'stroke="#c8c0b0" stroke-width="2" />'
                ),
                (
                    f'<circle cx="{dot_x:.2f}" cy="{y:.2f}" r="5.5" fill="{color}" '
                    'stroke="#fcfbf7" stroke-width="1.5" />'
                ),
                (
                    f'<text x="{width - 24}" y="{y + 4:.2f}" text-anchor="end" '
                    f'font-size="13" font-family="Consolas">{ratio:.2f}x</text>'
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


def write_surface_heatmap(
    frame: pl.DataFrame,
    *,
    x_column: str,
    y_column: str,
    winner_column: str,
    improvement_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    """Write a surface heatmap with winner labels and benchmark-relative shading."""

    required_columns = {x_column, y_column, winner_column, improvement_column}
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        message = f"Surface heatmap frame is missing required columns: {missing_columns}."
        raise ValueError(message)
    if frame.is_empty():
        return _write_svg(
            output_path,
            _svg_wrapper(800, 180, [f'<text x="24" y="48" font-size="18">{escape(title)}</text>']),
        )

    x_values = sorted({float(value) for value in frame[x_column].to_list()})
    y_values = sorted({float(value) for value in frame[y_column].to_list()})
    max_improvement = max(
        float(value) for value in frame[improvement_column].fill_null(0.0).to_list()
    )

    width = 1140
    height = 760
    margin_left = 118
    margin_right = 270
    margin_top = 104
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    cell_width = plot_width / max(len(x_values), 1)
    cell_height = plot_height / max(len(y_values), 1)

    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    present_models = [
        str(value)
        for value in frame[winner_column].unique(maintain_order=True).to_list()
    ]

    body = [
        (
            f'<text x="24" y="36" font-size="22" font-family="Segoe UI">{escape(title)}</text>'
        ),
        (
            '<text x="24" y="60" font-size="13" font-family="Segoe UI" fill="#5c5140">'
            'Cell fill shows improvement versus naive; cell text shows the winning model.</text>'
        ),
        (
            f'<text x="{margin_left}" y="{height - 22}" font-size="14" '
            f'font-family="Segoe UI">{escape(x_label)}</text>'
        ),
        (
            f'<text x="24" y="{margin_top - 22}" font-size="14" '
            f'font-family="Segoe UI">{escape(y_label)}</text>'
        ),
        (
            f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" '
            'fill="#ffffff" stroke="#d9d3c7" />'
        ),
    ]

    for x_value in x_values:
        x_pos = margin_left + (x_lookup[x_value] * cell_width)
        body.append(

                f'<text x="{x_pos + (cell_width / 2):.2f}" '
                f'y="{margin_top + plot_height + 24}" text-anchor="middle" font-size="12" '
                f'font-family="Consolas">{x_value:+.2f}</text>'

        )

    for y_value in y_values:
        y_pos = margin_top + (y_lookup[y_value] * cell_height)
        body.append(

                f'<text x="{margin_left - 12}" y="{y_pos + (cell_height / 2) + 4:.2f}" '
                f'text-anchor="end" font-size="12" font-family="Consolas">{int(y_value)}</text>'

        )

    for row in frame.iter_rows(named=True):
        x_value = float(row[x_column])
        y_value = float(row[y_column])
        winner = str(row[winner_column])
        improvement = float(row[improvement_column])
        fraction = 0.0 if max_improvement <= 0.0 else improvement / max_improvement
        fill_color = _interpolate_hex_color("#f3ede2", "#005f73", fraction)
        text_color = _text_color_for_fill(fill_color)
        x_pos = margin_left + (x_lookup[x_value] * cell_width)
        y_pos = margin_top + (y_lookup[y_value] * cell_height)
        label_x = x_pos + (cell_width / 2)
        body.extend(
            [
                (
                    f'<rect x="{x_pos:.2f}" y="{y_pos:.2f}" width="{cell_width:.2f}" '
                    f'height="{cell_height:.2f}" fill="{fill_color}" stroke="#fcfbf7" />'
                ),
                (
                    f'<text x="{label_x:.2f}" y="{y_pos + (cell_height / 2) - 4:.2f}" '
                    f'text-anchor="middle" font-size="12" font-family="Segoe UI" '
                    f'fill="{text_color}">{escape(_model_abbreviation(winner))}</text>'
                ),
                (
                    f'<text x="{label_x:.2f}" y="{y_pos + (cell_height / 2) + 12:.2f}" '
                    f'text-anchor="middle" font-size="10" font-family="Consolas" '
                    f'fill="{text_color}">+{improvement:.1f}%</text>'
                ),
            ]
        )

    legend_x = width - margin_right + 26
    legend_y = margin_top + 12
    body.extend(
        [
            "<defs>",
            '<linearGradient id="surface-improvement-scale" x1="0%" y1="0%" x2="100%" y2="0%">',
            '<stop offset="0%" stop-color="#f3ede2" />',
            '<stop offset="100%" stop-color="#005f73" />',
            "</linearGradient>",
            "</defs>",
            (
                f'<text x="{legend_x}" y="{legend_y}" font-size="14" font-family="Segoe UI">'
                "Improvement vs naive</text>"
            ),
            (
                f'<rect x="{legend_x}" y="{legend_y + 14}" width="170" height="14" '
                'fill="url(#surface-improvement-scale)" stroke="#d9d3c7" />'
            ),
            (
                f'<text x="{legend_x}" y="{legend_y + 44}" font-size="12" '
                'font-family="Consolas">0.0%</text>'
            ),
            (
                f'<text x="{legend_x + 170}" y="{legend_y + 44}" text-anchor="end" '
                f'font-size="12" font-family="Consolas">{max_improvement:.1f}%</text>'
            ),
            (
                f'<text x="{legend_x}" y="{legend_y + 82}" font-size="14" font-family="Segoe UI">'
                "Winner labels</text>"
            ),
        ]
    )

    for index, model_name in enumerate(present_models):
        body.append(

                f'<text x="{legend_x}" y="{legend_y + 106 + (index * 22)}" '
                f'font-size="12" font-family="Segoe UI">{escape(_model_abbreviation(model_name))}'
                f" = {escape(model_name)}</text>"

        )

    return _write_svg(output_path, _svg_wrapper(width, height, body))


def write_ecdf_chart(
    frame: pl.DataFrame,
    *,
    value_column: str,
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> Path:
    """Write a simple empirical CDF chart for one numeric column."""

    if value_column not in frame.columns:
        message = f"ECDF chart requires column {value_column!r}."
        raise ValueError(message)
    values = sorted(float(value) for value in frame[value_column].to_list())
    if not values:
        return _write_svg(
            output_path,
            _svg_wrapper(800, 180, [f'<text x="24" y="48" font-size="18">{escape(title)}</text>']),
        )
    if any(not math.isfinite(value) for value in values):
        message = "ECDF chart values must be finite."
        raise ValueError(message)

    width = 960
    height = 520
    margin_left = 88
    margin_right = 64
    margin_top = 72
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    min_x = min(values)
    max_x = max(values)
    x_span = max(max_x - min_x, 1.0e-12)

    def scale_x(value: float) -> float:
        return margin_left + ((value - min_x) / x_span) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + plot_height - (value * plot_height)

    ecdf_points = [
        (scale_x(value), scale_y((index + 1) / len(values)))
        for index, value in enumerate(values)
    ]
    polyline = " ".join(f"{x_pos:.2f},{y_pos:.2f}" for x_pos, y_pos in ecdf_points)
    median = values[len(values) // 2]
    p95 = values[min(len(values) - 1, math.ceil(len(values) * 0.95) - 1)]

    body = [
        f'<text x="24" y="36" font-size="22" font-family="Segoe UI">{escape(title)}</text>',
        (
            f'<text x="24" y="60" font-size="13" font-family="Segoe UI" fill="#5c5140">'
            f"Median {median:.6f}; 95th percentile {p95:.6f}</text>"
        ),
        (
            f'<text x="{margin_left}" y="{height - 18}" font-size="14" '
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
    ]

    for tick_index in range(6):
        y_value = tick_index / 5
        y_pos = scale_y(y_value)
        body.extend(
            [
                (
                    f'<line x1="{margin_left}" y1="{y_pos:.2f}" x2="{margin_left + plot_width}" '
                    f'y2="{y_pos:.2f}" stroke="#f0ece3" />'
                ),
                (
                    f'<text x="{margin_left - 10}" y="{y_pos + 4:.2f}" text-anchor="end" '
                    f'font-size="12" font-family="Consolas">{y_value:.1f}</text>'
                ),
            ]
        )

    x_ticks = 5
    for tick_index in range(x_ticks + 1):
        x_value = min_x + ((x_span / x_ticks) * tick_index)
        x_pos = scale_x(x_value)
        body.extend(
            [
                (
                    f'<line x1="{x_pos:.2f}" y1="{margin_top}" x2="{x_pos:.2f}" '
                    f'y2="{margin_top + plot_height}" stroke="#f0ece3" />'
                ),
                (
                    f'<text x="{x_pos:.2f}" y="{margin_top + plot_height + 18}" '
                    f'text-anchor="middle" font-size="12" font-family="Consolas">'
                    f"{x_value:.4f}</text>"
                ),
            ]
        )

    body.append(
        f'<polyline fill="none" stroke="{PALETTE[0]}" stroke-width="2.5" points="{polyline}" />'
    )
    return _write_svg(output_path, _svg_wrapper(width, height, body))

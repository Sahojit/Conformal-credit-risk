"""Matplotlib figures for the two key visuals: the naive reliability plot and
the coverage calibration chart. Kept separate from the modules that compute
the underlying numbers so those stay usable in a non-plotting context (CLI
JSON output, the dashboard, tests) without a matplotlib dependency.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd


def plot_reliability_curve(
    calibration_table: pd.DataFrame, title: str = "Reliability of Raw Predicted Probability"
) -> matplotlib.figure.Figure:
    """Predicted probability (x) vs. observed default rate (y), per bucket.

    A perfectly calibrated model's points sit on the diagonal. This is the
    "naive baseline is wrong" picture -- see naive_baseline.py.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
    ax.plot(
        calibration_table["mean_predicted_probability"],
        calibration_table["observed_default_rate"],
        marker="o",
        color="crimson",
        label="model (raw probability)",
    )
    ax.set_xlabel("mean predicted probability in bucket")
    ax.set_ylabel("observed default rate in bucket")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    return fig


_METHOD_STYLE = {
    "naive": {"color": "crimson", "label": "naive (raw probability)"},
    "standard_conformal": {"color": "steelblue", "label": "standard conformal"},
    "mondrian_conformal": {"color": "seagreen", "label": "Mondrian conformal"},
}


def plot_coverage_calibration(
    curve_table: pd.DataFrame, title: str = "Coverage Calibration"
) -> matplotlib.figure.Figure:
    """Nominal (requested) coverage vs. empirical (observed) coverage, per method.

    A correct method's points sit on the diagonal: ask for 90% coverage, get
    90% coverage. This is the project's headline result -- it's the chart
    that makes "does the guarantee actually hold" a single glance instead of
    a table of numbers.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    lower_bound = min(curve_table["nominal_coverage"].min(), curve_table["empirical_coverage"].min()) - 0.05
    upper_bound = max(curve_table["nominal_coverage"].max(), curve_table["empirical_coverage"].max()) + 0.05
    ax.plot([lower_bound, upper_bound], [lower_bound, upper_bound], linestyle="--", color="gray", label="perfect calibration")

    for method, group in curve_table.groupby("method"):
        style = _METHOD_STYLE.get(method, {})
        group = group.sort_values("nominal_coverage")
        ax.plot(
            group["nominal_coverage"],
            group["empirical_coverage"],
            marker="o",
            color=style.get("color"),
            label=style.get("label", method),
        )

    ax.set_xlabel("nominal (requested) coverage")
    ax.set_ylabel("empirical (observed) coverage")
    ax.set_title(title)
    ax.set_xlim(lower_bound, upper_bound)
    ax.set_ylim(lower_bound, upper_bound)
    ax.legend()
    fig.tight_layout()
    return fig

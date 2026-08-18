"""Sweeps target coverage levels and tabulates nominal vs. empirical coverage.

This is the data behind the project's headline chart: for a well-calibrated
method, a plot of (nominal, empirical) points should sit on the diagonal. The
naive baseline should not; standard and Mondrian conformal prediction should.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_credit_risk.conformal import run_split_conformal
from conformal_credit_risk.mondrian import run_mondrian_conformal
from conformal_credit_risk.naive_baseline import naive_prediction_sets


def _empirical_coverage(prediction_sets: list[frozenset[int]], y_true: np.ndarray) -> float:
    hits = [int(y) in pred_set for y, pred_set in zip(y_true, prediction_sets)]
    return float(np.mean(hits))


def build_coverage_curve(
    calibration_predicted_probability: np.ndarray,
    y_calibration: np.ndarray,
    test_predicted_probability: np.ndarray,
    y_test: np.ndarray,
    coverage_levels: list[float],
    calibration_groups: np.ndarray | None = None,
    test_groups: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return a tidy table of (method, nominal_coverage, empirical_coverage).

    Mondrian rows are only included when group arrays are supplied, since
    Mondrian coverage needs a group column to condition on.
    """
    rows = []

    for coverage_level in coverage_levels:
        naive_sets = naive_prediction_sets(test_predicted_probability, coverage_level)
        rows.append(
            {
                "method": "naive",
                "nominal_coverage": coverage_level,
                "empirical_coverage": _empirical_coverage(naive_sets, y_test),
            }
        )

        standard_result = run_split_conformal(
            calibration_predicted_probability, y_calibration, test_predicted_probability, coverage_level
        )
        rows.append(
            {
                "method": "standard_conformal",
                "nominal_coverage": coverage_level,
                "empirical_coverage": standard_result.empirical_coverage(y_test),
            }
        )

        if calibration_groups is not None and test_groups is not None:
            mondrian_result = run_mondrian_conformal(
                calibration_predicted_probability,
                y_calibration,
                calibration_groups,
                test_predicted_probability,
                test_groups,
                coverage_level,
            )
            rows.append(
                {
                    "method": "mondrian_conformal",
                    "nominal_coverage": coverage_level,
                    "empirical_coverage": mondrian_result.empirical_coverage(y_test),
                }
            )

    return pd.DataFrame(rows)

"""Group-conditional (Mondrian) conformal prediction.

Standard split conformal only guarantees coverage *on average* across the
whole test set. It says nothing about any particular subgroup: it's entirely
possible to hit 90% coverage overall while undercovering, say, low-income
applicants and overcovering high-income ones -- the errors just average out.
In lending that's a real fairness problem, not just a statistical curiosity:
the group with worse coverage is the group whose risk estimates you can
trust the least, and it's rarely the group evenly distributed by chance.

Mondrian conformal prediction (Vovk et al.) fixes this by computing a
separate nonconformity quantile *within each group*, so the coverage
guarantee holds group-by-group instead of only in aggregate. It costs
statistical power -- each group's threshold is fit on a smaller calibration
sample -- but that's the honest price of a guarantee that actually holds
where it's checked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from conformal_credit_risk.conformal import (
    build_prediction_sets,
    compute_nonconformity_scores,
    compute_quantile_threshold,
)


def per_group_coverage(
    prediction_sets: list[frozenset[int]], y_true: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    """Break down empirical coverage and set size by group, for any set of
    conformal prediction sets -- standard or Mondrian.

    This is what makes the standard-conformal failure mode visible: computed
    on a standard conformal result, it shows the per-group coverage spread
    even though the overall coverage looks fine.
    """
    if not (len(prediction_sets) == len(y_true) == len(groups)):
        raise ValueError(
            "prediction_sets, y_true, and groups must be the same length, got "
            f"{len(prediction_sets)}, {len(y_true)}, {len(groups)}"
        )

    rows = pd.DataFrame(
        {
            "group": groups,
            "covered": [
                int(y) in pred_set for y, pred_set in zip(y_true, prediction_sets)
            ],
            "set_size": [len(pred_set) for pred_set in prediction_sets],
        }
    )
    return (
        rows.groupby("group", observed=True)
        .agg(coverage=("covered", "mean"), mean_set_size=("set_size", "mean"), count=("covered", "size"))
        .reset_index()
    )


@dataclass
class MondrianResult:
    """Output of a Mondrian conformal run: one threshold per group."""

    coverage_level: float
    thresholds_by_group: dict[str, float]
    prediction_sets: list[frozenset[int]]
    test_groups: np.ndarray

    def empirical_coverage(self, y_true: np.ndarray) -> float:
        hits = [
            int(y) in pred_set for y, pred_set in zip(y_true, self.prediction_sets)
        ]
        return float(np.mean(hits))

    def per_group_coverage(self, y_true: np.ndarray) -> pd.DataFrame:
        return per_group_coverage(self.prediction_sets, y_true, self.test_groups)


def run_mondrian_conformal(
    calibration_predicted_probability: np.ndarray,
    y_calibration: np.ndarray,
    calibration_groups: np.ndarray,
    test_predicted_probability: np.ndarray,
    test_groups: np.ndarray,
    coverage_level: float,
) -> MondrianResult:
    """Fit a separate nonconformity quantile per group, then apply each test
    row's own group's threshold.

    Every group present in the test set must also appear in the calibration
    set -- there's no way to fit a threshold for a group with zero calibration
    examples, and silently falling back to a pooled threshold would defeat
    the point of computing group-conditional coverage in the first place.
    """
    calibration_groups = np.asarray(calibration_groups)
    test_groups = np.asarray(test_groups)

    calibration_only_groups = set(calibration_groups.tolist())
    test_only_groups = set(test_groups.tolist())
    missing_from_calibration = test_only_groups - calibration_only_groups
    if missing_from_calibration:
        raise ValueError(
            f"group(s) {missing_from_calibration} appear in the test set but not "
            "the calibration set -- cannot fit a Mondrian threshold for them"
        )

    thresholds_by_group: dict[str, float] = {}
    for group in sorted(calibration_only_groups):
        mask = calibration_groups == group
        group_scores = compute_nonconformity_scores(
            calibration_predicted_probability[mask], y_calibration[mask]
        )
        thresholds_by_group[group] = compute_quantile_threshold(group_scores, coverage_level)

    prediction_sets: list[frozenset[int]] = []
    for i, group in enumerate(test_groups):
        threshold = thresholds_by_group[group]
        pred_set = build_prediction_sets(
            test_predicted_probability[i : i + 1], threshold
        )[0]
        prediction_sets.append(pred_set)

    return MondrianResult(
        coverage_level=coverage_level,
        thresholds_by_group=thresholds_by_group,
        prediction_sets=prediction_sets,
        test_groups=test_groups,
    )

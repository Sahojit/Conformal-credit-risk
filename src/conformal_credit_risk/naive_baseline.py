"""Shows why treating a raw model probability as a calibrated confidence is wrong.

This is the "before" picture the rest of the project argues against: bucket
predicted probabilities into deciles and compare each bucket's average
predicted probability against the actual observed default rate in that
bucket. If the model's 0.70 genuinely meant "70% of these applicants
default," the two would track each other closely. They don't -- and the gap
is often *worse*, not better, after a model has been rebalanced for class
imbalance (see the scale_pos_weight note in model.py), because rebalancing
inflates predicted probabilities to fix the decision boundary, not to make
the output usable as a probability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conformal_credit_risk.conformal import build_prediction_sets


def compute_calibration_table(
    y_true: np.ndarray, predicted_probability: np.ndarray, n_buckets: int = 10
) -> pd.DataFrame:
    """Bucket predictions into deciles and compare mean predicted vs. observed rate.

    Uses qcut (equal-count buckets) rather than equal-width buckets, since this
    model's predicted probabilities cluster in a narrow range -- equal-width
    buckets would leave several buckets nearly empty and unreliable to average
    over.
    """
    if len(y_true) != len(predicted_probability):
        raise ValueError(
            f"y_true and predicted_probability must be the same length, got "
            f"{len(y_true)} and {len(predicted_probability)}"
        )

    bucket = pd.qcut(predicted_probability, q=n_buckets, duplicates="drop")
    table = pd.DataFrame(
        {"bucket": bucket, "predicted_probability": predicted_probability, "actual_outcome": y_true}
    )

    summary = (
        table.groupby("bucket", observed=True)
        .agg(
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_default_rate=("actual_outcome", "mean"),
            count=("actual_outcome", "size"),
        )
        .reset_index(drop=True)
    )
    return summary


def mean_calibration_gap(calibration_table: pd.DataFrame) -> float:
    """Average absolute gap between predicted probability and observed rate.

    A single number to headline the README with -- close to 0 means the raw
    probabilities are trustworthy as confidence measures, large means they aren't.
    """
    gap = (
        calibration_table["mean_predicted_probability"]
        - calibration_table["observed_default_rate"]
    ).abs()
    return float(gap.mean())


def naive_prediction_sets(
    predicted_probability_of_positive_class: np.ndarray, coverage_level: float
) -> list[frozenset[int]]:
    """The naive baseline for the coverage chart: skip calibration entirely.

    This is what "trusting the raw model probability" looks like when
    someone wants, say, 90% coverage: they use 0.10 as the cutoff directly,
    assuming the model's probability scale already means what it says. No
    calibration set is involved -- which is exactly the mistake this whole
    project argues against, and the coverage chart shows how far off it runs.
    """
    if not 0.0 < coverage_level < 1.0:
        raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
    naive_threshold = 1.0 - coverage_level
    return build_prediction_sets(predicted_probability_of_positive_class, naive_threshold)

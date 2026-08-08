import numpy as np
import pytest

from conformal_credit_risk.naive_baseline import (
    compute_calibration_table,
    mean_calibration_gap,
    naive_prediction_sets,
)


def test_compute_calibration_table_bucket_counts_cover_all_rows():
    rng = np.random.default_rng(0)
    n = 1000
    predicted_probability = rng.uniform(0, 1, size=n)
    y_true = (rng.random(n) < predicted_probability).astype(int)

    table = compute_calibration_table(y_true, predicted_probability, n_buckets=10)

    assert table["count"].sum() == n
    assert len(table) == 10


def test_compute_calibration_table_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_calibration_table(np.array([0, 1]), np.array([0.5]))


def test_mean_calibration_gap_is_zero_for_perfectly_calibrated_predictions():
    # every bucket's mean predicted probability exactly matches its observed rate
    import pandas as pd

    table = pd.DataFrame(
        {
            "mean_predicted_probability": [0.1, 0.5, 0.9],
            "observed_default_rate": [0.1, 0.5, 0.9],
            "count": [100, 100, 100],
        }
    )
    assert mean_calibration_gap(table) == pytest.approx(0.0)


def test_naive_prediction_sets_hand_checked():
    # coverage_level=0.8 -> naive threshold = 1 - 0.8 = 0.2, same threshold
    # logic as build_prediction_sets: include label if P(label) >= 1 - threshold = 0.8
    predicted_probability = np.array([0.9, 0.5, 0.1])
    sets = naive_prediction_sets(predicted_probability, coverage_level=0.8)

    assert sets[0] == frozenset({1})
    assert sets[1] == frozenset()
    assert sets[2] == frozenset({0})

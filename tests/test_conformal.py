import numpy as np
import pytest

from conformal_credit_risk.conformal import (
    build_prediction_sets,
    compute_nonconformity_scores,
    compute_quantile_threshold,
    run_split_conformal,
)


def test_compute_nonconformity_scores_hand_checked():
    predicted_probability_of_positive = np.array([0.9, 0.3, 0.5])
    y_true = np.array([1, 0, 1])

    # row 0: predicted P(true label=1) = 0.9 -> score 0.1
    # row 1: predicted P(true label=0) = 1 - 0.3 = 0.7 -> score 0.3
    # row 2: predicted P(true label=1) = 0.5 -> score 0.5
    expected = np.array([0.1, 0.3, 0.5])

    scores = compute_nonconformity_scores(predicted_probability_of_positive, y_true)
    np.testing.assert_allclose(scores, expected)


def test_compute_nonconformity_scores_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_nonconformity_scores(np.array([0.1, 0.2]), np.array([1]))


def test_compute_quantile_threshold_hand_checked():
    # n=4 calibration scores. With the (n+1)*coverage finite-sample correction:
    #   coverage=0.25 -> quantile_level = ceil(5*0.25)/4 = ceil(1.25)/4 = 2/4 = 0.5
    #   np.quantile([0.1, 0.4, 0.6, 0.9], 0.5, method="higher") = 0.6
    scores = np.array([0.9, 0.1, 0.6, 0.4])
    threshold = compute_quantile_threshold(scores, coverage_level=0.25)
    assert threshold == pytest.approx(0.6)


def test_compute_quantile_threshold_saturates_at_max_score():
    # coverage=0.5 -> quantile_level = ceil(5*0.5)/4 = ceil(2.5)/4 = 3/4 = 0.75
    # np.quantile([0.1, 0.4, 0.6, 0.9], 0.75, method="higher") = 0.9 (the max)
    scores = np.array([0.9, 0.1, 0.6, 0.4])
    threshold = compute_quantile_threshold(scores, coverage_level=0.5)
    assert threshold == pytest.approx(0.9)


def test_compute_quantile_threshold_rejects_invalid_coverage():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        compute_quantile_threshold(np.array([0.1, 0.2]), coverage_level=1.0)


def test_compute_quantile_threshold_rejects_empty_calibration_set():
    with pytest.raises(ValueError, match="empty calibration set"):
        compute_quantile_threshold(np.array([]), coverage_level=0.9)


def test_build_prediction_sets_hand_checked():
    predicted_probability_of_positive = np.array([0.9, 0.5, 0.05])
    threshold = 0.6  # minimum probability to include a label = 1 - 0.6 = 0.4

    sets = build_prediction_sets(predicted_probability_of_positive, threshold)

    assert sets[0] == frozenset({1})  # P(1)=0.9 clears bar, P(0)=0.1 doesn't
    assert sets[1] == frozenset({0, 1})  # both P(0)=0.5 and P(1)=0.5 clear bar
    assert sets[2] == frozenset({0})  # P(0)=0.95 clears bar, P(1)=0.05 doesn't


def test_split_conformal_coverage_holds_on_simulated_data():
    """The core guarantee, checked against ground truth: build a synthetic
    problem where the model's predicted probability equals the *true*
    data-generating probability, draw labels from it, and confirm empirical
    coverage on a held-out test set lands close to the requested level.

    A large n is used because the coverage guarantee is exact only in
    expectation over the randomness of the calibration split -- a single
    finite calibration/test split will have some sampling noise around the
    target.
    """
    rng = np.random.default_rng(seed=42)
    n_calibration = 5000
    n_test = 5000
    coverage_level = 0.90

    calibration_probability = rng.uniform(0.05, 0.95, size=n_calibration)
    y_calibration = (rng.random(n_calibration) < calibration_probability).astype(int)

    test_probability = rng.uniform(0.05, 0.95, size=n_test)
    y_test = (rng.random(n_test) < test_probability).astype(int)

    result = run_split_conformal(calibration_probability, y_calibration, test_probability, coverage_level)
    empirical = result.empirical_coverage(y_test)

    assert empirical == pytest.approx(coverage_level, abs=0.02)


def test_split_conformal_wider_target_gives_larger_sets():
    rng = np.random.default_rng(seed=1)
    n = 2000
    calibration_probability = rng.uniform(0.1, 0.9, size=n)
    y_calibration = (rng.random(n) < calibration_probability).astype(int)
    test_probability = rng.uniform(0.1, 0.9, size=n)

    narrow = run_split_conformal(calibration_probability, y_calibration, test_probability, 0.80)
    wide = run_split_conformal(calibration_probability, y_calibration, test_probability, 0.95)

    assert wide.mean_set_size() >= narrow.mean_set_size()

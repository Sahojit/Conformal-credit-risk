import numpy as np
import pytest

from conformal_credit_risk.mondrian import per_group_coverage, run_mondrian_conformal


def test_per_group_coverage_hand_checked():
    prediction_sets = [frozenset({1}), frozenset({0}), frozenset({0, 1}), frozenset({1})]
    y_true = np.array([1, 1, 0, 0])  # covered: True, False, True, False
    groups = np.array(["a", "a", "b", "b"])

    table = per_group_coverage(prediction_sets, y_true, groups)
    table = table.set_index("group")

    assert table.loc["a", "coverage"] == pytest.approx(0.5)  # 1 of 2 covered
    assert table.loc["b", "coverage"] == pytest.approx(0.5)  # 1 of 2 covered
    assert table.loc["a", "count"] == 2
    assert table.loc["a", "mean_set_size"] == pytest.approx(1.0)  # both size-1 sets
    assert table.loc["b", "mean_set_size"] == pytest.approx(1.5)  # sizes 2 and 1


def test_per_group_coverage_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        per_group_coverage([frozenset({0})], np.array([0, 1]), np.array(["a", "a"]))


def test_mondrian_raises_when_test_group_missing_from_calibration():
    calibration_probability = np.array([0.2, 0.3, 0.4])
    y_calibration = np.array([0, 1, 0])
    calibration_groups = np.array(["a", "a", "a"])
    test_probability = np.array([0.5])
    test_groups = np.array(["b"])  # never seen in calibration

    with pytest.raises(ValueError, match="not the calibration set"):
        run_mondrian_conformal(
            calibration_probability, y_calibration, calibration_groups, test_probability, test_groups, 0.9
        )


def test_mondrian_handles_single_example_group():
    """A group with exactly one calibration example is a degenerate but valid
    case: the finite-sample quantile formula still resolves to that example's
    own score as the threshold, rather than crashing or silently dropping
    the group.
    """
    rng = np.random.default_rng(0)
    n = 200
    calibration_probability = rng.uniform(0.1, 0.9, size=n)
    y_calibration = (rng.random(n) < calibration_probability).astype(int)
    calibration_groups = np.array(["common"] * (n - 1) + ["rare"])

    test_probability = rng.uniform(0.1, 0.9, size=10)
    test_groups = np.array(["common"] * 9 + ["rare"])

    result = run_mondrian_conformal(
        calibration_probability, y_calibration, calibration_groups, test_probability, test_groups, 0.9
    )

    assert "rare" in result.thresholds_by_group
    assert len(result.prediction_sets) == 10


def test_mondrian_coverage_holds_per_group_on_simulated_data():
    """Two groups with deliberately different score distributions: a naive
    pooled threshold would over-cover one and under-cover the other, but
    each group's own Mondrian threshold should hit its own target.
    """
    rng = np.random.default_rng(7)
    n_per_group = 4000
    coverage_level = 0.90

    def make_group(probability_low, probability_high, n):
        probability = rng.uniform(probability_low, probability_high, size=n)
        labels = (rng.random(n) < probability).astype(int)
        return probability, labels

    cal_prob_a, cal_y_a = make_group(0.05, 0.3, n_per_group)
    cal_prob_b, cal_y_b = make_group(0.5, 0.95, n_per_group)
    test_prob_a, test_y_a = make_group(0.05, 0.3, n_per_group)
    test_prob_b, test_y_b = make_group(0.5, 0.95, n_per_group)

    calibration_probability = np.concatenate([cal_prob_a, cal_prob_b])
    y_calibration = np.concatenate([cal_y_a, cal_y_b])
    calibration_groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)

    test_probability = np.concatenate([test_prob_a, test_prob_b])
    y_test = np.concatenate([test_y_a, test_y_b])
    test_groups = np.array(["a"] * n_per_group + ["b"] * n_per_group)

    result = run_mondrian_conformal(
        calibration_probability, y_calibration, calibration_groups, test_probability, test_groups, coverage_level
    )
    per_group = result.per_group_coverage(y_test).set_index("group")

    assert per_group.loc["a", "coverage"] == pytest.approx(coverage_level, abs=0.03)
    assert per_group.loc["b", "coverage"] == pytest.approx(coverage_level, abs=0.03)

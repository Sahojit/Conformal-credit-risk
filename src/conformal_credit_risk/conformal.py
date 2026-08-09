"""Split conformal prediction for binary classification, implemented by hand.

The method (often called LAC -- "least ambiguous set-valued classifier",
Sadinle et al. 2019) is:

1. On a calibration set the model has never trained on, score how wrong the
   model was about the *true* label: s = 1 - P(true label).
2. Take the (1-alpha)-quantile of those scores, with a small finite-sample
   correction (see `compute_quantile_threshold`).
3. For a new point, include every label whose predicted probability clears
   that threshold. The resulting set -- which can hold 0, 1, or (for binary
   classification) both labels -- is guaranteed to contain the true label at
   least (1-alpha) of the time, *without assuming anything about the model or
   the data distribution*, only that calibration and test rows are
   exchangeable with each other.

Nothing here is specific to XGBoost or to this dataset: it only needs
predicted class probabilities and true labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalResult:
    """Output of a split conformal run at one target coverage level."""

    coverage_level: float
    quantile_threshold: float
    prediction_sets: list[frozenset[int]]

    def empirical_coverage(self, y_true: np.ndarray) -> float:
        """Fraction of rows where the prediction set actually contains the true label."""
        hits = [
            int(y) in pred_set for y, pred_set in zip(y_true, self.prediction_sets)
        ]
        return float(np.mean(hits))

    def mean_set_size(self) -> float:
        """Average number of labels per prediction set.

        A useful companion to coverage: a method can "achieve" 100% coverage
        trivially by always outputting {0, 1}. Set size is what shows whether
        the intervals are actually informative, not just wide enough to be safe.
        """
        return float(np.mean([len(s) for s in self.prediction_sets]))


def compute_nonconformity_scores(
    predicted_probability_of_positive_class: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """s_i = 1 - P(true label), i.e. how much probability mass the model
    withheld from the label that actually occurred.

    A confident, correct model produces small scores; a confident, wrong
    model produces scores near 1. This is computed only on the calibration
    set -- it needs the true label, which the test set's whole purpose is to
    keep hidden until final evaluation.
    """
    if len(predicted_probability_of_positive_class) != len(y_true):
        raise ValueError(
            "predicted probabilities and y_true must be the same length, got "
            f"{len(predicted_probability_of_positive_class)} and {len(y_true)}"
        )
    predicted_probability_of_true_label = np.where(
        y_true == 1,
        predicted_probability_of_positive_class,
        1.0 - predicted_probability_of_positive_class,
    )
    return 1.0 - predicted_probability_of_true_label


def compute_quantile_threshold(
    nonconformity_scores: np.ndarray, coverage_level: float
) -> float:
    """The calibration-set quantile that gives a finite-sample coverage guarantee.

    Using the plain (1-alpha) quantile of n calibration scores would only give
    the right coverage as n -> infinity. Vovk et al.'s finite-sample
    correction instead takes the ceil((n+1)(1-alpha))/n quantile -- slightly
    higher than naively expected -- which is what makes the (1-alpha)
    coverage guarantee hold exactly for finite calibration sets, not just
    asymptotically. This is the detail most from-scratch implementations get
    wrong.
    """
    if not 0.0 < coverage_level < 1.0:
        raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")

    n = len(nonconformity_scores)
    if n == 0:
        raise ValueError("cannot compute a quantile threshold from an empty calibration set")

    quantile_level = np.ceil((n + 1) * coverage_level) / n
    quantile_level = min(quantile_level, 1.0)
    return float(np.quantile(nonconformity_scores, quantile_level, method="higher"))


def build_prediction_sets(
    predicted_probability_of_positive_class: np.ndarray, quantile_threshold: float
) -> list[frozenset[int]]:
    """Include label k in the set whenever 1 - P(k) <= threshold, i.e. P(k) >= 1 - threshold.

    For binary classification this can produce {}, {0}, {1}, or {0, 1} per row.
    An empty set means the model was confident about both classes and wrong
    about the calibration set often enough that neither passes the bar here --
    rare in practice, but not a bug when it happens.
    """
    probability_of_negative_class = 1.0 - predicted_probability_of_positive_class
    minimum_probability_to_include = 1.0 - quantile_threshold

    prediction_sets: list[frozenset[int]] = []
    for p_negative, p_positive in zip(
        probability_of_negative_class, predicted_probability_of_positive_class
    ):
        included_labels = set()
        if p_negative >= minimum_probability_to_include:
            included_labels.add(0)
        if p_positive >= minimum_probability_to_include:
            included_labels.add(1)
        prediction_sets.append(frozenset(included_labels))
    return prediction_sets


def run_split_conformal(
    calibration_predicted_probability: np.ndarray,
    y_calibration: np.ndarray,
    test_predicted_probability: np.ndarray,
    coverage_level: float,
) -> ConformalResult:
    """Fit the quantile threshold on calibration data, apply it to test data."""
    scores = compute_nonconformity_scores(calibration_predicted_probability, y_calibration)
    threshold = compute_quantile_threshold(scores, coverage_level)
    prediction_sets = build_prediction_sets(test_predicted_probability, threshold)
    return ConformalResult(
        coverage_level=coverage_level,
        quantile_threshold=threshold,
        prediction_sets=prediction_sets,
    )

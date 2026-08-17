"""Cross-checks the hand-rolled split conformal implementation against MAPIE.

This module exists purely as a correctness check on `conformal.py` -- it is
not used anywhere else in the pipeline. MAPIE's `SplitConformalClassifier`
with `conformity_score="lac"` implements the same "least ambiguous
classifier" method used in `conformal.py`, so if the two disagree by more
than floating-point noise, something in the hand-rolled version is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from mapie.classification import SplitConformalClassifier
from sklearn.base import ClassifierMixin


@dataclass
class MapieComparisonResult:
    coverage_level: float
    mapie_empirical_coverage: float
    hand_rolled_empirical_coverage: float
    mapie_mean_set_size: float
    hand_rolled_mean_set_size: float

    @property
    def coverage_difference(self) -> float:
        return abs(self.mapie_empirical_coverage - self.hand_rolled_empirical_coverage)


def run_mapie_split_conformal(
    fitted_model: ClassifierMixin,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    coverage_level: float,
) -> tuple[float, float]:
    """Run MAPIE's split conformal classifier and return (empirical coverage, mean set size).

    `prefit=True` tells MAPIE to use `fitted_model` as-is rather than
    retraining it -- the base model was already trained on the training
    split, and MAPIE's job here is only to calibrate and produce sets.
    """
    mapie_classifier = SplitConformalClassifier(
        estimator=fitted_model,
        confidence_level=coverage_level,
        conformity_score="lac",
        prefit=True,
    )
    mapie_classifier.conformalize(X_calibration, y_calibration)

    _, prediction_sets = mapie_classifier.predict_set(X_test)
    # prediction_sets shape is (n_samples, n_classes, n_confidence_levels);
    # a single confidence level was requested, so squeeze that last axis.
    prediction_sets = prediction_sets[:, :, 0]

    y_test_array = y_test.to_numpy()
    covered = prediction_sets[np.arange(len(y_test_array)), y_test_array]
    empirical_coverage = float(covered.mean())
    mean_set_size = float(prediction_sets.sum(axis=1).mean())

    return empirical_coverage, mean_set_size


def compare_with_hand_rolled(
    fitted_model: ClassifierMixin,
    X_calibration: pd.DataFrame,
    y_calibration: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    coverage_level: float,
    hand_rolled_empirical_coverage: float,
    hand_rolled_mean_set_size: float,
) -> MapieComparisonResult:
    mapie_coverage, mapie_set_size = run_mapie_split_conformal(
        fitted_model, X_calibration, y_calibration, X_test, y_test, coverage_level
    )
    return MapieComparisonResult(
        coverage_level=coverage_level,
        mapie_empirical_coverage=mapie_coverage,
        hand_rolled_empirical_coverage=hand_rolled_empirical_coverage,
        mapie_mean_set_size=mapie_set_size,
        hand_rolled_mean_set_size=hand_rolled_mean_set_size,
    )

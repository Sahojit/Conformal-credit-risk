"""The base model conformal prediction wraps: XGBoost predicting default probability.

Conformal prediction doesn't care how good this model is -- the coverage
guarantee holds regardless, a weaker model just produces wider intervals. So
this module stays deliberately simple: train a reasonable gradient-boosted
tree model and expose its raw probability output, nothing more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from conformal_credit_risk.config import ModelConfig


def train_default_model(
    X_train: pd.DataFrame, y_train: pd.Series, config: ModelConfig
) -> XGBClassifier:
    """Fit an XGBoost classifier on the training split only.

    Uses XGBoost's native categorical handling (enable_categorical=True)
    rather than one-hot encoding, since ORGANIZATION_TYPE alone has ~58
    categories and one-hot encoding every categorical column here would
    balloon the feature count without adding information a tree split can't
    already extract from a native categorical column.
    """
    scale_pos_weight = 1.0
    if config.balance_class_weight:
        positive_count = int(y_train.sum())
        negative_count = len(y_train) - positive_count
        if positive_count == 0:
            raise ValueError(
                "training split has zero positive (default) examples -- "
                "cannot compute a class balance weight"
            )
        scale_pos_weight = negative_count / positive_count

    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=scale_pos_weight,
        random_state=config.random_seed,
        enable_categorical=True,
        tree_method="hist",
        eval_metric="auc",
    )
    model.fit(X_train, y_train)
    return model


def predict_default_probability(model: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Return P(TARGET=1) for each row -- the model's raw, uncalibrated estimate."""
    return model.predict_proba(X)[:, 1]

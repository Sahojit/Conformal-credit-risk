"""Typed configuration for the conformal prediction pipeline.

Every threshold and knob used downstream lives here, with defaults chosen for
this specific dataset and documented reasoning, rather than scattered as magic
numbers through the modules that use them.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class DataConfig(BaseModel):
    """Controls how application_train.csv is loaded, cleaned, and split."""

    target_column: str = "TARGET"
    id_column: str = "SK_ID_CURR"

    # A three-way split, not the usual train/test, because conformal prediction
    # needs data the base model has never seen (calibration) to measure how wrong
    # the model actually is. Reusing training data would make nonconformity scores
    # artificially small -- the model always looks confident on rows it memorized --
    # and the resulting intervals would undercover on genuinely new applicants.
    train_fraction: float = 0.6
    calibration_fraction: float = 0.2
    test_fraction: float = 0.2

    random_seed: int = 42

    # Home Credit has columns that are missing on the large majority of rows
    # (e.g. building/apartment detail fields only filled in for a subset of
    # applicants). Past this threshold we drop the column outright rather than
    # imputing it, since a value imputed for >40% of rows is mostly noise, not
    # signal, and silently keeping it would make the missingness pattern itself
    # (rather than the feature) drive the model.
    max_missing_fraction: float = 0.4

    # Column used for group-conditional (Mondrian) coverage. Income bracket is
    # derived at load time by quantile-binning AMT_INCOME_TOTAL -- raw income is
    # continuous and Mondrian conformal prediction needs discrete groups.
    mondrian_group_column: str = "income_bracket"
    income_bracket_count: int = 4

    @model_validator(mode="after")
    def _fractions_sum_to_one(self) -> "DataConfig":
        total = self.train_fraction + self.calibration_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"train/calibration/test fractions must sum to 1.0, got {total}"
            )
        return self


class ModelConfig(BaseModel):
    """Hyperparameters for the base XGBoost default-probability model.

    Conformal prediction wraps whatever this model outputs -- it doesn't need
    the model itself to be heavily tuned, since the coverage guarantee holds
    regardless of model quality (a worse model just produces wider intervals).
    So these are reasonable defaults for a gradient-boosted tree on tabular
    data with class imbalance, not the result of a tuning sweep.
    """

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_seed: int = 42

    # If True, scale_pos_weight is computed from the training set's actual class
    # ratio (~8% default) instead of left at 1, so the model doesn't collapse
    # to predicting "no default" for everyone.
    balance_class_weight: bool = True


class ConformalConfig(BaseModel):
    """Target coverage levels for split and Mondrian conformal prediction."""

    # The levels swept for the coverage calibration chart -- requested (nominal)
    # coverage on one axis, empirically observed coverage on the other.
    coverage_levels: list[float] = Field(default_factory=lambda: [0.80, 0.90, 0.95])

    # The single level used when a diagnostic needs one concrete choice (e.g.
    # the worked Mondrian under/over-coverage example) rather than a sweep.
    default_coverage: float = 0.90

    @model_validator(mode="after")
    def _levels_in_unit_interval(self) -> "ConformalConfig":
        all_levels = [*self.coverage_levels, self.default_coverage]
        if not all(0.0 < level < 1.0 for level in all_levels):
            raise ValueError("coverage levels must be strictly between 0 and 1")
        return self


class ExperimentConfig(BaseModel):
    """Top-level config bundling data, model, and conformal settings."""

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    conformal: ConformalConfig = Field(default_factory=ConformalConfig)

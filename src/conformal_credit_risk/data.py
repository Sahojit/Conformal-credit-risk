"""Loading, cleaning, and splitting the loan application data.

The public entry point is `load_and_split`, which returns three disjoint
DataFrames (train / calibration / test) plus the fitted `Preprocessor` used to
produce them. Everything else in this module supports that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from conformal_credit_risk.config import DataConfig

# Home Credit uses 365243 as a sentinel for "not currently employed" in
# DAYS_EMPLOYED, instead of a null. Left as-is it reads as someone employed
# 1000 years, which corrupts anything downstream that treats it as a real
# duration -- median imputation, scaling, tree splits, all of it.
_DAYS_EMPLOYED_SENTINEL = 365243

REQUIRED_COLUMNS = ("TARGET",)


class DataValidationError(Exception):
    """Raised when the input CSV doesn't look like usable application data."""


def load_raw_csv(path: str, id_column: str, target_column: str) -> pd.DataFrame:
    """Load a CSV and check it has the minimum shape this pipeline needs.

    Kept separate from cleaning so a malformed file fails fast with a clear
    message, before any of the more expensive preprocessing runs.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise DataValidationError(
            f"no file found at {path!r} -- check the path, or download the "
            "dataset first (see README for the Kaggle CLI command)"
        ) from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(
            f"{path!r} could not be parsed as CSV: {exc}"
        ) from exc

    missing_required = [c for c in (id_column, target_column) if c not in df.columns]
    if missing_required:
        raise DataValidationError(
            f"{path!r} is missing required column(s): {missing_required}. "
            f"Found columns: {list(df.columns)[:10]}..."
        )

    unexpected_targets = set(df[target_column].dropna().unique()) - {0, 1}
    if unexpected_targets:
        raise DataValidationError(
            f"expected {target_column!r} to be binary (0/1), found values: "
            f"{unexpected_targets}"
        )

    return df


def _fix_days_employed_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    if "DAYS_EMPLOYED" not in df.columns:
        return df
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(
        _DAYS_EMPLOYED_SENTINEL, np.nan
    )
    return df


@dataclass
class Preprocessor:
    """Cleans raw application rows into a model-ready feature frame.

    All statistics used for cleaning (which columns to drop, imputation
    values, income bracket boundaries) are computed once via `fit` on the
    training split only, then reused by `transform` on calibration and test.
    Fitting on the full dataset would leak information from calibration/test
    rows into choices like "what counts as a high-income applicant," which
    would quietly inflate how well-calibrated everything downstream looks.
    """

    config: DataConfig
    columns_to_drop: list[str] | None = None
    numeric_medians: pd.Series | None = None
    categorical_columns: list[str] | None = None
    numeric_columns: list[str] | None = None
    income_bracket_edges: np.ndarray | None = None

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        df = _fix_days_employed_sentinel(train_df)

        feature_df = df.drop(columns=[self.config.id_column, self.config.target_column])

        missing_fraction = feature_df.isna().mean()
        self.columns_to_drop = missing_fraction[
            missing_fraction > self.config.max_missing_fraction
        ].index.tolist()
        feature_df = feature_df.drop(columns=self.columns_to_drop)

        self.numeric_columns = feature_df.select_dtypes(include="number").columns.tolist()
        self.categorical_columns = feature_df.select_dtypes(exclude="number").columns.tolist()

        self.numeric_medians = feature_df[self.numeric_columns].median()

        # Quantile edges for income bracket come from training data only, then
        # get reused as fixed cut points on calibration/test -- see class docstring.
        # The outer edges are widened to +-inf: calibration/test can contain an
        # applicant with income below the training minimum or above the training
        # maximum, and such a row should fall into the lowest/highest bracket
        # rather than come out as an unassigned NaN.
        _, edges = pd.qcut(
            df["AMT_INCOME_TOTAL"],
            q=self.config.income_bracket_count,
            retbins=True,
            duplicates="drop",
        )
        edges[0] = -np.inf
        edges[-1] = np.inf
        self.income_bracket_edges = edges

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.columns_to_drop is None:
            raise RuntimeError("Preprocessor.transform called before fit")

        df = _fix_days_employed_sentinel(df)

        income_bracket = pd.cut(
            df["AMT_INCOME_TOTAL"],
            bins=self.income_bracket_edges,
            include_lowest=True,
            labels=[f"bracket_{i}" for i in range(len(self.income_bracket_edges) - 1)],
        )

        feature_df = df.drop(columns=[self.config.id_column, self.config.target_column])
        feature_df = feature_df.drop(columns=self.columns_to_drop)

        feature_df[self.numeric_columns] = feature_df[self.numeric_columns].fillna(
            self.numeric_medians
        )

        for col in self.categorical_columns:
            feature_df[col] = feature_df[col].fillna("Missing").astype("category")

        feature_df[self.config.mondrian_group_column] = income_bracket.astype(str)
        feature_df[self.config.mondrian_group_column] = feature_df[
            self.config.mondrian_group_column
        ].astype("category")

        return feature_df


def load_and_split(
    csv_path: str, config: DataConfig
) -> tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Preprocessor
]:
    """Load the raw CSV, split into train/calibration/test, and clean each split.

    Returns (X_train, y_train, X_calibration, y_calibration, X_test, y_test, preprocessor).

    The split happens on raw data before any cleaning, and is stratified on
    the target so all three sets keep the same ~8% default rate as the full
    population -- otherwise, with a base rate this low, an unlucky split could
    leave the calibration set with too few positive examples to estimate
    nonconformity scores for the minority class reliably.
    """
    raw = load_raw_csv(csv_path, config.id_column, config.target_column)

    train_raw, remainder_raw = train_test_split(
        raw,
        train_size=config.train_fraction,
        stratify=raw[config.target_column],
        random_state=config.random_seed,
    )
    calibration_fraction_of_remainder = config.calibration_fraction / (
        config.calibration_fraction + config.test_fraction
    )
    calibration_raw, test_raw = train_test_split(
        remainder_raw,
        train_size=calibration_fraction_of_remainder,
        stratify=remainder_raw[config.target_column],
        random_state=config.random_seed,
    )

    preprocessor = Preprocessor(config=config).fit(train_raw)

    X_train = preprocessor.transform(train_raw)
    X_calibration = preprocessor.transform(calibration_raw)
    X_test = preprocessor.transform(test_raw)

    y_train = train_raw[config.target_column].reset_index(drop=True)
    y_calibration = calibration_raw[config.target_column].reset_index(drop=True)
    y_test = test_raw[config.target_column].reset_index(drop=True)

    X_train = X_train.reset_index(drop=True)
    X_calibration = X_calibration.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    return X_train, y_train, X_calibration, y_calibration, X_test, y_test, preprocessor

import numpy as np
import pandas as pd
import pytest

from conformal_credit_risk.config import DataConfig
from conformal_credit_risk.data import (
    DataValidationError,
    Preprocessor,
    load_and_split,
    load_raw_csv,
)


def test_load_raw_csv_missing_file():
    with pytest.raises(DataValidationError, match="no file found"):
        load_raw_csv("does_not_exist.csv", id_column="SK_ID_CURR", target_column="TARGET")


def test_load_raw_csv_missing_required_column(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "bad.csv"
    synthetic_application_df.drop(columns=["TARGET"]).to_csv(csv_path, index=False)

    with pytest.raises(DataValidationError, match="missing required column"):
        load_raw_csv(str(csv_path), id_column="SK_ID_CURR", target_column="TARGET")


def test_load_raw_csv_non_binary_target(tmp_path, synthetic_application_df):
    df = synthetic_application_df.copy()
    df.loc[0, "TARGET"] = 2
    csv_path = tmp_path / "bad_target.csv"
    df.to_csv(csv_path, index=False)

    with pytest.raises(DataValidationError, match="binary"):
        load_raw_csv(str(csv_path), id_column="SK_ID_CURR", target_column="TARGET")


def test_load_raw_csv_valid(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "good.csv"
    synthetic_application_df.to_csv(csv_path, index=False)

    df = load_raw_csv(str(csv_path), id_column="SK_ID_CURR", target_column="TARGET")
    assert len(df) == len(synthetic_application_df)


def test_preprocessor_drops_high_missing_columns(synthetic_application_df):
    config = DataConfig(max_missing_fraction=0.4)
    preprocessor = Preprocessor(config=config).fit(synthetic_application_df)

    assert "MOSTLY_MISSING_COLUMN" in preprocessor.columns_to_drop

    transformed = preprocessor.transform(synthetic_application_df)
    assert "MOSTLY_MISSING_COLUMN" not in transformed.columns


def test_preprocessor_fixes_days_employed_sentinel(synthetic_application_df):
    config = DataConfig()
    preprocessor = Preprocessor(config=config).fit(synthetic_application_df)
    transformed = preprocessor.transform(synthetic_application_df)

    # The sentinel should have been treated as missing and imputed away --
    # it must not survive into the model-ready feature frame.
    assert 365243 not in transformed["DAYS_EMPLOYED"].to_numpy()


def test_preprocessor_no_nans_after_transform(synthetic_application_df):
    config = DataConfig()
    preprocessor = Preprocessor(config=config).fit(synthetic_application_df)
    transformed = preprocessor.transform(synthetic_application_df)

    assert not transformed.isna().any().any()


def test_preprocessor_income_bracket_edges_from_train_only_extend_to_outliers(
    synthetic_application_df,
):
    """A row with income above the training set's max shouldn't come out as NaN --
    it should fall in the top bracket. This is the scenario that broke during
    development: qcut's outer edges are exactly the train min/max, so any
    out-of-range calibration/test value used to fail silently into NaN.
    """
    config = DataConfig()
    preprocessor = Preprocessor(config=config).fit(synthetic_application_df)

    extreme_row = synthetic_application_df.iloc[[0]].copy()
    extreme_row["AMT_INCOME_TOTAL"] = synthetic_application_df["AMT_INCOME_TOTAL"].max() * 100

    transformed = preprocessor.transform(extreme_row)
    assert transformed["income_bracket"].notna().all()
    assert transformed["income_bracket"].iloc[0] == f"bracket_{config.income_bracket_count - 1}"


def test_load_and_split_is_disjoint_and_stratified(tmp_path, synthetic_application_df):
    csv_path = tmp_path / "app.csv"
    synthetic_application_df.to_csv(csv_path, index=False)

    config = DataConfig(train_fraction=0.6, calibration_fraction=0.2, test_fraction=0.2)
    X_train, y_train, X_cal, y_cal, X_test, y_test, _ = load_and_split(str(csv_path), config)

    assert len(X_train) + len(X_cal) + len(X_test) == len(synthetic_application_df)

    overall_rate = synthetic_application_df["TARGET"].mean()
    for y_split in (y_train, y_cal, y_test):
        assert y_split.mean() == pytest.approx(overall_rate, abs=0.05)

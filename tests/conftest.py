import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_application_df() -> pd.DataFrame:
    """A small stand-in for application_train.csv with the same quirks:
    an imbalanced binary target, a DAYS_EMPLOYED sentinel value, a mostly-missing
    column, and a categorical column -- enough to exercise every branch of
    the preprocessing logic without needing the real 3.2GB dataset.
    """
    rng = np.random.default_rng(seed=0)
    n_rows = 400

    target = (rng.random(n_rows) < 0.10).astype(int)
    days_employed = rng.integers(-10000, -100, size=n_rows).astype(float)
    # About a third of rows use the known Home Credit sentinel for "not employed."
    sentinel_mask = rng.random(n_rows) < 0.3
    days_employed[sentinel_mask] = 365243

    income = rng.lognormal(mean=11.5, sigma=0.6, size=n_rows)

    mostly_missing = np.full(n_rows, np.nan)
    mostly_missing[:20] = rng.random(20)  # only 5% populated -> should get dropped

    income_type = rng.choice(
        ["Working", "Pensioner", "Commercial associate", "Student"], size=n_rows
    )

    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n_rows),
            "TARGET": target,
            "DAYS_EMPLOYED": days_employed,
            "AMT_INCOME_TOTAL": income,
            "MOSTLY_MISSING_COLUMN": mostly_missing,
            "NAME_INCOME_TYPE": income_type,
        }
    )

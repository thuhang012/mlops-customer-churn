from pathlib import Path
import os

import pandas as pd
import pytest


DATA_PATH = Path("data/raw/netflix_large.csv")
REQUIRED_COLUMNS = {
    "user_id",
    "subscription_plan",
    "monthly_fee",
    "churn_status",
    "watch_time_minutes",
}


@pytest.mark.skipif(not DATA_PATH.exists(), reason="Dataset not available in CI/development environment")
def test_raw_data_exists_and_has_required_columns():
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing required columns: {sorted(missing_cols)}"

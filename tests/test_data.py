from pathlib import Path

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

pytestmark = pytest.mark.fast


def test_data_raw_csv_contains_required_columns():
    assert DATA_PATH.exists(), f"Real dataset not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing required columns: {sorted(missing_cols)}"

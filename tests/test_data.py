from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/netflix_large.csv")
REQUIRED_COLUMNS = {
    "user_id",
    "subscription_plan",
    "monthly_fee",
    "churn_status",
    "watch_time_minutes",
}


def test_raw_data_exists_and_has_required_columns():
    assert DATA_PATH.exists(), f"Real dataset not found: {DATA_PATH}"
    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing required columns: {sorted(missing_cols)}"

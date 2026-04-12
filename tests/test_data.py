from pathlib import Path

import pytest

DATA_PATH = Path("data/raw/netflix_large.csv")


@pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Dataset not available - use DVC to pull",
)
def test_raw_data_exists_and_has_required_columns():
    assert DATA_PATH.exists(), f"Missing dataset: {DATA_PATH}"

    import pandas as pd

    df = pd.read_csv(DATA_PATH)
    required_cols = [
        "user_id",
        "age_group",
        "gender",
        "country",
        "subscription_plan",
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

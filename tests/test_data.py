from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path("data/raw/telcom_churn.csv")
DVC_POINTER_PATH = Path("data/raw/telcom_churn.csv.dvc")
REQUIRED_COLUMNS = {
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
}
TARGET_COLUMN_ALIASES = {"Churn", "churn", "churn_status"}

pytestmark = pytest.mark.fast


def test_data_raw_csv_contains_required_columns():
    if not DATA_PATH.exists():
        pytest.skip(f"Real dataset not pulled yet: {DATA_PATH} (DVC pointer: {DVC_POINTER_PATH})")

    df = pd.read_csv(DATA_PATH)
    assert len(df) > 0, "Dataset is empty"

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    assert not missing_cols, f"Missing required columns: {sorted(missing_cols)}"

    assert TARGET_COLUMN_ALIASES.intersection(df.columns), (
        f"Missing target column. Expected one of: {sorted(TARGET_COLUMN_ALIASES)}"
    )

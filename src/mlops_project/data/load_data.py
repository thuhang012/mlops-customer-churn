from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "customerID",
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
    "Churn",
}


def load_raw_data(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Raw data file not found: {input_path}")

    df = pd.read_csv(input_path)
    df.columns = [str(col).strip() for col in df.columns]

    missing_cols = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing_cols:
        raise ValueError(
            "Input data is missing required columns: "
            f"{missing_cols}. Available columns: {list(df.columns)}"
        )

    return df

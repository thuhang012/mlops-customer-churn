from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "age_group",
    "subscription_start_date",
    "churn_status",
    "watch_time_minutes",
    "completion_percentage",
    "date_watched",
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

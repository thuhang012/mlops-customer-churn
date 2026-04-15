from pathlib import Path

import pandas as pd


CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
INFERENCE_DRIFT_PATH = Path("data/processed/inference_drift_data.csv")


def create_drift_data(
    cleaned_data_path: Path,
    output_path: Path,
    drift_fraction: float = 0.4,
    sample_size: int = 300,
) -> None:
    df = pd.read_csv(cleaned_data_path)

    if df.empty:
        raise ValueError("Cleaned data is empty, cannot create drift dataset.")

    if sample_size >= len(df):
        current_data = df.copy()
    else:
        current_data = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    current_data = current_data.copy()
    current_data["data_split"] = "inference"

    drift_count = int(len(current_data) * drift_fraction)
    drift_idx = current_data.sample(n=drift_count, random_state=42).index

    numeric_columns = [
        "age_group",
        "monthly_fee",
        "watch_time_minutes",
        "completion_percentage",
        "rating",
        "days_since_last_watch",
        "avg_weekly_watch_time",
    ]
    for col in numeric_columns:
        if col in current_data.columns:
            current_data.loc[drift_idx, col] = current_data.loc[drift_idx, col] * 2 + 2

    if "churn_status" in current_data.columns:
        current_data.loc[drift_idx, "churn_status"] = 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_data.to_csv(output_path, index=False)
    print(f"Created inference drift data at: {output_path}")
    print(f"Reference rows: {len(df)}, current rows: {len(current_data)}")
    print(f"Drift rows: {drift_count}")


def main() -> None:
    create_drift_data(CLEANED_DATA_PATH, INFERENCE_DRIFT_PATH)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mlops_project.data.load_data import load_raw_data
from src.mlops_project.data.validate_data import (
    TARGET_COLUMN,
    clean_raw_dataframe,
)
from src.mlops_project.features.build_features import (
    build_preprocessor,
    prepare_feature_inputs,
)


RAW_DATA_PATH = Path("data/raw/netflix_large.csv")
CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
PREPROCESSOR_PATH = Path("artifacts/preprocessors/preprocessor.pkl")

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _split_data(
    feature_df: pd.DataFrame,
    target_series: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None, pd.Series | None]:
    if target_series is None:
        x_train, x_test = train_test_split(
            feature_df,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        return x_train, x_test, None, None

    value_counts = target_series.value_counts(dropna=False)
    can_stratify = target_series.nunique() > 1 and value_counts.min() >= 2
    stratify_target = target_series if can_stratify else None
    x_train, x_test, y_train, y_test = train_test_split(
        feature_df,
        target_series,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_target,
    )
    return x_train, x_test, y_train, y_test


def run_preprocessing(
    raw_data_path: Path,
    cleaned_data_path: Path,
    preprocessor_path: Path,
) -> None:
    raw_df = load_raw_data(raw_data_path)
    validated_df, validation_report = clean_raw_dataframe(raw_df)

    feature_source_df, feature_report = prepare_feature_inputs(validated_df)

    target_series = None
    if TARGET_COLUMN in feature_source_df.columns:
        target_series = feature_source_df[TARGET_COLUMN].copy()
        feature_df = feature_source_df.drop(columns=[TARGET_COLUMN])
    else:
        feature_df = feature_source_df

    x_train, x_test, _, _ = _split_data(feature_df, target_series)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(feature_df)
    preprocessor.fit(x_train)

    transformed = preprocessor.transform(feature_df)
    feature_names = preprocessor.get_feature_names_out()
    cleaned_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=feature_df.index,
    )

    if target_series is not None:
        cleaned_df[TARGET_COLUMN] = target_series.values

    cleaned_df["data_split"] = "train"
    cleaned_df.loc[x_test.index, "data_split"] = "test"
    cleaned_df = cleaned_df.reset_index(drop=True)

    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(cleaned_data_path, index=False)
    joblib.dump(
        {
            "pipeline": preprocessor,
            "feature_columns": feature_df.columns.tolist(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "target_column": TARGET_COLUMN if target_series is not None else None,
            "dropped_columns": feature_report.dropped_columns,
            "leakage_columns_removed": feature_report.leakage_columns_removed,
            "validation_report": validation_report,
            "split_config": {
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "stratify": bool(
                    target_series is not None and target_series.nunique() > 1
                ),
            },
            "train_indices": x_train.index.tolist(),
            "test_indices": x_test.index.tolist(),
        },
        preprocessor_path,
    )

    print(f"Raw shape: {raw_df.shape}")
    print(f"Validated shape: {validated_df.shape}")
    print(f"Model input shape: {feature_df.shape}")
    print(f"Final cleaned shape: {cleaned_df.shape}")
    print(f"Duplicates removed: {validation_report.get('duplicates_removed', 0)}")
    print(f"Leakage columns removed: {feature_report.leakage_columns_removed}")
    print(f"Saved cleaned data to: {cleaned_data_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Netflix dataset for churn modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CLEANED_DATA_PATH,
        help="Path to output CSV.",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=PREPROCESSOR_PATH,
        help="Path to save preprocessor artifact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocessing(args.input, args.output, args.preprocessor)


if __name__ == "__main__":
    main()

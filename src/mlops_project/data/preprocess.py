from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.mlops_project.data.load_data import load_raw_data
from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import (
    TARGET_COLUMN,
    build_linear_preprocessor,
    build_tree_preprocessor,
    prepare_feature_inputs,
)


RAW_DATA_PATH = Path("data/raw/telcom_churn.csv")
CLEANED_TREE_DATA_PATH = Path("data/processed/cleaned_data_tree.csv")
CLEANED_LINEAR_DATA_PATH = Path("data/processed/cleaned_data_linear.csv")
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


def _build_output_dataframe(
    transformed: object,
    feature_names: list[str],
    feature_index: pd.Index,
    target_series: pd.Series | None,
    x_test_index: pd.Index,
) -> pd.DataFrame:
    cleaned_df = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=feature_index,
    )

    if target_series is not None:
        cleaned_df[TARGET_COLUMN] = target_series.values

    cleaned_df["data_split"] = "train"
    cleaned_df.loc[x_test_index, "data_split"] = "test"
    return cleaned_df.reset_index(drop=True)


def run_preprocessing(
    raw_data_path: Path,
    cleaned_tree_data_path: Path,
    cleaned_linear_data_path: Path,
    preprocessor_path: Path,
) -> None:
    raw_df = load_raw_data(raw_data_path)
    validated_df, validation_report = clean_raw_dataframe(
        raw_df,
        strict_schema=True,
        require_target=True,
    )

    feature_source_df, feature_report = prepare_feature_inputs(validated_df)

    target_series = None
    if TARGET_COLUMN in feature_source_df.columns:
        target_series = feature_source_df[TARGET_COLUMN].copy()
        feature_df = feature_source_df.drop(columns=[TARGET_COLUMN])
    else:
        feature_df = feature_source_df

    x_train, x_test, _, _ = _split_data(feature_df, target_series)

    tree_preprocessor, tree_numeric_cols, tree_categorical_cols = build_tree_preprocessor(
        feature_df
    )
    tree_preprocessor.fit(x_train)
    tree_transformed = tree_preprocessor.transform(feature_df)
    tree_feature_names = tree_preprocessor.get_feature_names_out().tolist()
    cleaned_tree_df = _build_output_dataframe(
        transformed=tree_transformed,
        feature_names=tree_feature_names,
        feature_index=feature_df.index,
        target_series=target_series,
        x_test_index=x_test.index,
    )

    linear_preprocessor, linear_numeric_cols, linear_categorical_cols = (
        build_linear_preprocessor(feature_df)
    )
    linear_preprocessor.fit(x_train)
    linear_transformed = linear_preprocessor.transform(feature_df)
    linear_feature_names = linear_preprocessor.get_feature_names_out().tolist()
    cleaned_linear_df = _build_output_dataframe(
        transformed=linear_transformed,
        feature_names=linear_feature_names,
        feature_index=feature_df.index,
        target_series=target_series,
        x_test_index=x_test.index,
    )

    cleaned_tree_data_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_linear_data_path.parent.mkdir(parents=True, exist_ok=True)
    preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned_tree_df.to_csv(cleaned_tree_data_path, index=False)
    cleaned_linear_df.to_csv(cleaned_linear_data_path, index=False)
    joblib.dump(
        {
            # Keep tree pipeline as default for API backward compatibility.
            "pipeline": tree_preprocessor,
            "feature_columns": feature_df.columns.tolist(),
            "numeric_columns": tree_numeric_cols,
            "categorical_columns": tree_categorical_cols,
            "linear_pipeline": linear_preprocessor,
            "linear_numeric_columns": linear_numeric_cols,
            "linear_categorical_columns": linear_categorical_cols,
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
            "output_datasets": {
                "tree": str(cleaned_tree_data_path),
                "linear": str(cleaned_linear_data_path),
            },
        },
        preprocessor_path,
    )

    print(f"Raw shape: {raw_df.shape}")
    print(f"Validated shape: {validated_df.shape}")
    print(f"Model input shape: {feature_df.shape}")
    print(f"Final tree dataset shape: {cleaned_tree_df.shape}")
    print(f"Final linear dataset shape: {cleaned_linear_df.shape}")
    print(f"Duplicates removed: {validation_report.get('duplicates_removed', 0)}")
    print(f"Leakage columns removed: {feature_report.leakage_columns_removed}")
    print(f"Saved tree dataset to: {cleaned_tree_data_path}")
    print(f"Saved linear dataset to: {cleaned_linear_data_path}")
    print(f"Saved preprocessor to: {preprocessor_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Telco dataset for churn modeling."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help="Path to raw input CSV.",
    )
    parser.add_argument(
        "--output-tree",
        type=Path,
        default=CLEANED_TREE_DATA_PATH,
        help="Path to output CSV for tree-family models.",
    )
    parser.add_argument(
        "--output-linear",
        type=Path,
        default=CLEANED_LINEAR_DATA_PATH,
        help="Path to output CSV for linear-family models.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated alias for --output-tree.",
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
    output_tree = args.output if args.output is not None else args.output_tree
    run_preprocessing(
        raw_data_path=args.input,
        cleaned_tree_data_path=output_tree,
        cleaned_linear_data_path=args.output_linear,
        preprocessor_path=args.preprocessor,
    )


if __name__ == "__main__":
    main()

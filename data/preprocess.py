from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/netflix_large.csv")
CLEANED_DATA_PATH = Path("data/processed/cleaned_data.csv")
PREPROCESSOR_PATH = Path("artifacts/preprocessors/preprocessor.pkl")

TARGET_COLUMN = "churn_status"
DATE_COLUMNS = ["subscription_start_date", "date_watched"] 
LEAKAGE_DROP_COLUMNS = ["subscription_end_date"]
HIGH_CARDINALITY_DROP_COLUMNS = ["user_id", "title"]
TARGET_LABEL_MAPPING = {"Active": 0, "Cancelled": 1}

TEST_SIZE = 0.2
RANDOM_STATE = 42


def _build_one_hot_encoder() -> OneHotEncoder:
	# Keep compatibility with both old and new scikit-learn versions.
	try:
		return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
	except TypeError:
		return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    # Chỉ xử lý các cột ngày an toàn (Start date, etc.)
    for col in DATE_COLUMNS:
        if col not in work_df.columns:
            continue
        dt = pd.to_datetime(work_df[col], errors="coerce")
        work_df[f"{col}_year"] = dt.dt.year
        work_df[f"{col}_month"] = dt.dt.month
        work_df[f"{col}_day"] = dt.dt.day
        work_df[f"{col}_weekday"] = dt.dt.weekday

    # Tính toán thời gian từ lúc bắt đầu đến lúc xem (An toàn)
    if {"subscription_start_date", "date_watched"}.issubset(work_df.columns):
        start_dt = pd.to_datetime(work_df["subscription_start_date"], errors="coerce")
        watched_dt = pd.to_datetime(work_df["date_watched"], errors="coerce")
        work_df["days_from_start_to_watch"] = (watched_dt - start_dt).dt.days

    # Xóa các cột gốc và các cột gây Leakage
    drop_candidates = DATE_COLUMNS + ["subscription_end_date"]
    existing_drop_cols = [col for col in drop_candidates if col in work_df.columns]
    work_df = work_df.drop(columns=existing_drop_cols)

    return work_df


def _build_preprocessor(feature_df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
	numeric_cols = feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
	categorical_cols = feature_df.select_dtypes(exclude=["number", "bool"]).columns.tolist()

	numeric_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="median")),
			("scaler", StandardScaler()),
		]
	)
	categorical_pipeline = Pipeline(
		steps=[
			("imputer", SimpleImputer(strategy="most_frequent")),
			("onehot", _build_one_hot_encoder()),
		]
	)

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_pipeline, numeric_cols),
			("cat", categorical_pipeline, categorical_cols),
		],
		remainder="drop",
		verbose_feature_names_out=False,
	)
	return preprocessor, numeric_cols, categorical_cols


def _encode_target_labels(target_series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
	# Convert string labels to deterministic numeric targets for ML training.
	normalized = target_series.astype(str).str.strip()
	encoded = normalized.map(TARGET_LABEL_MAPPING)

	if encoded.isna().any():
		unknown_values = sorted(normalized[encoded.isna()].unique().tolist())
		raise ValueError(
			f"Unsupported target labels found in '{TARGET_COLUMN}': {unknown_values}. "
			f"Expected labels: {sorted(TARGET_LABEL_MAPPING.keys())}"
		)

	return encoded.astype(int), TARGET_LABEL_MAPPING.copy()


def run_preprocessing(
	raw_data_path: Path,
	cleaned_data_path: Path,
	preprocessor_path: Path,
) -> None:
	# Step 1: Load raw data from DVC-managed folder.
	raw_df = pd.read_csv(raw_data_path)

	# Step 2: Apply deterministic feature cleaning rules.
	work_df = _add_date_features(raw_df)

	# Drop high-cardinality identifiers/text to prevent unstable one-hot dimensions.
	drop_cols = [col for col in HIGH_CARDINALITY_DROP_COLUMNS if col in work_df.columns]
	if drop_cols:
		work_df = work_df.drop(columns=drop_cols)

	# Extract target from feature columns when available.
	target_series = None
	target_mapping: dict[str, int] | None = None
	train_indices = None
	test_indices = None
	if TARGET_COLUMN in work_df.columns:
		target_series, target_mapping = _encode_target_labels(work_df[TARGET_COLUMN].copy())
		feature_df = work_df.drop(columns=[TARGET_COLUMN])
	else:
		feature_df = work_df

	# Step 3: Split first, then fit preprocessor only on train to prevent leakage.
	if target_series is not None:
		x_train, x_test, _, _ = train_test_split(
			feature_df,
			target_series,
			test_size=TEST_SIZE,
			random_state=RANDOM_STATE,
			stratify=target_series,
		)
	else:
		x_train, x_test = train_test_split(
			feature_df,
			test_size=TEST_SIZE,
			random_state=RANDOM_STATE,
		)

	train_indices = x_train.index
	test_indices = x_test.index

	# Step 4: Fit preprocessing pipeline (imputation + encoding + scaling) on train only.
	preprocessor, numeric_cols, categorical_cols = _build_preprocessor(feature_df)
	preprocessor.fit(x_train)
	transformed = preprocessor.transform(feature_df)
	feature_names = preprocessor.get_feature_names_out()
	cleaned_df = pd.DataFrame(transformed, columns=feature_names, index=feature_df.index)

	if target_series is not None:
		cleaned_df[TARGET_COLUMN] = target_series.values

	# Keep split membership for reproducible train/test downstream usage.
	cleaned_df["data_split"] = "train"
	cleaned_df.loc[test_indices, "data_split"] = "test"
	cleaned_df = cleaned_df.reset_index(drop=True)

	# Step 5: Save transformed dataset and fitted preprocessor artifact.
	cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
	preprocessor_path.parent.mkdir(parents=True, exist_ok=True)

	cleaned_df.to_csv(cleaned_data_path, index=False)
	joblib.dump(
		{
			"pipeline": preprocessor,
			"feature_columns": feature_df.columns.tolist(),
			"numeric_columns": numeric_cols,
			"categorical_columns": categorical_cols,
			"dropped_columns": drop_cols,
			"target_column": TARGET_COLUMN if target_series is not None else None,
			"target_mapping": target_mapping,
			"split_config": {
				"test_size": TEST_SIZE,
				"random_state": RANDOM_STATE,
				"stratify": bool(target_series is not None),
			},
			"train_indices": train_indices.tolist() if train_indices is not None else None,
			"test_indices": test_indices.tolist() if test_indices is not None else None,
		},
		preprocessor_path,
	)

	print(f"Raw shape: {raw_df.shape}")
	print(f"Cleaned shape: {cleaned_df.shape}")
	print(f"Saved cleaned data to: {cleaned_data_path}")
	print(f"Saved preprocessor to: {preprocessor_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Preprocess raw Netflix dataset for MLOps pipeline.")
	parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to raw input CSV.")
	parser.add_argument("--output", type=Path, default=CLEANED_DATA_PATH, help="Path to cleaned output CSV.")
	parser.add_argument(
		"--preprocessor",
		type=Path,
		default=PREPROCESSOR_PATH,
		help="Path to save fitted preprocessing object.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	run_preprocessing(args.input, args.output, args.preprocessor)


if __name__ == "__main__":
	main()

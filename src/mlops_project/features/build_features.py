from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.mlops_project.data.validate_data import TARGET_COLUMN


LEAKAGE_COLUMNS: list[str] = []
HIGH_CARDINALITY_DROP_COLUMNS = ["customerID"]
REPLACED_SOURCE_COLUMNS = ["Contract"]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

LINEAR_SCALE_REQUIRED_COLUMNS = {
    "TotalCharges",
    "MonthlyCharges",
    "tenure",
    "service_yes_count",
}


@dataclass
class FeatureBuildReport:
    dropped_columns: list[str]
    leakage_columns_removed: list[str]


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _is_yes(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().str.strip().eq("yes").astype(int)


def _is_binary_yes_no(series: pd.Series) -> bool:
    non_null = series.dropna()
    if non_null.empty:
        return False

    normalized = set(non_null.astype(str).str.strip().str.lower().unique().tolist())
    return normalized.issubset({"yes", "no"})


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    if "Contract" in work_df.columns:
        work_df["is_monthly_contract"] = (
            work_df["Contract"].astype(str).str.lower().str.strip().eq("month-to-month")
        ).astype(int)

        # Replace the original high-cardinality contract text with a compact binary signal.
        work_df = work_df.drop(columns=["Contract"])
    else:
        work_df["is_monthly_contract"] = 0

    if "PaymentMethod" in work_df.columns:
        work_df["is_electronic_check"] = (
            work_df["PaymentMethod"]
            .astype(str)
            .str.lower()
            .str.strip()
            .eq("electronic check")
        ).astype(int)
    else:
        work_df["is_electronic_check"] = 0

    if "OnlineSecurity" in work_df.columns:
        work_df["has_security"] = _is_yes(work_df["OnlineSecurity"])
    else:
        work_df["has_security"] = 0

    if "TechSupport" in work_df.columns:
        work_df["has_techsupport"] = _is_yes(work_df["TechSupport"])
    else:
        work_df["has_techsupport"] = 0

    if "MonthlyCharges" in work_df.columns:
        monthly = pd.to_numeric(work_df["MonthlyCharges"], errors="coerce")
        threshold = float(monthly.quantile(0.75)) if monthly.notna().any() else 0.0
        if np.isnan(threshold):
            threshold = 0.0
        work_df["high_monthly_charge"] = (monthly >= threshold).astype(int)
    else:
        work_df["high_monthly_charge"] = 0

    if "tenure" in work_df.columns:
        tenure = pd.to_numeric(work_df["tenure"], errors="coerce").fillna(0)
        work_df["tenure"] = tenure
        work_df["tenure_group"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, 72],
            labels=["0-12", "13-24", "25-48", "49-72"],
            include_lowest=True,
        )

    service_sum = pd.Series(0, index=work_df.index, dtype="int64")
    for col in SERVICE_COLUMNS:
        if col in work_df.columns:
            service_sum = service_sum.add(_is_yes(work_df[col]), fill_value=0)
    work_df["service_yes_count"] = service_sum.astype(int)

    work_df["high_monthly_charge_x_is_monthly_contract"] = (
        work_df["high_monthly_charge"] * work_df["is_monthly_contract"]
    )

    return work_df


def prepare_feature_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuildReport]:
    work_df = df.copy()

    # Guardrail: labels must be encoded in validate_data to keep target logic centralized.
    if "Churn" in work_df.columns or "churn" in work_df.columns:
        raise ValueError(
            "Raw target column detected. Please run clean_raw_dataframe before "
            "prepare_feature_inputs so target encoding is consistent (Yes->1, No->0)."
        )

    if TARGET_COLUMN in work_df.columns and not pd.api.types.is_numeric_dtype(
        work_df[TARGET_COLUMN]
    ):
        raise ValueError(
            f"Column '{TARGET_COLUMN}' must be numeric. "
            "Run clean_raw_dataframe to encode target labels before feature building."
        )

    leakage_cols = [col for col in LEAKAGE_COLUMNS if col in work_df.columns]
    if leakage_cols:
        work_df = work_df.drop(columns=leakage_cols)

    high_cardinality_cols = [
        col for col in HIGH_CARDINALITY_DROP_COLUMNS if col in work_df.columns
    ]
    if high_cardinality_cols:
        work_df = work_df.drop(columns=high_cardinality_cols)

    columns_before_engineering = set(work_df.columns)
    work_df = _add_engineered_features(work_df)
    replaced_source_cols = [
        col
        for col in REPLACED_SOURCE_COLUMNS
        if col in columns_before_engineering and col not in work_df.columns
    ]

    dropped_cols = leakage_cols + high_cardinality_cols + replaced_source_cols
    report = FeatureBuildReport(
        dropped_columns=dropped_cols,
        leakage_columns_removed=leakage_cols,
    )
    return work_df, report


def build_preprocessor(
    feature_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Backward-compatible default preprocessor for tree-family models."""
    return build_tree_preprocessor(feature_df)


def _split_feature_types(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str], list[str]]:
    model_feature_df = feature_df.drop(columns=[TARGET_COLUMN], errors="ignore")

    numeric_cols = model_feature_df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = model_feature_df.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    # Route strictly Yes/No categorical columns to OrdinalEncoder to keep one compact 0/1 column.
    binary_categorical_cols = [
        col for col in categorical_cols if _is_binary_yes_no(model_feature_df[col])
    ]
    ohe_categorical_cols = [
        col for col in categorical_cols if col not in binary_categorical_cols
    ]

    return (
        model_feature_df,
        numeric_cols,
        categorical_cols,
        binary_categorical_cols,
        ohe_categorical_cols,
    )


def _build_categorical_pipelines() -> tuple[Pipeline, Pipeline]:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", _build_one_hot_encoder()),
        ]
    )

    binary_categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return categorical_pipeline, binary_categorical_pipeline


def build_tree_preprocessor(
    feature_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    (
        _model_feature_df,
        numeric_cols,
        categorical_cols,
        binary_categorical_cols,
        ohe_categorical_cols,
    ) = _split_feature_types(feature_df)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found after feature preparation.")

    # Tree-family models do not need scaling.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline, binary_categorical_pipeline = _build_categorical_pipelines()

    transformers = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if binary_categorical_cols:
        transformers.append(
            ("cat_binary", binary_categorical_pipeline, binary_categorical_cols)
        )
    if ohe_categorical_cols:
        transformers.append(("cat", categorical_pipeline, ohe_categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def build_linear_preprocessor(
    feature_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    (
        _model_feature_df,
        numeric_cols,
        categorical_cols,
        binary_categorical_cols,
        ohe_categorical_cols,
    ) = _split_feature_types(feature_df)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found after feature preparation.")

    # Only selected continuous/count columns are scaled.
    scaled_numeric_cols = [
        col for col in numeric_cols if col in LINEAR_SCALE_REQUIRED_COLUMNS
    ]
    unscaled_numeric_cols = [
        col for col in numeric_cols if col not in LINEAR_SCALE_REQUIRED_COLUMNS
    ]

    scaled_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    unscaled_numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    categorical_pipeline, binary_categorical_pipeline = _build_categorical_pipelines()

    transformers = []
    if scaled_numeric_cols:
        transformers.append(("num_scaled", scaled_numeric_pipeline, scaled_numeric_cols))
    if unscaled_numeric_cols:
        transformers.append(("num", unscaled_numeric_pipeline, unscaled_numeric_cols))
    if binary_categorical_cols:
        transformers.append(
            ("cat_binary", binary_categorical_pipeline, binary_categorical_cols)
        )
    if ohe_categorical_cols:
        transformers.append(("cat", categorical_pipeline, ohe_categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols

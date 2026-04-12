from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LEAKAGE_COLUMNS = ["subscription_end_date"]
HIGH_CARDINALITY_DROP_COLUMNS = ["user_id", "title"]
DATE_COLUMNS = ["subscription_start_date", "date_watched"]
TARGET_COLUMN = "churn_status"


@dataclass
class FeatureBuildReport:
    dropped_columns: list[str]
    leakage_columns_removed: list[str]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric features to learned quantile bounds (winsorization)."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        frame = self._to_frame(X)
        self._columns = frame.columns.tolist()
        self._lower_bounds = frame.quantile(self.lower_quantile)
        self._upper_bounds = frame.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        frame = self._to_frame(X).copy()
        for col in self._columns:
            if col in frame.columns:
                frame[col] = frame[col].clip(
                    lower=self._lower_bounds[col],
                    upper=self._upper_bounds[col],
                )
        if isinstance(X, pd.DataFrame):
            return frame
        return frame.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(self._columns, dtype=object)
        return np.asarray(input_features, dtype=object)

    def _to_frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if hasattr(self, "_columns"):
            return pd.DataFrame(X, columns=self._columns)
        return pd.DataFrame(X)


def _build_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _parse_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce")


def _add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()

    start_dt = None
    watched_dt = None

    if "subscription_start_date" in work_df.columns:
        start_dt = _parse_datetime(work_df["subscription_start_date"])
        work_df["subscription_start_year"] = start_dt.dt.year
        work_df["subscription_start_month"] = start_dt.dt.month
        work_df["subscription_start_weekday"] = start_dt.dt.weekday

    if "date_watched" in work_df.columns:
        watched_dt = _parse_datetime(work_df["date_watched"])
        work_df["watched_year"] = watched_dt.dt.year
        work_df["watched_month"] = watched_dt.dt.month
        work_df["watched_weekday"] = watched_dt.dt.weekday

    if start_dt is not None and watched_dt is not None:
        days_diff = (watched_dt - start_dt).dt.days
        work_df["days_from_start_to_watch"] = days_diff.clip(lower=0)

    existing_date_cols = [col for col in DATE_COLUMNS if col in work_df.columns]
    if existing_date_cols:
        work_df = work_df.drop(columns=existing_date_cols)

    return work_df


def prepare_feature_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureBuildReport]:
    work_df = df.copy()

    leakage_cols = [col for col in LEAKAGE_COLUMNS if col in work_df.columns]
    if leakage_cols:
        work_df = work_df.drop(columns=leakage_cols)

    high_cardinality_cols = [
        col
        for col in HIGH_CARDINALITY_DROP_COLUMNS
        if col in work_df.columns
    ]
    if high_cardinality_cols:
        work_df = work_df.drop(columns=high_cardinality_cols)

    work_df = _add_date_features(work_df)

    dropped_cols = leakage_cols + high_cardinality_cols
    report = FeatureBuildReport(
        dropped_columns=dropped_cols,
        leakage_columns_removed=leakage_cols,
    )
    return work_df, report


def build_preprocessor(
    feature_df: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = feature_df.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_cols = feature_df.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("winsorizer", QuantileClipper(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
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

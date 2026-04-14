from __future__ import annotations

import pandas as pd


TARGET_COLUMN = "churn_status"
RAW_TARGET_COLUMN = "Churn"

TELCO_REQUIRED_FEATURE_COLUMNS = {
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
}

TARGET_COLUMNS = (RAW_TARGET_COLUMN, "churn", TARGET_COLUMN)

TELCO_COLUMN_ALIASES = {
    "customerid": "customerID",
    "gender": "gender",
    "seniorcitizen": "SeniorCitizen",
    "partner": "Partner",
    "dependents": "Dependents",
    "tenure": "tenure",
    "phoneservice": "PhoneService",
    "multiplelines": "MultipleLines",
    "internetservice": "InternetService",
    "onlinesecurity": "OnlineSecurity",
    "onlinebackup": "OnlineBackup",
    "deviceprotection": "DeviceProtection",
    "techsupport": "TechSupport",
    "streamingtv": "StreamingTV",
    "streamingmovies": "StreamingMovies",
    "contract": "Contract",
    "paperlessbilling": "PaperlessBilling",
    "paymentmethod": "PaymentMethod",
    "monthlycharges": "MonthlyCharges",
    "totalcharges": "TotalCharges",
    "churn": "Churn",
    "churn_status": TARGET_COLUMN,
}

TARGET_LABEL_MAPPING = {
    "yes": 1,
    "no": 0,
    "0": 0,
    "1": 1,
}

SANITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, None),
    "MonthlyCharges": (0, None),
    "TotalCharges": (0, None),
}


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]

    for col in list(work_df.columns):
        canonical = TELCO_COLUMN_ALIASES.get(col.lower())
        if canonical is None or canonical == col:
            continue

        if canonical in work_df.columns:
            work_df[canonical] = work_df[canonical].combine_first(work_df[col])
            work_df = work_df.drop(columns=[col])
        else:
            work_df = work_df.rename(columns={col: canonical})

    return work_df


def _normalize_object_values(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    object_cols = work_df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        work_df[col] = work_df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    return work_df


def _coerce_telco_numeric_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work_df = df.copy()
    coercion_report: dict[str, int] = {}

    numeric_candidates = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for col in numeric_candidates:
        if col not in work_df.columns:
            continue

        raw_series = work_df[col]
        numeric_series = pd.to_numeric(raw_series, errors="coerce")
        invalid_casts = (raw_series.notna() & numeric_series.isna()).sum()
        coercion_report[f"{col}_invalid_casts"] = int(invalid_casts)
        work_df[col] = numeric_series

    if "TotalCharges" in work_df.columns:
        filled_totalcharges = int(work_df["TotalCharges"].isna().sum())
        work_df["TotalCharges"] = work_df["TotalCharges"].fillna(0.0)
        coercion_report["TotalCharges_filled_with_zero"] = filled_totalcharges

    if "SeniorCitizen" in work_df.columns:
        work_df["SeniorCitizen"] = work_df["SeniorCitizen"].fillna(0).round().astype(int)

    return work_df, coercion_report


def _encode_target_labels(target_series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    normalized = target_series.astype(str).str.strip().str.lower()
    encoded = normalized.map(TARGET_LABEL_MAPPING)

    if encoded.isna().any():
        unknown_values = sorted(normalized[encoded.isna()].dropna().unique().tolist())
        raise ValueError(
            f"Unsupported target labels found in '{TARGET_COLUMN}': {unknown_values}. "
            "Supported labels include Yes/No or 0/1."
        )

    return encoded.astype(int), {"no": 0, "yes": 1}


def _encode_target_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, dict[str, int] | None]:
    work_df = df.copy()

    source_target = None
    for col in TARGET_COLUMNS:
        if col in work_df.columns:
            source_target = col
            break

    if source_target is None:
        return work_df, None, None

    encoded_target, target_mapping = _encode_target_labels(work_df[source_target])
    work_df[TARGET_COLUMN] = encoded_target

    if source_target != TARGET_COLUMN:
        work_df = work_df.drop(columns=[source_target])

    return work_df, source_target, target_mapping


def _find_missing_telco_columns(df: pd.DataFrame) -> list[str]:
    return sorted(TELCO_REQUIRED_FEATURE_COLUMNS.difference(df.columns))


def _apply_sanity_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    work_df = df.copy()
    clipped_counts: dict[str, int] = {}

    for col, (lower, upper) in SANITY_BOUNDS.items():
        if col not in work_df.columns:
            continue

        numeric = pd.to_numeric(work_df[col], errors="coerce")
        invalid_mask = pd.Series(False, index=work_df.index)
        if lower is not None:
            invalid_mask = invalid_mask | (numeric < lower)
        if upper is not None:
            invalid_mask = invalid_mask | (numeric > upper)

        clipped_counts[col] = int(invalid_mask.fillna(False).sum())
        work_df[col] = numeric.clip(lower=lower, upper=upper)

    if "SeniorCitizen" in work_df.columns:
        work_df["SeniorCitizen"] = work_df["SeniorCitizen"].fillna(0).round().astype(int)

    return work_df, clipped_counts


def clean_raw_dataframe(
    df: pd.DataFrame,
    *,
    strict_schema: bool = False,
    require_target: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work_df = df.copy()
    report: dict[str, object] = {}

    work_df = _standardize_column_names(work_df)
    work_df = _normalize_object_values(work_df)

    missing_telco_columns = _find_missing_telco_columns(work_df)
    report["missing_telco_columns"] = missing_telco_columns
    if strict_schema and missing_telco_columns:
        raise ValueError(
            "Input data is missing required Telco columns: "
            f"{missing_telco_columns}. Available columns: {list(work_df.columns)}"
        )

    before_rows = len(work_df)
    work_df = work_df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before_rows - len(work_df)

    if "customerID" in work_df.columns:
        report["customerID_duplicates"] = int(work_df["customerID"].duplicated().sum())

    work_df, coercion_report = _coerce_telco_numeric_types(work_df)
    report["numeric_coercion"] = coercion_report

    work_df, clipped_counts = _apply_sanity_bounds(work_df)
    report["sanity_clipped_counts"] = clipped_counts

    work_df, source_target, target_mapping = _encode_target_column(work_df)
    if source_target is not None:
        report["target_source_column"] = source_target
        report["target_mapping"] = target_mapping
    elif require_target:
        raise ValueError(
            "Missing target column. Expected one of: "
            f"{list(TARGET_COLUMNS)}"
        )

    remaining_missing = {
        col: int(count)
        for col, count in work_df.isna().sum().items()
        if int(count) > 0
    }
    report["remaining_missing_after_cleaning"] = remaining_missing

    return work_df, report

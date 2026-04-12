from __future__ import annotations

from datetime import datetime

import pandas as pd


TARGET_COLUMN = "churn_status"

TARGET_LABEL_MAPPING = {
    "active": 0,
    "cancelled": 1,
    "canceled": 1,
    "churn": 1,
    "churned": 1,
    "0": 0,
    "1": 1,
}

# Business sanity bounds. Values outside these bounds are clipped.
SANITY_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "age_group": (0, 100),
    "completion_percentage": (0, 100),
    "watch_time_minutes": (0, None),
    "avg_weekly_watch_time": (0, None),
    "session_count": (0, None),
    "days_since_last_watch": (0, None),
    "content_diversity_score": (0, 1),
    "rating": (1, 5),
    "monthly_fee": (0, None),
    "release_year": (1900, float(datetime.now().year + 1)),
}


def _encode_target_labels(target_series: pd.Series) -> tuple[pd.Series, dict[str, int]]:
    normalized = target_series.astype(str).str.strip().str.lower()
    encoded = normalized.map(TARGET_LABEL_MAPPING)

    if encoded.isna().any():
        unknown_values = sorted(normalized[encoded.isna()].dropna().unique().tolist())
        raise ValueError(
            f"Unsupported target labels found in '{TARGET_COLUMN}': {unknown_values}. "
            "Supported labels include Active/Cancelled or 0/1."
        )

    return encoded.astype(int), {"active": 0, "cancelled": 1}


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

    return work_df, clipped_counts


def clean_raw_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    work_df = df.copy()
    report: dict[str, object] = {}

    before_rows = len(work_df)
    work_df = work_df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = before_rows - len(work_df)

    work_df, clipped_counts = _apply_sanity_bounds(work_df)
    report["sanity_clipped_counts"] = clipped_counts

    if TARGET_COLUMN in work_df.columns:
        encoded_target, target_mapping = _encode_target_labels(work_df[TARGET_COLUMN])
        work_df[TARGET_COLUMN] = encoded_target
        report["target_mapping"] = target_mapping

    return work_df, report

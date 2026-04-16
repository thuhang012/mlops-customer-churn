#!/usr/bin/env python
"""Prometheus exporter for drift metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from prometheus_client import Gauge, Info, start_http_server

from src.mlops_project.monitoring.drift_calculations import (
    DRIFT_FRACTION_CRITICAL,
    JS_CRITICAL,
    KS_CRITICAL,
    PSI_CRITICAL,
    assess_retraining_need,
    evaluate_drift,
)


EXPORTER_INFO = Info("drift_exporter", "Drift exporter build information")
EXPORTER_UP = Gauge("drift_exporter_up", "1 when exporter update succeeds")
DRIFT_FEATURES_TOTAL = Gauge("drift_features_total", "Total evaluated features")
DRIFT_FEATURES_ALERT = Gauge("drift_features_alert", "Total drifted features")
DRIFT_FEATURES_ALERT_FRACTION = Gauge(
    "drift_features_alert_fraction",
    "Fraction of evaluated features in alert state",
)
DRIFT_NUMERIC_KS_MAX = Gauge("drift_numeric_ks_max", "Maximum KS statistic over numeric features")
DRIFT_CATEGORICAL_JS_MAX = Gauge(
    "drift_categorical_js_max",
    "Maximum JS divergence over categorical features",
)
DRIFT_NUMERIC_PSI_MAX = Gauge("drift_numeric_psi_max", "Maximum PSI over numeric features")
DRIFT_AVG_SEVERITY = Gauge("drift_avg_severity", "Average severity among drifted features")
DRIFT_RETRAIN_RECOMMENDED = Gauge(
    "drift_retrain_recommended",
    "1 when retraining is recommended, otherwise 0",
)
DRIFT_RETRAIN_CONFIDENCE = Gauge(
    "drift_retrain_confidence",
    "Retraining confidence encoded as low=1, medium=2, high=3",
)
DRIFT_FEATURE_STATE = Gauge(
    "drift_feature_state",
    "Per-feature drift state (1 drift, 0 ok)",
    ["feature", "feature_type"],
)

DRIFT_THRESHOLD_KS = Gauge("drift_threshold_ks", "Critical KS threshold")
DRIFT_THRESHOLD_JS = Gauge("drift_threshold_js", "Critical JS threshold")
DRIFT_THRESHOLD_PSI = Gauge("drift_threshold_psi", "Critical PSI threshold")
DRIFT_THRESHOLD_FEATURE_FRACTION = Gauge(
    "drift_threshold_feature_fraction",
    "Critical drifted feature fraction threshold",
)
DRIFT_EXPORT_LAST_UPDATE_UNIX = Gauge(
    "drift_export_last_update_unix",
    "Unix timestamp of the last successful drift metric refresh",
)
DRIFT_EXPORT_LAST_ROW_COUNT = Gauge(
    "drift_export_last_row_count",
    "Inference row count seen at the last successful drift metric refresh",
)

DQ_MISSING_RATIO = Gauge(
    "dq_missing_ratio",
    "Overall missing value ratio across common reference/current columns",
)
DQ_SCHEMA_VIOLATION_FRACTION = Gauge(
    "dq_schema_violation_fraction",
    "Fraction of required columns missing from current data",
)
DQ_UNSEEN_CATEGORY_RATIO = Gauge(
    "dq_unseen_category_ratio",
    "Average ratio of unseen categorical values versus reference",
)
DQ_OUT_OF_RANGE_RATIO = Gauge(
    "dq_out_of_range_ratio",
    "Average ratio of values outside reference min/max for numeric features",
)
DQ_FRESHNESS_LAG_SECONDS = Gauge(
    "dq_freshness_lag_seconds",
    "Seconds since last inference record timestamp",
)

MODEL_PREDICTION_COUNT = Gauge(
    "model_prediction_count",
    "Count of prediction values observed in current window",
)
MODEL_PREDICTION_MEAN = Gauge(
    "model_prediction_mean",
    "Mean prediction probability in current window",
)
MODEL_PREDICTION_STD = Gauge(
    "model_prediction_std",
    "Std dev of prediction probability in current window",
)
MODEL_PREDICTION_POSITIVE_RATE = Gauge(
    "model_prediction_positive_rate",
    "Fraction of prediction probabilities >= 0.5",
)
MODEL_LOW_CONFIDENCE_FRACTION = Gauge(
    "model_low_confidence_fraction",
    "Fraction of low-confidence predictions in current window",
)

DQ_THRESHOLD_MISSING_RATIO = Gauge(
    "dq_threshold_missing_ratio",
    "Threshold for missing value ratio",
)
DQ_THRESHOLD_SCHEMA_VIOLATION_FRACTION = Gauge(
    "dq_threshold_schema_violation_fraction",
    "Threshold for schema violation fraction",
)
DQ_THRESHOLD_UNSEEN_CATEGORY_RATIO = Gauge(
    "dq_threshold_unseen_category_ratio",
    "Threshold for unseen category ratio",
)
DQ_THRESHOLD_OUT_OF_RANGE_RATIO = Gauge(
    "dq_threshold_out_of_range_ratio",
    "Threshold for out-of-range ratio",
)
DQ_THRESHOLD_FRESHNESS_LAG_SECONDS = Gauge(
    "dq_threshold_freshness_lag_seconds",
    "Threshold for inference data freshness lag in seconds",
)

MODEL_THRESHOLD_LOW_CONFIDENCE_FRACTION = Gauge(
    "model_threshold_low_confidence_fraction",
    "Threshold for low-confidence prediction fraction",
)
MODEL_THRESHOLD_POSITIVE_RATE_MIN = Gauge(
    "model_threshold_positive_rate_min",
    "Minimum acceptable positive prediction rate",
)
MODEL_THRESHOLD_POSITIVE_RATE_MAX = Gauge(
    "model_threshold_positive_rate_max",
    "Maximum acceptable positive prediction rate",
)

MISSING_RATIO_THRESHOLD = 0.10
SCHEMA_VIOLATION_THRESHOLD = 0.0
UNSEEN_CATEGORY_THRESHOLD = 0.05
OUT_OF_RANGE_THRESHOLD = 0.05
FRESHNESS_LAG_THRESHOLD_SECONDS = 1800.0

LOW_CONFIDENCE_PROB_THRESHOLD = 0.60
LOW_CONFIDENCE_FRACTION_THRESHOLD = 0.40
POSITIVE_RATE_MIN_THRESHOLD = 0.05
POSITIVE_RATE_MAX_THRESHOLD = 0.95


def _confidence_to_level(confidence: str) -> float:
    mapping = {"low": 1.0, "medium": 2.0, "high": 3.0}
    return mapping.get(confidence.lower(), 0.0)


def _set_threshold_metrics() -> None:
    DRIFT_THRESHOLD_KS.set(KS_CRITICAL)
    DRIFT_THRESHOLD_JS.set(JS_CRITICAL)
    DRIFT_THRESHOLD_PSI.set(PSI_CRITICAL)
    DRIFT_THRESHOLD_FEATURE_FRACTION.set(DRIFT_FRACTION_CRITICAL)
    DQ_THRESHOLD_MISSING_RATIO.set(MISSING_RATIO_THRESHOLD)
    DQ_THRESHOLD_SCHEMA_VIOLATION_FRACTION.set(SCHEMA_VIOLATION_THRESHOLD)
    DQ_THRESHOLD_UNSEEN_CATEGORY_RATIO.set(UNSEEN_CATEGORY_THRESHOLD)
    DQ_THRESHOLD_OUT_OF_RANGE_RATIO.set(OUT_OF_RANGE_THRESHOLD)
    DQ_THRESHOLD_FRESHNESS_LAG_SECONDS.set(FRESHNESS_LAG_THRESHOLD_SECONDS)
    MODEL_THRESHOLD_LOW_CONFIDENCE_FRACTION.set(LOW_CONFIDENCE_FRACTION_THRESHOLD)
    MODEL_THRESHOLD_POSITIVE_RATE_MIN.set(POSITIVE_RATE_MIN_THRESHOLD)
    MODEL_THRESHOLD_POSITIVE_RATE_MAX.set(POSITIVE_RATE_MAX_THRESHOLD)


def _compute_data_quality_metrics(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict[str, float]:
    common_columns = sorted(set(reference_df.columns) & set(current_df.columns))
    required_columns = sorted(set(reference_df.columns))

    total_cells = len(current_df) * len(common_columns)
    if total_cells > 0:
        missing_ratio = float(current_df[common_columns].isna().sum().sum() / total_cells)
    else:
        missing_ratio = 0.0

    missing_required = [col for col in required_columns if col not in current_df.columns]
    schema_violation_fraction = (
        float(len(missing_required) / len(required_columns)) if required_columns else 0.0
    )

    categorical_columns = [
        col
        for col in common_columns
        if pd.api.types.is_object_dtype(reference_df[col])
        or pd.api.types.is_categorical_dtype(reference_df[col])
        or pd.api.types.is_bool_dtype(reference_df[col])
    ]
    unseen_ratios: list[float] = []
    for column in categorical_columns:
        ref_values = set(reference_df[column].dropna().astype(str).unique())
        cur_series = current_df[column].dropna().astype(str)
        if len(cur_series) == 0:
            continue
        unseen = (~cur_series.isin(ref_values)).sum()
        unseen_ratios.append(float(unseen / len(cur_series)))
    unseen_category_ratio = float(np.mean(unseen_ratios)) if unseen_ratios else 0.0

    numeric_columns = [
        col
        for col in common_columns
        if pd.api.types.is_numeric_dtype(reference_df[col])
        and pd.api.types.is_numeric_dtype(current_df[col])
    ]
    out_of_range_ratios: list[float] = []
    for column in numeric_columns:
        ref_series = pd.to_numeric(reference_df[column], errors="coerce").dropna()
        cur_series = pd.to_numeric(current_df[column], errors="coerce").dropna()
        if ref_series.empty or cur_series.empty:
            continue
        lower_bound = float(ref_series.min())
        upper_bound = float(ref_series.max())
        outside = ((cur_series < lower_bound) | (cur_series > upper_bound)).sum()
        out_of_range_ratios.append(float(outside / len(cur_series)))
    out_of_range_ratio = float(np.mean(out_of_range_ratios)) if out_of_range_ratios else 0.0

    return {
        "missing_ratio": missing_ratio,
        "schema_violation_fraction": schema_violation_fraction,
        "unseen_category_ratio": unseen_category_ratio,
        "out_of_range_ratio": out_of_range_ratio,
    }


def _compute_freshness_lag_seconds(current_path: Path) -> float:
    if not current_path.exists():
        return 0.0

    try:
        # Read only timestamp column to keep heartbeat checks lightweight.
        timestamp_df = pd.read_csv(current_path, usecols=["timestamp"])
    except Exception:
        return 0.0

    if timestamp_df.empty or "timestamp" not in timestamp_df.columns:
        return 0.0

    parsed_ts = pd.to_datetime(timestamp_df["timestamp"], errors="coerce", utc=True)
    latest_ts = parsed_ts.max()
    if pd.isna(latest_ts):
        return 0.0
    return float(max(time.time() - latest_ts.timestamp(), 0.0))


def _load_current_window(current_path: Path, current_window_size: int) -> pd.DataFrame:
    current_df = pd.read_csv(current_path)
    if current_window_size > 0 and len(current_df) > current_window_size:
        return current_df.tail(current_window_size).reset_index(drop=True)
    return current_df


def collect_fast_metrics(current_path: Path) -> int:
    current_rows = _count_data_rows(current_path)
    freshness_lag_seconds = _compute_freshness_lag_seconds(current_path)
    DQ_FRESHNESS_LAG_SECONDS.set(freshness_lag_seconds)
    DRIFT_EXPORT_LAST_ROW_COUNT.set(current_rows)
    EXPORTER_UP.set(1.0)
    return current_rows


def collect_medium_metrics(reference_path: Path, current_path: Path, current_window_size: int) -> None:
    reference_df = pd.read_csv(reference_path)
    current_df = _load_current_window(current_path=current_path, current_window_size=current_window_size)

    data_quality = _compute_data_quality_metrics(reference_df=reference_df, current_df=current_df)
    model_quality = _compute_model_quality_metrics(current_df=current_df)

    DQ_MISSING_RATIO.set(data_quality["missing_ratio"])
    DQ_SCHEMA_VIOLATION_FRACTION.set(data_quality["schema_violation_fraction"])
    DQ_UNSEEN_CATEGORY_RATIO.set(data_quality["unseen_category_ratio"])
    DQ_OUT_OF_RANGE_RATIO.set(data_quality["out_of_range_ratio"])

    MODEL_PREDICTION_COUNT.set(model_quality["prediction_count"])
    MODEL_PREDICTION_MEAN.set(model_quality["prediction_mean"])
    MODEL_PREDICTION_STD.set(model_quality["prediction_std"])
    MODEL_PREDICTION_POSITIVE_RATE.set(model_quality["prediction_positive_rate"])
    MODEL_LOW_CONFIDENCE_FRACTION.set(model_quality["low_confidence_fraction"])

    EXPORTER_UP.set(1.0)


def collect_heavy_drift_metrics(reference_path: Path, current_path: Path, current_window_size: int) -> None:
    reference_df = pd.read_csv(reference_path)
    current_df = _load_current_window(current_path=current_path, current_window_size=current_window_size)

    metrics = evaluate_drift(reference_df, current_df)
    retrain = assess_retraining_need(metrics)

    total_features = len(metrics)
    drifted_metrics = [m for m in metrics if m["alert"]]
    drifted_count = len(drifted_metrics)
    drift_fraction = (drifted_count / total_features) if total_features else 0.0

    numeric_metrics = [m for m in metrics if m["feature_type"] == "numeric"]
    categorical_metrics = [m for m in metrics if m["feature_type"] == "categorical"]

    ks_max = max((m["value"] for m in numeric_metrics), default=0.0)
    js_max = max((m["value"] for m in categorical_metrics), default=0.0)
    psi_max = max((m.get("psi") or 0.0 for m in numeric_metrics), default=0.0)
    avg_severity = float(retrain.get("avg_severity", 0.0))

    DRIFT_FEATURES_TOTAL.set(total_features)
    DRIFT_FEATURES_ALERT.set(drifted_count)
    DRIFT_FEATURES_ALERT_FRACTION.set(drift_fraction)
    DRIFT_NUMERIC_KS_MAX.set(ks_max)
    DRIFT_CATEGORICAL_JS_MAX.set(js_max)
    DRIFT_NUMERIC_PSI_MAX.set(psi_max)
    DRIFT_AVG_SEVERITY.set(avg_severity)
    DRIFT_RETRAIN_RECOMMENDED.set(1.0 if retrain.get("recommended") else 0.0)
    DRIFT_RETRAIN_CONFIDENCE.set(_confidence_to_level(str(retrain.get("confidence", "low"))))

    DRIFT_FEATURE_STATE.clear()
    for metric in metrics:
        DRIFT_FEATURE_STATE.labels(
            feature=str(metric["feature"]),
            feature_type=str(metric["feature_type"]),
        ).set(1.0 if metric["alert"] else 0.0)

    EXPORTER_UP.set(1.0)


def _compute_model_quality_metrics(current_df: pd.DataFrame) -> dict[str, float]:
    if "prediction" not in current_df.columns or current_df.empty:
        return {
            "prediction_count": 0.0,
            "prediction_mean": 0.0,
            "prediction_std": 0.0,
            "prediction_positive_rate": 0.0,
            "low_confidence_fraction": 0.0,
        }

    prediction_series = pd.to_numeric(current_df["prediction"], errors="coerce").dropna()
    if prediction_series.empty:
        return {
            "prediction_count": 0.0,
            "prediction_mean": 0.0,
            "prediction_std": 0.0,
            "prediction_positive_rate": 0.0,
            "low_confidence_fraction": 0.0,
        }

    probs = prediction_series.clip(lower=0.0, upper=1.0)
    confidences = np.maximum(probs, 1.0 - probs)

    return {
        "prediction_count": float(len(probs)),
        "prediction_mean": float(probs.mean()),
        "prediction_std": float(probs.std(ddof=0)),
        "prediction_positive_rate": float((probs >= 0.5).mean()),
        "low_confidence_fraction": float((confidences < LOW_CONFIDENCE_PROB_THRESHOLD).mean()),
    }


def _count_data_rows(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8") as file_handle:
        total_lines = sum(1 for _ in file_handle)
    # Minus header row when file has at least one line.
    return max(total_lines - 1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drift metrics Prometheus exporter.")
    parser.add_argument("--port", type=int, default=9108, help="Exporter listen port")
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval seconds to evaluate whether refresh conditions are met",
    )
    parser.add_argument(
        "--fast-interval",
        type=int,
        default=300,
        help="Refresh interval seconds when enough new rows arrive (default: 5 minutes)",
    )
    parser.add_argument(
        "--slow-interval",
        type=int,
        default=1800,
        help="Refresh interval seconds fallback (default: 30 minutes)",
    )
    parser.add_argument(
        "--force-refresh-interval",
        type=int,
        default=60,
        help="[Deprecated] legacy full-refresh interval; use fast/medium/heavy intervals instead",
    )
    parser.add_argument(
        "--fast-refresh-interval",
        type=int,
        default=60,
        help="Fast refresh interval seconds for lightweight heartbeat/freshness metrics",
    )
    parser.add_argument(
        "--medium-refresh-interval",
        type=int,
        default=300,
        help="Medium refresh interval seconds for data-quality and model-quality metrics",
    )
    parser.add_argument(
        "--min-new-rows",
        type=int,
        default=100,
        help="Minimum number of new rows required to use fast refresh schedule",
    )
    parser.add_argument(
        "--current-window-size",
        type=int,
        default=5000,
        help="Rolling window size (latest rows) from current inference data for hybrid baseline drift checks",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("data/processed/cleaned_data_tree.csv"),
        help="Reference dataset CSV",
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=Path("data/processed/inference_log.csv"),
        help="Current/inference dataset CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    EXPORTER_INFO.info({"name": "drift_metrics_exporter", "version": "1.0.0"})
    _set_threshold_metrics()
    start_http_server(args.port)
    print(f"Drift metrics exporter listening on :{args.port}")

    last_fast_ts = 0.0
    last_medium_ts = 0.0
    last_heavy_ts = 0.0
    last_heavy_rows = 0

    while True:
        now = time.time()
        current_rows = _count_data_rows(args.current)
        new_rows_since_heavy = max(current_rows - last_heavy_rows, 0)

        first_run = last_heavy_ts == 0.0
        fast_heartbeat_due = (now - last_fast_ts) >= max(args.fast_refresh_interval, 15)
        medium_due = (now - last_medium_ts) >= max(args.medium_refresh_interval, 60)
        heavy_slow_due = (now - last_heavy_ts) >= max(args.slow_interval, 60)
        heavy_fast_due = (
            new_rows_since_heavy >= max(args.min_new_rows, 1)
            and (now - last_heavy_ts) >= max(args.fast_interval, 60)
        )

        try:
            if first_run or fast_heartbeat_due:
                latest_rows = collect_fast_metrics(current_path=args.current)
                last_fast_ts = now
                DRIFT_EXPORT_LAST_ROW_COUNT.set(latest_rows)

            if first_run or medium_due:
                collect_medium_metrics(
                    reference_path=args.reference,
                    current_path=args.current,
                    current_window_size=max(args.current_window_size, 0),
                )
                last_medium_ts = now

            if first_run or heavy_fast_due or heavy_slow_due:
                collect_heavy_drift_metrics(
                    reference_path=args.reference,
                    current_path=args.current,
                    current_window_size=max(args.current_window_size, 0),
                )
                last_heavy_ts = now
                last_heavy_rows = current_rows
                DRIFT_EXPORT_LAST_UPDATE_UNIX.set(last_heavy_ts)
                DRIFT_EXPORT_LAST_ROW_COUNT.set(last_heavy_rows)
        except Exception as exc:
            EXPORTER_UP.set(0.0)
            print(f"[drift_exporter] collection error: {exc}")
        time.sleep(max(args.poll_interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())

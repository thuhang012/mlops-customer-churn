#!/usr/bin/env python
"""Prometheus exporter for drift metrics."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

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


def _confidence_to_level(confidence: str) -> float:
    mapping = {"low": 1.0, "medium": 2.0, "high": 3.0}
    return mapping.get(confidence.lower(), 0.0)


def _set_threshold_metrics() -> None:
    DRIFT_THRESHOLD_KS.set(KS_CRITICAL)
    DRIFT_THRESHOLD_JS.set(JS_CRITICAL)
    DRIFT_THRESHOLD_PSI.set(PSI_CRITICAL)
    DRIFT_THRESHOLD_FEATURE_FRACTION.set(DRIFT_FRACTION_CRITICAL)


def collect_and_export(reference_path: Path, current_path: Path, current_window_size: int) -> None:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)
    if current_window_size > 0 and len(current_df) > current_window_size:
        current_df = current_df.tail(current_window_size).reset_index(drop=True)

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
        default=Path("data/processed/cleaned_data_linear.csv"),
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

    last_refresh_ts = 0.0
    last_refresh_rows = 0

    while True:
        now = time.time()
        current_rows = _count_data_rows(args.current)
        new_rows = max(current_rows - last_refresh_rows, 0)

        first_run = last_refresh_ts == 0.0
        slow_due = (now - last_refresh_ts) >= max(args.slow_interval, 60)
        fast_due = (
            new_rows >= max(args.min_new_rows, 1)
            and (now - last_refresh_ts) >= max(args.fast_interval, 60)
        )

        try:
            if first_run or fast_due or slow_due:
                collect_and_export(
                    reference_path=args.reference,
                    current_path=args.current,
                    current_window_size=max(args.current_window_size, 0),
                )
                last_refresh_ts = now
                last_refresh_rows = current_rows
                DRIFT_EXPORT_LAST_UPDATE_UNIX.set(last_refresh_ts)
                DRIFT_EXPORT_LAST_ROW_COUNT.set(last_refresh_rows)
        except Exception as exc:
            EXPORTER_UP.set(0.0)
            print(f"[drift_exporter] collection error: {exc}")
        time.sleep(max(args.poll_interval, 5))


if __name__ == "__main__":
    raise SystemExit(main())

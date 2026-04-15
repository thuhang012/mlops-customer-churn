#!/usr/bin/env python
"""Monitoring checks for drift detection and model degradation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DEFAULT_REFERENCE_PATH = Path("data/raw/netflix_large.csv")
DEFAULT_PRODUCTION_LOG_PATH = Path("monitoring/inference/inference_log.csv")
DEFAULT_BASELINE_METRICS_PATH = Path("artifacts/baseline/metrics.json")
DEFAULT_CURRENT_METRICS_PATH = Path("artifacts/metrics/metrics.json")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if expected.empty or actual.empty:
        return 0.0

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) <= 1:
        return 0.0

    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_percents = np.where(expected_percents == 0, 1e-4, expected_percents)
    actual_percents = np.where(actual_percents == 0, 1e-4, actual_percents)
    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return float(np.sum(psi_values))


def _compute_feature_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    min_drifted_share: float,
    ks_alpha: float,
    psi_threshold: float,
) -> tuple[bool, dict]:
    reference_numeric = reference_df.select_dtypes(include=[np.number])
    production_numeric = production_df.select_dtypes(include=[np.number])
    common_columns = sorted(set(reference_numeric.columns) & set(production_numeric.columns))

    if not common_columns:
        return False, {
            "status": "SKIPPED",
            "reason": "No common numeric columns between reference and production data",
            "drift_share": 0.0,
            "drifted_columns": 0,
            "total_columns": 0,
            "column_details": {},
        }

    drifted_columns = 0
    details: dict[str, dict] = {}
    for col in common_columns:
        ref_col = reference_numeric[col].dropna()
        prod_col = production_numeric[col].dropna()
        if ref_col.empty or prod_col.empty:
            continue

        ks_stat, p_value = ks_2samp(ref_col, prod_col)
        psi_score = _calculate_psi(ref_col, prod_col)
        drift = bool((p_value < ks_alpha) or (psi_score >= psi_threshold))
        details[col] = {
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "psi_score": float(psi_score),
            "drift_detected": drift,
        }
        if drift:
            drifted_columns += 1

    total_columns = len(details)
    drift_share = (drifted_columns / total_columns) if total_columns else 0.0
    drift_detected = drift_share >= min_drifted_share and total_columns > 0
    status = "ALERT" if drift_detected else "OK"
    return drift_detected, {
        "status": status,
        "drift_share": float(drift_share),
        "drifted_columns": drifted_columns,
        "total_columns": total_columns,
        "ks_alpha": float(ks_alpha),
        "psi_threshold": float(psi_threshold),
        "column_details": details,
    }


def _compute_performance_degradation(
    baseline_metrics: dict,
    current_metrics: dict,
    max_allowed_degradation: float,
) -> tuple[bool, dict]:
    monitored_keys = ["f1_score", "roc_auc", "accuracy", "precision", "recall"]
    degradation_details: dict[str, dict] = {}
    degraded_metrics = 0

    for key in monitored_keys:
        baseline = baseline_metrics.get(key)
        current = current_metrics.get(key)
        if baseline is None or current is None:
            continue
        baseline = float(baseline)
        current = float(current)
        if baseline <= 0:
            continue

        allowed_floor = baseline * (1 - max_allowed_degradation)
        degradation_pct = (baseline - current) / baseline
        degraded = current < allowed_floor
        if degraded:
            degraded_metrics += 1

        degradation_details[key] = {
            "baseline": baseline,
            "current": current,
            "allowed_floor": float(allowed_floor),
            "degradation_pct": float(degradation_pct),
            "degraded": degraded,
        }

    degradation_detected = degraded_metrics > 0
    status = "ALERT" if degradation_detected else "OK"
    return degradation_detected, {
        "status": status,
        "max_allowed_degradation": float(max_allowed_degradation),
        "degraded_metrics": degraded_metrics,
        "total_checked_metrics": len(degradation_details),
        "metrics": degradation_details,
    }


def run_monitoring_checks(
    reference_path: Path = DEFAULT_REFERENCE_PATH,
    production_log_path: Path = DEFAULT_PRODUCTION_LOG_PATH,
    baseline_metrics_path: Path = DEFAULT_BASELINE_METRICS_PATH,
    current_metrics_path: Path = DEFAULT_CURRENT_METRICS_PATH,
    min_drifted_share: float = 0.30,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.20,
    max_allowed_degradation: float = 0.05,
) -> dict:
    """Run monitoring checks with real data-driven logic."""
    checks: dict[str, dict] = {}
    drift_detected = False
    degradation_detected = False
    issues: list[str] = []

    if reference_path.exists() and production_log_path.exists():
        reference_df = pd.read_csv(reference_path)
        production_df = pd.read_csv(production_log_path)
        drift_detected, drift_report = _compute_feature_drift(
            reference_df=reference_df,
            production_df=production_df,
            min_drifted_share=min_drifted_share,
            ks_alpha=ks_alpha,
            psi_threshold=psi_threshold,
        )
        checks["feature_drift"] = drift_report
        if production_df.empty:
            issues.append(f"Production log is empty: {production_log_path}")
    else:
        missing = [str(p) for p in [reference_path, production_log_path] if not p.exists()]
        checks["feature_drift"] = {"status": "SKIPPED", "reason": f"Missing files: {missing}"}
        issues.append("Feature drift check skipped due to missing data files")

    if baseline_metrics_path.exists() and current_metrics_path.exists():
        baseline_metrics = _load_json(baseline_metrics_path)
        current_metrics = _load_json(current_metrics_path)
        degradation_detected, degradation_report = _compute_performance_degradation(
            baseline_metrics=baseline_metrics,
            current_metrics=current_metrics,
            max_allowed_degradation=max_allowed_degradation,
        )
        checks["performance_degradation"] = degradation_report
    else:
        missing = [str(p) for p in [baseline_metrics_path, current_metrics_path] if not p.exists()]
        checks["performance_degradation"] = {"status": "SKIPPED", "reason": f"Missing files: {missing}"}
        issues.append("Performance degradation check skipped due to missing metrics files")

    checks["data_quality"] = {
        "status": "OK" if not issues else "WARN",
        "issues": issues,
    }

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "drift_detected": bool(drift_detected),
        "degradation_detected": bool(degradation_detected),
        "checks": checks,
    }


def save_results(results: dict) -> None:
    report_path = Path("reports/monitoring")
    report_path.mkdir(parents=True, exist_ok=True)
    with (report_path / "latest_check.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"Monitoring results saved to {report_path}/latest_check.json")


def output_github_variables(results: dict) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as file:
            file.write(f"drift_detected={str(results['drift_detected']).lower()}\n")
            file.write(f"degradation_detected={str(results['degradation_detected']).lower()}\n")
    print(f"drift_detected={results['drift_detected']}")
    print(f"degradation_detected={results['degradation_detected']}")


def main() -> int:
    print("Running monitoring checks...")
    try:
        results = run_monitoring_checks()
        print(json.dumps(results, indent=2))
        save_results(results)
        output_github_variables(results)
        print("Monitoring checks completed successfully")
        return 0
    except Exception as exc:
        print(f"Error during monitoring: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

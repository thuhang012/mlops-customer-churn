#!/usr/bin/env python
"""Monitoring checks for drift detection and model degradation."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


DEFAULT_REFERENCE_PATH = Path("data/processed/drift_reference_clean.csv")
DEFAULT_PRODUCTION_LOG_PATH = Path("data/processed/inference_log.csv")
DEFAULT_BASELINE_METRICS_PATH = Path("artifacts/baseline/metrics.json")
DEFAULT_CURRENT_METRICS_PATH = Path("artifacts/metrics/metrics.json")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


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
    max_allowed_degradation: float = 0.05,
    fail_on_missing_inputs: bool = False,
) -> dict:
    """Run monitoring checks focused on degradation and input availability.

    Drift alerting is handled by Prometheus + drift metrics exporter.
    """
    checks: dict[str, dict] = {}
    drift_detected = False
    degradation_detected = False
    issues: list[str] = []
    missing_inputs: list[str] = []

    if reference_path.exists() and production_log_path.exists():
        production_df = pd.read_csv(production_log_path)
        checks["feature_drift"] = {
            "status": "SKIPPED",
            "reason": "Drift decisioning is owned by Prometheus alert rules (drift_metrics_exporter)",
        }
        if production_df.empty:
            issues.append(f"Production log is empty: {production_log_path}")
    else:
        missing = [str(p) for p in [reference_path, production_log_path] if not p.exists()]
        checks["feature_drift"] = {"status": "SKIPPED", "reason": f"Missing files: {missing}"}
        missing_inputs.extend(missing)
        issues.append("Feature drift data availability check failed")

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
        missing_inputs.extend(missing)
        issues.append("Performance degradation check skipped due to missing metrics files")

    checks["data_quality"] = {
        "status": "OK" if not issues else "WARN",
        "issues": issues,
    }

    if fail_on_missing_inputs and missing_inputs:
        unique_missing_inputs = sorted(set(missing_inputs))
        raise FileNotFoundError(
            "Missing required monitoring inputs: " + ", ".join(unique_missing_inputs)
        )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
        fail_on_missing_inputs = _env_bool("MONITORING_FAIL_ON_MISSING_INPUTS", True)
        print(f"Strict input checks enabled: {fail_on_missing_inputs}")
        results = run_monitoring_checks(
            fail_on_missing_inputs=fail_on_missing_inputs,
        )
        print(json.dumps(results, indent=2))
        save_results(results)
        output_github_variables(results)
        print("Monitoring checks completed successfully")
        return 0
    except Exception as exc:
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "drift_detected": False,
            "degradation_detected": False,
            "error": str(exc),
            "checks": {
                "feature_drift": {"status": "ERROR", "reason": str(exc)},
                "performance_degradation": {"status": "ERROR", "reason": str(exc)},
                "data_quality": {"status": "ERROR", "issues": [str(exc)]},
            },
        }
        print(f"Error during monitoring: {exc}")
        print(json.dumps(results, indent=2))
        save_results(results)
        output_github_variables(results)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

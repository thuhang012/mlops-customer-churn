import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.monitoring.checks import run_monitoring_checks


pytestmark = pytest.mark.fast


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_monitoring_flags_drift_and_degradation_when_thresholds_are_exceeded(tmp_path: Path):
    reference_path = tmp_path / "reference.csv"
    production_path = tmp_path / "production.csv"
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"

    pd.DataFrame(
        {
            "monthly_fee": [10, 11, 12, 13, 14, 15],
            "watch_time_minutes": [20, 21, 22, 23, 24, 25],
        }
    ).to_csv(reference_path, index=False)

    pd.DataFrame(
        {
            "monthly_fee": [110, 120, 130, 140, 150, 160],
            "watch_time_minutes": [220, 230, 240, 250, 260, 270],
        }
    ).to_csv(production_path, index=False)

    _write_json(
        baseline_path,
        {"f1_score": 0.8, "roc_auc": 0.8, "accuracy": 0.82},
    )
    _write_json(
        current_path,
        {"f1_score": 0.6, "roc_auc": 0.65, "accuracy": 0.7},
    )

    results = run_monitoring_checks(
        reference_path=reference_path,
        production_log_path=production_path,
        baseline_metrics_path=baseline_path,
        current_metrics_path=current_path,
        min_drifted_share=0.2,
        ks_alpha=0.2,
        psi_threshold=0.1,
        max_allowed_degradation=0.05,
    )

    assert results["drift_detected"] is True
    assert results["degradation_detected"] is True
    assert results["checks"]["feature_drift"]["status"] == "ALERT"
    assert results["checks"]["performance_degradation"]["status"] == "ALERT"


def test_monitoring_skips_checks_without_errors_when_input_files_are_missing(tmp_path: Path):
    reference_path = tmp_path / "missing_reference.csv"
    production_path = tmp_path / "missing_production.csv"
    baseline_path = tmp_path / "missing_baseline.json"
    current_path = tmp_path / "missing_current.json"

    results = run_monitoring_checks(
        reference_path=reference_path,
        production_log_path=production_path,
        baseline_metrics_path=baseline_path,
        current_metrics_path=current_path,
    )

    assert results["drift_detected"] is False
    assert results["degradation_detected"] is False
    assert results["checks"]["feature_drift"]["status"] == "SKIPPED"
    assert results["checks"]["performance_degradation"]["status"] == "SKIPPED"
    assert results["checks"]["data_quality"]["status"] == "WARN"

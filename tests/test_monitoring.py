import json
from pathlib import Path

import pandas as pd
import pytest

from src.mlops_project.monitoring.drift_calculations import evaluate_drift
from src.mlops_project.monitoring.drift_metrics_exporter import _compute_data_quality_metrics
from src.mlops_project.monitoring.checks import run_monitoring_checks


pytestmark = pytest.mark.fast


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_telco_reference_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customerID": ["0001", "0002", "0003", "0004"],
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "No", "Yes", "Yes"],
            "tenure": [1, 12, 24, 36],
            "PhoneService": ["Yes", "Yes", "No", "Yes"],
            "MultipleLines": ["No", "Yes", "No phone service", "Yes"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30],
            "TotalCharges": [29.85, 1889.50, 108.15, 1840.75],
            "churn_status": [1, 0, 0, 1],
        }
    )


def test_monitoring_checks_only_performance_degradation_when_metrics_drop(tmp_path: Path):
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
        max_allowed_degradation=0.05,
    )

    assert results["drift_detected"] is False
    assert results["degradation_detected"] is True
    assert results["checks"]["feature_drift"]["status"] == "SKIPPED"
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


def test_monitoring_strict_mode_fails_when_input_files_are_missing(tmp_path: Path):
    reference_path = tmp_path / "missing_reference.csv"
    production_path = tmp_path / "missing_production.csv"
    baseline_path = tmp_path / "missing_baseline.json"
    current_path = tmp_path / "missing_current.json"

    with pytest.raises(FileNotFoundError):
        run_monitoring_checks(
            reference_path=reference_path,
            production_log_path=production_path,
            baseline_metrics_path=baseline_path,
            current_metrics_path=current_path,
            fail_on_missing_inputs=True,
        )


def test_evaluate_drift_excludes_identifier_columns() -> None:
    reference_df = _build_telco_reference_df()
    current_df = reference_df.copy()
    current_df["customerID"] = ["A", "B", "C", "D"]

    metrics = evaluate_drift(reference_df, current_df)

    assert all(item["feature"] != "customerID" for item in metrics)


def test_seniorcitizen_is_scored_with_js_not_ks_psi() -> None:
    reference_df = _build_telco_reference_df()
    current_df = reference_df.copy()
    current_df["SeniorCitizen"] = [1, 1, 1, 1]

    metrics = evaluate_drift(reference_df, current_df)
    seniorcitizen_metric = next(item for item in metrics if item["feature"] == "SeniorCitizen")

    assert seniorcitizen_metric["feature_type"] == "categorical"
    assert seniorcitizen_metric["js"] is not None
    assert seniorcitizen_metric["ks"] is None
    assert seniorcitizen_metric["psi"] is None


def test_schema_violation_ignores_optional_target_column() -> None:
    reference_df = _build_telco_reference_df()
    current_df = reference_df.drop(columns=["churn_status", "customerID"])

    quality = _compute_data_quality_metrics(reference_df=reference_df, current_df=current_df)

    assert quality["schema_violation_fraction"] == 0.0

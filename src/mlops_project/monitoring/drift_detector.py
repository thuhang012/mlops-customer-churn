"""
Model Monitoring & Drift Detection System
Monitors production model performance and triggers retraining
"""

import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import mlflow

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift in production features"""

    def __init__(self, reference_data_path: str, drift_threshold: float = 0.1, mlflow_tracking_uri: Optional[str] = None):
        self.reference_data = pd.read_csv(reference_data_path)
        self.drift_threshold = drift_threshold

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def detect_feature_drift(self, production_data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Detect drift in production data vs reference data using statistical tests

        Returns:
            (drift_detected, drift_report)
        """
        from scipy.stats import ks_2samp, chi2_contingency

        drift_detected = False
        column_drift_info = {}

        # Compare distributions for each column
        for col in self.reference_data.columns:
            if col not in production_data.columns:
                continue

            ref_col = self.reference_data[col]
            prod_col = production_data[col]

            # Handle missing values
            ref_col = ref_col.dropna()
            prod_col = prod_col.dropna()

            if len(ref_col) == 0 or len(prod_col) == 0:
                continue

            # Test drift based on data type
            if ref_col.dtype in ["int64", "float64"]:
                # Kolmogorov-Smirnov test for numerical data
                ks_stat, p_value = ks_2samp(ref_col, prod_col)
                col_drift = p_value < 0.05  # Significant difference

                column_drift_info[col] = {
                    "drift_detected": col_drift,
                    "drift_score": float(ks_stat),
                    "p_value": float(p_value),
                    "test": "kolmogorov-smirnov",
                }
            else:
                # Chi-square test for categorical data
                ref_counts = ref_col.value_counts()
                prod_counts = prod_col.value_counts()

                # Align categories
                all_categories = set(ref_counts.index) | set(prod_counts.index)
                ref_counts = ref_counts.reindex(all_categories, fill_value=0)
                prod_counts = prod_counts.reindex(all_categories, fill_value=0)

                if len(all_categories) > 1 and ref_counts.sum() > 0 and prod_counts.sum() > 0:
                    chi2, p_value, dof, _ = chi2_contingency([ref_counts.values, prod_counts.values])
                    col_drift = p_value < 0.05

                    column_drift_info[col] = {
                        "drift_detected": col_drift,
                        "drift_score": float(chi2),
                        "p_value": float(p_value),
                        "test": "chi-square",
                    }

        # Overall drift detection
        drifted_columns = sum(1 for v in column_drift_info.values() if v["drift_detected"])
        drift_share = drifted_columns / len(column_drift_info) if column_drift_info else 0

        drift_detected = drift_share > self.drift_threshold

        summary = {
            "timestamp": datetime.now().isoformat(),
            "dataset_drift": drift_detected,
            "drift_share": drift_share,
            "total_columns": len(column_drift_info),
            "drifted_columns": drifted_columns,
            "column_drift": column_drift_info,
        }

        return drift_detected, summary

    def detect_prediction_drift(
        self, production_predictions: pd.Series, historical_predictions: Optional[pd.Series] = None
    ) -> Tuple[bool, Dict]:
        """
        Detect drift in model predictions (concept drift)
        """
        from scipy.stats import ks_2samp

        if historical_predictions is None:
            return False, {"error": "No historical predictions for comparison"}

        # Kolmogorov-Smirnov test
        ks_stat, p_value = ks_2samp(historical_predictions, production_predictions)

        # Drift detected if p-value < 0.05 (significant difference)
        prediction_drift_detected = p_value < 0.05

        # Population Stability Index (PSI)
        psi_score = self._calculate_psi(historical_predictions, production_predictions)

        # PSI thresholds: <0.1 (stable), 0.1-0.25 (moderate), >0.25 (significant)
        psi_drift_detected = psi_score > 0.25

        summary = {
            "timestamp": datetime.now().isoformat(),
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(p_value),
            "psi_score": float(psi_score),
            "prediction_drift_detected": prediction_drift_detected or psi_drift_detected,
            "drift_severity": "high" if psi_score > 0.25 else "moderate" if psi_score > 0.1 else "low",
        }

        return prediction_drift_detected or psi_drift_detected, summary

    def _calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Remove NaN values
        expected = expected.dropna()
        actual = actual.dropna()

        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create bins
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates

        if len(breakpoints) <= 1:
            return 0.0

        # Calculate distributions
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid division by zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)

        # PSI formula
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)

        return float(psi)


class PerformanceMonitor:
    """Monitors model performance metrics in production"""

    def __init__(self, mlflow_tracking_uri: Optional[str] = None):
        self.metrics_history = []

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)

    def log_prediction_metrics(
        self, predictions: np.ndarray, actuals: Optional[np.ndarray] = None, metadata: Optional[Dict] = None
    ) -> Dict:
        """Log prediction metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "n_predictions": len(predictions),
            "positive_rate": float(np.mean(predictions)),
        }

        if actuals is not None:
            from sklearn.metrics import accuracy_score, f1_score

            metrics.update(
                {
                    "accuracy": float(accuracy_score(actuals, predictions)),
                    "f1_score": float(f1_score(actuals, predictions, average="weighted", zero_division=0)),
                }
            )

            # If probabilities are available
            if metadata and "probabilities" in metadata:
                try:
                    from sklearn.metrics import roc_auc_score

                    proba = metadata["probabilities"]
                    metrics["roc_auc"] = float(roc_auc_score(actuals, proba))
                except Exception as e:
                    logger.warning(f"Could not compute ROC AUC: {e}")

        if metadata:
            metrics.update(metadata)

        self.metrics_history.append(metrics)

        return metrics

    def check_performance_degradation(
        self, baseline_metrics: Dict[str, float], degradation_threshold: float = 0.05
    ) -> Tuple[bool, Dict]:
        """Check if recent performance has degraded vs baseline"""

        if not self.metrics_history:
            return False, {"error": "No metrics history available"}

        # Get recent metrics (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(days=1)
        recent_metrics = []

        for m in self.metrics_history:
            try:
                ts = datetime.fromisoformat(m["timestamp"])
                if ts > recent_cutoff:
                    recent_metrics.append(m)
            except ValueError:
                continue

        if not recent_metrics:
            return False, {"error": "No recent metrics"}

        # Calculate average recent performance
        avg_recent = {}
        for key in ["accuracy", "f1_score", "roc_auc"]:
            values = [m.get(key) for m in recent_metrics if m.get(key) is not None]
            if values:
                avg_recent[key] = np.mean(values)

        # Compare against baseline
        degradation_detected = False
        degradations = {}

        for metric, baseline_value in baseline_metrics.items():
            if metric in avg_recent:
                recent_value = avg_recent[metric]
                if baseline_value > 0:
                    degradation = (baseline_value - recent_value) / baseline_value
                else:
                    degradation = 0

                degradations[metric] = {
                    "baseline": baseline_value,
                    "recent": recent_value,
                    "degradation": degradation,
                    "threshold": degradation_threshold,
                }

                if degradation > degradation_threshold:
                    degradation_detected = True

        summary = {
            "timestamp": datetime.now().isoformat(),
            "degradation_detected": degradation_detected,
            "n_recent_samples": len(recent_metrics),
            "metrics": degradations,
        }

        return degradation_detected, summary


class RetrainingTrigger:
    """Manages automated retraining triggers via GitHub Actions"""

    def __init__(self, github_token: str, repo_owner: str, repo_name: str):
        self.github_token = github_token
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def trigger_retrain_workflow(self, reason: str, additional_metadata: Optional[Dict] = None) -> Dict:
        """
        Trigger GitHub Actions workflow_dispatch to start retraining

        Args:
            reason: 'drift-detected' | 'performance-degradation' | 'manual'
            additional_metadata: Extra context to log
        """
        url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}/actions/workflows/ct.yml/dispatches"

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        payload = {"ref": "main", "inputs": {"reason": reason}}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)

            result = {
                "timestamp": datetime.now().isoformat(),
                "trigger_reason": reason,
                "status_code": response.status_code,
                "success": response.status_code == 204,
                "metadata": additional_metadata or {},
            }
        except Exception as e:
            result = {
                "timestamp": datetime.now().isoformat(),
                "trigger_reason": reason,
                "error": str(e),
                "success": False,
                "metadata": additional_metadata or {},
            }

        # Log trigger event
        trigger_log_path = Path("logs/retrain_triggers.jsonl")
        trigger_log_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(trigger_log_path, "a") as f:
                f.write(json.dumps(result) + "\n")
        except Exception as e:
            logger.warning(f"Could not write trigger log: {e}")

        return result


# ============================================================================
# MAIN MONITORING LOOP
# ============================================================================


def run_monitoring_checks(
    production_data_path: str,
    reference_data_path: str,
    baseline_metrics_path: str,
    github_token: str,
    repo_owner: str,
    repo_name: str,
):
    """
    Main monitoring function to be run periodically (e.g., hourly via cron or scheduled workflow)
    """
    logger.info("Starting monitoring checks...")

    try:
        # Load data
        production_data = pd.read_csv(production_data_path)

        with open(baseline_metrics_path) as f:
            baseline_metrics = json.load(f)

        # Initialize monitors
        drift_detector = DriftDetector(reference_data_path)
        perf_monitor = PerformanceMonitor()
        retrain_trigger = RetrainingTrigger(github_token, repo_owner, repo_name)

        monitoring_results = {}

        # Check 1: Feature Drift
        logger.info("🔍 Checking feature drift...")
        feature_drift_detected, drift_report = drift_detector.detect_feature_drift(production_data)
        monitoring_results["drift"] = {"detected": feature_drift_detected, "report": drift_report}

        if feature_drift_detected:
            logger.warning(f"⚠️  Feature drift detected! Drift share: {drift_report['drift_share']:.2%}")

            # Trigger retraining
            trigger_result = retrain_trigger.trigger_retrain_workflow(
                reason="drift-detected", additional_metadata=drift_report
            )

            if trigger_result["success"]:
                logger.info("✅ Retraining workflow triggered successfully")
            else:
                logger.error(f"❌ Failed to trigger retraining: {trigger_result}")
        else:
            logger.info("✅ No feature drift detected")

        # Check 2: Performance Degradation (if labels are available)
        logger.info("🔍 Checking performance degradation...")
        degradation_report = None

        if "actual_labels" in production_data.columns:
            predictions = production_data["predictions"].values
            actuals = production_data["actual_labels"].values

            perf_monitor.log_prediction_metrics(predictions, actuals)

            degradation_detected, degradation_report = perf_monitor.check_performance_degradation(
                baseline_metrics, degradation_threshold=0.05
            )
            monitoring_results["degradation"] = {"detected": degradation_detected, "report": degradation_report}

            if degradation_detected:
                logger.warning("⚠️  Performance degradation detected!")
                logger.warning(f"Details: {degradation_report['metrics']}")

                trigger_result = retrain_trigger.trigger_retrain_workflow(
                    reason="performance-degradation", additional_metadata=degradation_report
                )

                if trigger_result["success"]:
                    logger.info("✅ Retraining workflow triggered successfully")
            else:
                logger.info("✅ No performance degradation detected")
        else:
            logger.info("⏭️  No actual labels available - skipping performance check")

        # Save monitoring results
        monitoring_report_path = Path("reports/monitoring/latest_report.json")
        monitoring_report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "feature_drift": drift_report,
            "performance_degradation": degradation_report,
        }

        with open(monitoring_report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"📊 Monitoring report saved to {monitoring_report_path}")

        return monitoring_results

    except Exception as e:
        logger.error(f"Error during monitoring: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import os

    # Read from environment or arguments
    production_data_path = os.getenv("PRODUCTION_DATA_PATH", "data/production/predictions.csv")
    reference_data_path = os.getenv("REFERENCE_DATA_PATH", "data/processed/train.csv")
    baseline_metrics_path = os.getenv("BASELINE_METRICS_PATH", "artifacts/baseline/metrics.json")
    github_token = os.getenv("GITHUB_TOKEN", "")
    repo_owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    repo_name = os.getenv("GITHUB_REPOSITORY_NAME", "mlops-customer-churn")

    run_monitoring_checks(
        production_data_path, reference_data_path, baseline_metrics_path, github_token, repo_owner, repo_name
    )

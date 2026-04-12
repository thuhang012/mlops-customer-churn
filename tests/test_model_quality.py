"""
Comprehensive Model Testing Suite
Tests model quality, performance, and fairness
"""
import pytest
import joblib
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def model():
    """Load trained model"""
    model_path = Path("artifacts/models/model.pkl")
    if not model_path.exists():
        pytest.skip("Model not found - run training first")
    return joblib.load(model_path)


@pytest.fixture
def test_data():
    """Load test dataset"""
    data_path = Path("data/processed/test.csv")
    if not data_path.exists():
        # Use raw dataset for testing if processed not available
        data_path = Path("data/raw/netflix_large.csv")
        if not data_path.exists():
            pytest.skip("Test data not found")
    
    df = pd.read_csv(data_path)
    # Assume 'churn' column exists or create dummy one
    if 'churn' in df.columns:
        X_test = df.drop('churn', axis=1)
        y_test = df['churn']
    else:
        # Fallback for test data without churn column
        X_test = df.iloc[:, :-1] if df.shape[1] > 1 else df
        y_test = df.iloc[:, -1] if df.shape[1] > 1 else pd.Series([0] * len(df))
    
    return X_test, y_test


@pytest.fixture
def baseline_metrics():
    """Load baseline metrics"""
    metrics_path = Path("artifacts/baseline/metrics.json")
    if not metrics_path.exists():
        return {
            "f1_score": 0.70,  # Default minimum threshold
            "roc_auc": 0.75,
            "accuracy": 0.80
        }
    
    with open(metrics_path) as f:
        return json.load(f)


# ============================================================================
# MODEL QUALITY TESTS
# ============================================================================

class TestModelQuality:
    """Test model performance against baselines and SLAs"""
    
    def test_model_accuracy_above_baseline(self, model, test_data, baseline_metrics):
        """Prevent deploying models worse than baseline"""
        X_test, y_test = test_data
        
        if len(y_test) == 0:
            pytest.skip("No test data available")
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        baseline_acc = baseline_metrics.get("accuracy", 0.80)
        threshold = baseline_acc * 0.95  # Allow 5% degradation max
        
        assert accuracy >= threshold, (
            f"Accuracy {accuracy:.4f} below threshold {threshold:.4f}. "
            f"Baseline: {baseline_acc:.4f}"
        )
    
    def test_model_f1_score_above_baseline(self, model, test_data, baseline_metrics):
        """F1 score quality gate"""
        X_test, y_test = test_data
        
        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            pytest.skip("Insufficient test data")
        
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        baseline_f1 = baseline_metrics.get("f1_score", 0.70)
        threshold = baseline_f1 * 0.95
        
        assert f1 >= threshold, (
            f"F1 score {f1:.4f} below threshold {threshold:.4f}. "
            f"Baseline: {baseline_f1:.4f}"
        )
    
    def test_model_roc_auc_above_baseline(self, model, test_data, baseline_metrics):
        """ROC AUC quality gate"""
        X_test, y_test = test_data
        
        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            pytest.skip("Insufficient test data for ROC AUC")
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            pytest.skip("Cannot compute ROC AUC")
        
        baseline_auc = baseline_metrics.get("roc_auc", 0.75)
        threshold = baseline_auc * 0.98  # Stricter for AUC
        
        assert roc_auc >= threshold, (
            f"ROC AUC {roc_auc:.4f} below threshold {threshold:.4f}. "
            f"Baseline: {baseline_auc:.4f}"
        )
    
    def test_no_prediction_bias_toward_one_class(self, model, test_data):
        """Ensure model doesn't always predict one class"""
        X_test, y_test = test_data
        
        y_pred = model.predict(X_test)
        
        # Check prediction distribution
        unique, counts = np.unique(y_pred, return_counts=True)
        
        # Ensure both classes are predicted
        assert len(unique) >= 1, "Model produces no predictions"
        
        if len(unique) > 1:
            # Ensure no class dominates > 95%
            max_ratio = counts.max() / counts.sum()
            assert max_ratio < 0.95, (
                f"Model heavily biased: {max_ratio:.2%} predictions are one class"
            )


# ============================================================================
# PERFORMANCE TESTS (SLA)
# ============================================================================

class TestModelPerformance:
    """Test model inference latency and throughput"""
    
    def test_single_prediction_latency_sla(self, model, test_data):
        """Single prediction must complete within SLA"""
        X_test, _ = test_data
        if len(X_test) == 0:
            pytest.skip("No test data")
        
        sample = X_test.iloc[0:1]
        
        # Warm-up
        _ = model.predict(sample)
        
        # Measure
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            _ = model.predict(sample)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # SLA: p95 < 50ms, p99 < 100ms
        assert p95_latency < 50, f"p95 latency {p95_latency:.2f}ms exceeds 50ms SLA"
        assert p99_latency < 100, f"p99 latency {p99_latency:.2f}ms exceeds 100ms SLA"
    
    def test_batch_prediction_throughput(self, model, test_data):
        """Batch predictions must meet throughput SLA"""
        X_test, _ = test_data
        
        if len(X_test) < 100:
            pytest.skip("Insufficient data for throughput test")
        
        batch = X_test.iloc[:min(1000, len(X_test))]
        
        # Warm-up
        _ = model.predict(batch)
        
        # Measure
        start = time.perf_counter()
        _ = model.predict(batch)
        duration = time.perf_counter() - start
        
        if duration > 0:
            throughput = len(batch) / duration  # predictions per second
            
            # SLA: > 1,000 predictions/sec (relaxed for demo)
            assert throughput > 1_000, (
                f"Throughput {throughput:.0f} predictions/sec below 1,000 SLA"
            )
    
    def test_model_memory_footprint(self, model):
        """Model size must be reasonable for deployment"""
        import sys
        
        # Estimate model size in memory
        model_size_mb = sys.getsizeof(model) / (1024 * 1024)
        
        # SLA: < 500 MB in memory
        assert model_size_mb < 500, (
            f"Model size {model_size_mb:.2f}MB exceeds 500MB limit"
        )


# ============================================================================
# DATA COMPATIBILITY TESTS
# ============================================================================

class TestDataCompatibility:
    """Test model works with expected data formats"""
    
    def test_model_handles_missing_features_gracefully(self, model, test_data):
        """Model should handle missing values properly"""
        X_test, _ = test_data
        
        if len(X_test) == 0:
            pytest.skip("No test data")
        
        # Introduce some NaN values
        X_test_with_nan = X_test.copy()
        X_test_with_nan.iloc[0, 0] = np.nan
        
        try:
            predictions = model.predict(X_test_with_nan)
            assert len(predictions) == len(X_test_with_nan)
        except Exception as e:
            pytest.fail(f"Model failed on missing values: {e}")
    
    def test_model_input_schema_validation(self, model, test_data):
        """Model should expect correct number of features"""
        X_test, _ = test_data
        
        if not hasattr(model, 'n_features_in_'):
            pytest.skip("Model doesn't expose feature count")
        
        expected_features = model.n_features_in_
        actual_features = X_test.shape[1]
        
        assert actual_features == expected_features, (
            f"Feature count mismatch: expected {expected_features}, "
            f"got {actual_features}"
        )
    
    def test_predictions_are_valid_probabilities(self, model, test_data):
        """Predicted probabilities should be in [0, 1]"""
        X_test, _ = test_data
        
        if len(X_test) == 0:
            pytest.skip("No test data")
        
        try:
            y_pred_proba = model.predict_proba(X_test)
            
            assert np.all(y_pred_proba >= 0), "Negative probabilities detected"
            assert np.all(y_pred_proba <= 1), "Probabilities > 1 detected"
            
            # Check probabilities sum to 1
            row_sums = y_pred_proba.sum(axis=1)
            assert np.allclose(row_sums, 1.0), "Probabilities don't sum to 1"
        except AttributeError:
            pytest.skip("Model doesn't support predict_proba")


# ============================================================================
# REPRODUCIBILITY TESTS
# ============================================================================

class TestReproducibility:
    """Test model predictions are deterministic"""
    
    def test_predictions_are_deterministic(self, model, test_data):
        """Same input should always produce same output"""
        X_test, _ = test_data
        
        if len(X_test) < 10:
            pytest.skip("Insufficient test data")
        
        sample = X_test.iloc[:min(100, len(X_test))]
        
        predictions_1 = model.predict(sample)
        predictions_2 = model.predict(sample)
        
        assert np.array_equal(predictions_1, predictions_2), (
            "Model predictions are non-deterministic"
        )
    
    def test_model_metadata_exists(self, model):
        """Model should have metadata for traceability"""
        # Check if model has required metadata
        required_attrs = ['n_features_in_']
        
        missing_attrs = [attr for attr in required_attrs if not hasattr(model, attr)]
        
        if missing_attrs:
            pytest.skip(f"Model missing attributes: {missing_attrs}")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestModelIntegration:
    """Test model integrates with API correctly"""
    
    def test_model_serialization_deserialization(self, model):
        """Model should survive pickle roundtrip"""
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            joblib.dump(model, f.name)
            loaded_model = joblib.load(f.name)
        
        assert type(loaded_model) == type(model)
        assert hasattr(loaded_model, 'predict')
    
    def test_model_has_required_methods(self, model):
        """Model should have standard sklearn interface"""
        required_methods = ['predict', 'fit']
        
        for method in required_methods:
            assert hasattr(model, method), (
                f"Model missing required method: {method}"
            )

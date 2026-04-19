from pathlib import Path
import json
import os

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.utils.validation import check_is_fitted

from src.mlops_project.api.service import MODEL_PATH, PREPROCESSOR_PATH
from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs


pytestmark = pytest.mark.ct


TARGET_COLUMN = "churn_status"
RAW_DATA_PATH = Path("data/raw/telcom_churn.csv")
BASELINE_METRICS_PATH = Path("artifacts/baseline/metrics.json")


def _require_file_exists_or_skip(path: Path, purpose: str) -> None:
    if not path.exists():
        pytest.skip(f"Missing required {purpose}: {path}")


def _load_model_bundle() -> tuple[object, float]:
    model_path = Path(MODEL_PATH)
    _require_file_exists_or_skip(model_path, "model artifact")

    try:
        loaded = joblib.load(model_path)
    except Exception as exc:
        pytest.skip(f"Cannot load model artifact {model_path}: {exc}")

    # Extract model and threshold from loaded bundle
    if isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        threshold = loaded.get("threshold", 0.5)
    else:
        # Assume loaded object is the model itself
        model = loaded
        threshold = 0.5

    return model, threshold


def _model_is_fitted(model) -> bool:
    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def _require_fitted(model) -> None:
    if not _model_is_fitted(model):
        force_run = os.getenv("FORCE_RUN_UNFITTED_MODEL_TESTS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if force_run:
            return
        pytest.skip("Real model artifact exists but is not fitted yet")


def _load_preprocessor_bundle() -> tuple[object, list[str]]:
    preprocessor_path = Path(PREPROCESSOR_PATH)
    _require_file_exists_or_skip(preprocessor_path, "preprocessor artifact")

    try:
        bundle = joblib.load(preprocessor_path)
    except Exception as exc:
        pytest.skip(f"Cannot load preprocessor artifact {preprocessor_path}: {exc}")

    assert "pipeline" in bundle, "Preprocessor artifact must contain 'pipeline'"
    assert "feature_columns" in bundle, "Preprocessor artifact must contain 'feature_columns'"
    return bundle["pipeline"], list(bundle["feature_columns"])


def _load_real_feature_inputs() -> tuple[pd.DataFrame, pd.Series]:
    _require_file_exists_or_skip(RAW_DATA_PATH, "raw dataset")

    raw_df = pd.read_csv(RAW_DATA_PATH)
    validated_df, _ = clean_raw_dataframe(raw_df)
    feature_source_df, _ = prepare_feature_inputs(validated_df)

    assert TARGET_COLUMN in feature_source_df.columns, "Target column missing after feature preparation"
    y_true = feature_source_df[TARGET_COLUMN]
    x_inputs = feature_source_df.drop(columns=[TARGET_COLUMN])
    return x_inputs, y_true


def _transform_inputs(x_inputs: pd.DataFrame, preprocessor, feature_columns: list[str]) -> pd.DataFrame:
    model_input_df = x_inputs.reindex(columns=feature_columns)
    transformed = preprocessor.transform(model_input_df)
    feature_names = preprocessor.get_feature_names_out()
    return pd.DataFrame(transformed, columns=feature_names, index=x_inputs.index)


def _load_baseline_metrics() -> dict[str, float]:
    if BASELINE_METRICS_PATH.exists():
        with BASELINE_METRICS_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"f1_score": 0.60, "roc_auc": 0.65}


def test_ct_model_preprocessor_and_raw_dataset_on_disk():
    _require_file_exists_or_skip(Path(MODEL_PATH), "model artifact")
    _require_file_exists_or_skip(Path(PREPROCESSOR_PATH), "preprocessor artifact")
    _require_file_exists_or_skip(RAW_DATA_PATH, "raw dataset")


def test_ct_preprocessor_columns_are_compatible_with_engineered_features():
    x_inputs, _ = _load_real_feature_inputs()
    _, feature_columns = _load_preprocessor_bundle()

    missing_for_preprocessor = set(feature_columns) - set(x_inputs.columns)
    assert not missing_for_preprocessor, (
        f"Real feature inputs are missing expected preprocessor columns: {sorted(missing_for_preprocessor)}"
    )


def test_ct_saved_estimator_is_fitted_for_inference():
    model, _ = _load_model_bundle()
    assert _model_is_fitted(model), "Model artifact exists but is not fitted for inference"


def test_ct_predicted_probabilities_are_within_unit_interval():
    model, _ = _load_model_bundle()
    _require_fitted(model)

    preprocessor, feature_columns = _load_preprocessor_bundle()
    x_inputs, _ = _load_real_feature_inputs()

    transformed_df = _transform_inputs(x_inputs, preprocessor, feature_columns)
    y_proba = model.predict_proba(transformed_df)[:, 1]

    assert len(y_proba) == len(transformed_df)
    assert np.all(y_proba >= 0.0)
    assert np.all(y_proba <= 1.0)


def test_ct_metrics_meet_baseline_quality_gate():
    model, threshold = _load_model_bundle()
    _require_fitted(model)

    preprocessor, feature_columns = _load_preprocessor_bundle()
    x_inputs, y_true = _load_real_feature_inputs()
    baseline = _load_baseline_metrics()

    transformed_df = _transform_inputs(x_inputs, preprocessor, feature_columns)
    y_proba = model.predict_proba(transformed_df)[:, 1]

    # Compute metrics inline
    y_pred = (y_proba >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    metrics = {"f1_score": f1, "roc_auc": auc}

    baseline_f1 = float(baseline.get("f1_score", 0.60))
    baseline_auc = float(baseline.get("roc_auc", 0.65))
    f1_threshold = baseline_f1 * 0.95
    auc_threshold = baseline_auc * 0.95

    assert metrics["f1_score"] >= f1_threshold, f"F1 {metrics['f1_score']:.4f} below quality gate {f1_threshold:.4f}"
    assert metrics["roc_auc"] >= auc_threshold, f"ROC AUC {metrics['roc_auc']:.4f} below quality gate {auc_threshold:.4f}"


def test_ct_classifier_predictions_are_identical_on_repeated_forward_pass():
    model, _ = _load_model_bundle()
    _require_fitted(model)

    preprocessor, feature_columns = _load_preprocessor_bundle()
    x_inputs, _ = _load_real_feature_inputs()

    sample = x_inputs.iloc[: min(100, len(x_inputs))]
    transformed_df = _transform_inputs(sample, preprocessor, feature_columns)

    predictions_1 = model.predict(transformed_df)
    predictions_2 = model.predict(transformed_df)
    assert np.array_equal(predictions_1, predictions_2), "Predictions are non-deterministic"


def test_ct_roc_auc_is_within_valid_numeric_bounds():
    model, _ = _load_model_bundle()
    _require_fitted(model)

    preprocessor, feature_columns = _load_preprocessor_bundle()
    x_inputs, y_true = _load_real_feature_inputs()

    transformed_df = _transform_inputs(x_inputs, preprocessor, feature_columns)
    y_proba = model.predict_proba(transformed_df)[:, 1]
    auc = roc_auc_score(y_true, y_proba)

    assert 0.0 <= auc <= 1.0

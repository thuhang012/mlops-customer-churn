from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.utils.validation import check_is_fitted

from src.mlops_project.api.serve import app
from src.mlops_project.api.service import MODEL_PATH

client = TestClient(app)

RAW_DATA_PATH = Path("data/raw/netflix_large.csv")
pytestmark = pytest.mark.fast


def _load_real_payloads() -> list[dict]:
    assert RAW_DATA_PATH.exists(), f"Real dataset not found: {RAW_DATA_PATH}"
    df = pd.read_csv(RAW_DATA_PATH)
    assert not df.empty, "Real dataset is empty"

    records = df.head(2).to_dict(orient="records")
    payloads = []
    for record in records:
        payloads.append(
            {
                "user_id": str(record["user_id"]),
                "age_group": int(record["age_group"]),
                "gender": str(record["gender"]),
                "country": str(record["country"]),
                "region": str(record["region"]),
                "subscription_plan": str(record["subscription_plan"]),
                "monthly_fee": float(record["monthly_fee"]),
                "subscription_start_date": str(record["subscription_start_date"]),
                "subscription_end_date": str(record["subscription_end_date"]),
                "payment_method": str(record["payment_method"]),
                "discount_applied": str(record["discount_applied"]),
                "title": str(record["title"]),
                "content_type": str(record["content_type"]),
                "genre": str(record["genre"]),
                "language": str(record["language"]),
                "release_year": int(record["release_year"]),
                "device_type": str(record["device_type"]),
                "watch_time_minutes": int(record["watch_time_minutes"]),
                "session_count": int(record["session_count"]),
                "completion_percentage": int(record["completion_percentage"]),
                "date_watched": str(record["date_watched"]),
                "time_of_day": str(record["time_of_day"]),
                "rating": int(record["rating"]),
                "liked": str(record["liked"]),
                "recommendation_source": str(record["recommendation_source"]),
                "days_since_last_watch": int(record["days_since_last_watch"]),
                "avg_weekly_watch_time": int(record["avg_weekly_watch_time"]),
                "content_diversity_score": float(record["content_diversity_score"]),
            }
        )

    assert payloads, "Could not construct any payload from real dataset"
    return payloads


def _is_model_artifact_fitted() -> bool:
    model_path = Path(MODEL_PATH)
    assert model_path.exists(), f"Model artifact not found: {model_path}"

    model = joblib.load(model_path)
    if isinstance(model, dict):
        model = model.get("model")

    try:
        check_is_fitted(model)
        return True
    except Exception:
        return False


def test_api_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API is running"}


def test_api_health_reports_ok_and_loaded_artifacts():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data.get("model_loaded") is True
    assert data.get("preprocessor_loaded") is True


def test_api_predict_single_respects_model_fit_state():
    sample_payload = _load_real_payloads()[0]
    response = client.post("/predict", json=sample_payload)

    if _is_model_artifact_fitted():
        assert response.status_code == 200, response.text
        data = response.json()
        assert "churn_probability" in data
        assert isinstance(data["churn_probability"], float)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["prediction"] in [0, 1]
    else:
        assert response.status_code == 400, response.text
        detail = response.json().get("detail", "")
        assert "not fitted" in detail.lower()


def test_api_predict_batch_respects_model_fit_state():
    payloads = _load_real_payloads()
    response = client.post("/batch-predict", json=payloads)

    if _is_model_artifact_fitted():
        assert response.status_code == 200, response.text
        predictions = response.json()
        assert len(predictions) == len(payloads)
        for prediction in predictions:
            assert 0.0 <= prediction["churn_probability"] <= 1.0
            assert prediction["prediction"] in [0, 1]
    else:
        assert response.status_code == 400, response.text
        detail = response.json().get("detail", "")
        assert "not fitted" in detail.lower()


def test_api_predict_rejects_missing_required_field():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload.pop("gender")

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_api_predict_rejects_invalid_field_type():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload["monthly_fee"] = "abc"

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422

import pytest
from uuid import uuid4
from fastapi.testclient import TestClient

from src.mlops_project.api.serve import app

client = TestClient(app)

pytestmark = pytest.mark.fast


def _load_real_payloads() -> list[dict]:
    suffix = uuid4().hex[:8]
    return [
        {
            "customerID": f"0001-A-{suffix}",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.5,
            "TotalCharges": 954.0,
        },
        {
            "customerID": f"0002-B-{suffix}",
            "gender": "Male",
            "SeniorCitizen": 1,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 2,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "No",
            "PaymentMethod": "Mailed check",
            "MonthlyCharges": 24.25,
            "TotalCharges": 48.5,
        },
    ]


def test_api_root_returns_welcome_message():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API is running"}


def test_api_health_reports_ok_and_loaded_artifacts():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert isinstance(data.get("model_loaded"), bool)
    assert isinstance(data.get("preprocessor_loaded"), bool)


def test_api_predict_single_respects_model_fit_state():
    sample_payload = _load_real_payloads()[0]
    response = client.post("/predict", json=sample_payload)

    if response.status_code == 200:
        assert response.status_code == 200, response.text
        data = response.json()
        assert "churn_probability" in data
        assert isinstance(data["churn_probability"], float)
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["prediction"] in [0, 1]
    else:
        assert response.status_code in [400, 500], response.text
        detail = response.json().get("detail", "")
        detail_lower = detail.lower()
        assert (
            "not fitted" in detail_lower
            or "prediction failed" in detail_lower
            or "error loading model artifact" in detail_lower
            or "internal server error" in detail_lower
        )


def test_api_predict_batch_respects_model_fit_state():
    payloads = _load_real_payloads()
    response = client.post("/batch-predict", json=payloads)

    if response.status_code == 200:
        assert response.status_code == 200, response.text
        predictions = response.json()
        assert len(predictions) == len(payloads)
        for prediction in predictions:
            assert 0.0 <= prediction["churn_probability"] <= 1.0
            assert prediction["prediction"] in [0, 1]
    else:
        assert response.status_code in [400, 500], response.text
        detail = response.json().get("detail", "")
        detail_lower = detail.lower()
        assert (
            "not fitted" in detail_lower
            or "batch prediction failed" in detail_lower
            or "error loading model artifact" in detail_lower
            or "internal server error" in detail_lower
        )


def test_api_predict_rejects_missing_required_field():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload.pop("gender")

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_api_predict_rejects_invalid_field_type():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload["MonthlyCharges"] = "abc"

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_api_predict_rejects_null_customer_id():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload["customerID"] = None

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422


def test_api_predict_rejects_total_charges_less_than_monthly():
    sample_payload = _load_real_payloads()[0]
    bad_payload = sample_payload.copy()
    bad_payload["MonthlyCharges"] = 100.0
    bad_payload["TotalCharges"] = 50.0

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422

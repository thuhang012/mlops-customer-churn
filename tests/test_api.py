from fastapi.testclient import TestClient

from src.mlops_project.api.serve import app

client = TestClient(app)


sample_payload = {
    "user_id": "U001",
    "age_group": 2,
    "gender": "Female",
    "country": "Vietnam",
    "region": "HCM",
    "subscription_plan": "Premium",
    "monthly_fee": 12.99,
    "subscription_start_date": "2024-01-01",
    "subscription_end_date": "2024-12-31",
    "payment_method": "Credit Card",
    "discount_applied": "Yes",
    "title": "Movie A",
    "content_type": "Movie",
    "genre": "Drama",
    "language": "English",
    "release_year": 2023,
    "device_type": "Mobile",
    "watch_time_minutes": 120,
    "session_count": 5,
    "completion_percentage": 80,
    "date_watched": "2026-04-01",
    "time_of_day": "Evening",
    "rating": 4,
    "liked": "Yes",
    "recommendation_source": "Homepage",
    "days_since_last_watch": 2,
    "avg_weekly_watch_time": 300,
    "content_diversity_score": 0.75
}


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Churn Prediction API is running"}


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_success():
    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    data = response.json()
    assert "churn_probability" in data
    assert isinstance(data["churn_probability"], float)
    assert 0.0 <= data["churn_probability"] <= 1.0

# case thiếu field
def test_predict_missing_field():
    bad_payload = sample_payload.copy()
    bad_payload.pop("gender")

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422

# sai dtype
def test_predict_invalid_type():
    bad_payload = sample_payload.copy()
    bad_payload["monthly_fee"] = "abc"

    response = client.post("/predict", json=bad_payload)

    assert response.status_code == 422
from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="Netflix Churn Prediction Mock API")

class CustomerData(BaseModel):
    tenure: int
    contractType: str
    monthlyCharges: float

class PredictionOutput(BaseModel):
    churn: int
    probability: float

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "churn-api"}

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerData):
    # Dữ liệu nháp, luôn trả về random xác suất để kiểm thử API
    mock_prob = round(random.uniform(0.1, 0.9), 2)
    mock_churn = 1 if mock_prob > 0.5 else 0
    return {"churn": mock_churn, "probability": mock_prob}

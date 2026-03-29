from fastapi import FastAPI

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.api.service import predict

app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    description="Mock FastAPI for sprint 1"
)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    churn_probability = predict(data)
    return PredictionOutput(churn_probability=churn_probability)

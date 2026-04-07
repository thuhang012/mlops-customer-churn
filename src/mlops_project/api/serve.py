from fastapi import FastAPI, HTTPException

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.api.service import predict
from src.mlops_project.utils.logger import log_inference

app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    description="FastAPI service for churn prediction"
)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    try:
        result = predict(data)

        # log_inference(data.dict(), churn_probability)  # nếu Pydantic v2 thì đổi model_dump()
        log_inference(data.model_dump(), result["churn_probability"])
        return PredictionOutput(
            churn_probability=result["churn_probability"],
            prediction=result["prediction"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
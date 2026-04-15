# uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, HTTPException

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.api.service import predict, batch_predict, artifacts_status
from src.mlops_project.utils.logger import log_inference

from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(
    title="Churn Prediction API",
    version="0.1.0",
    description="FastAPI service for churn prediction",
)

Instrumentator().instrument(app).expose(app)

@app.get("/")
def root():
    return {"message": "Churn Prediction API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok", **artifacts_status()}


@app.post("/predict", response_model=PredictionOutput)
def predict_churn(data: CustomerInput):
    try:
        result = predict(data)
        log_inference(data.model_dump(), result.churn_probability)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/batch-predict", response_model=list[PredictionOutput])
def batch_predict_churn(data: list[CustomerInput]):
    try:
        results = batch_predict(data)

        for item, result in zip(data, results):
            log_inference(item.model_dump(), result.churn_probability)

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail=str(e))

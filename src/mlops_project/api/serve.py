# uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram

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

API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total API requests",
    ["endpoint", "method", "status_class"],
)
API_REQUEST_ERRORS_TOTAL = Counter(
    "api_request_errors_total",
    "Total API request errors",
    ["endpoint", "method", "error_type"],
)
API_REQUEST_LATENCY_SECONDS = Histogram(
    "api_request_latency_seconds",
    "API request latency in seconds",
    ["endpoint", "method", "status_class"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
API_INFLIGHT_REQUESTS = Gauge(
    "api_inflight_requests",
    "In-flight API requests",
    ["endpoint"],
)
API_BATCH_SIZE = Histogram(
    "api_batch_size",
    "Batch size for /batch-predict requests",
    buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500),
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "api_model_predictions_total",
    "Total model predictions served by API",
    ["endpoint"],
)
MODEL_PREDICTION_CLASS_TOTAL = Counter(
    "api_model_prediction_class_total",
    "Predicted class counts served by API",
    ["predicted_class"],
)
MODEL_PREDICTION_PROBABILITY = Histogram(
    "api_model_prediction_probability",
    "Predicted churn probability distribution",
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_LOW_CONFIDENCE_TOTAL = Counter(
    "api_model_low_confidence_total",
    "Low-confidence model predictions",
    ["endpoint"],
)

LOW_CONFIDENCE_THRESHOLD = 0.60


def _observe_prediction(probability: float, endpoint: str) -> None:
    prob = min(max(float(probability), 0.0), 1.0)
    MODEL_PREDICTIONS_TOTAL.labels(endpoint=endpoint).inc()
    MODEL_PREDICTION_PROBABILITY.observe(prob)
    MODEL_PREDICTION_CLASS_TOTAL.labels(predicted_class="1" if prob >= 0.5 else "0").inc()

    confidence = max(prob, 1.0 - prob)
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        MODEL_LOW_CONFIDENCE_TOTAL.labels(endpoint=endpoint).inc()


@app.middleware("http")
async def prometheus_request_metrics(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    API_INFLIGHT_REQUESTS.labels(endpoint=endpoint).inc()
    start = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except HTTPException as exc:
        status_code = exc.status_code
        API_REQUEST_ERRORS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            error_type=f"http_{exc.status_code}",
        ).inc()
        raise
    except Exception:
        status_code = 500
        API_REQUEST_ERRORS_TOTAL.labels(
            endpoint=endpoint,
            method=method,
            error_type="unhandled_exception",
        ).inc()
        raise
    finally:
        duration = perf_counter() - start
        status_class = f"{status_code // 100}xx"
        API_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_class=status_class).inc()
        API_REQUEST_LATENCY_SECONDS.labels(
            endpoint=endpoint,
            method=method,
            status_class=status_class,
        ).observe(duration)
        API_INFLIGHT_REQUESTS.labels(endpoint=endpoint).dec()

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
        _observe_prediction(probability=result.churn_probability, endpoint="/predict")
        log_inference(data.model_dump(), result.churn_probability)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.post("/batch-predict", response_model=list[PredictionOutput])
def batch_predict_churn(data: list[CustomerInput]):
    try:
        API_BATCH_SIZE.observe(len(data))
        results = batch_predict(data)

        for item, result in zip(data, results):
            _observe_prediction(probability=result.churn_probability, endpoint="/batch-predict")
            log_inference(item.model_dump(), result.churn_probability)

        return results

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

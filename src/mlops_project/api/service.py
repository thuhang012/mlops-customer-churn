import joblib
import pandas as pd

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs

import os

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/Telco_Churn_Model_final.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessors/preprocessor.pkl")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

model = None
threshold = 0.5
model_name = None

preprocessor = None
raw_feature_columns = None
transformed_feature_columns = None

def _load_artifacts():
    global model, threshold, model_name
    global preprocessor, raw_feature_columns, transformed_feature_columns

    if model is None or preprocessor is None:
        try:
            model_artifact = joblib.load(MODEL_PATH)
            preprocessor_artifact = joblib.load(PREPROCESSOR_PATH)

            model = model_artifact["model"]
            threshold = float(model_artifact.get("threshold", 0.5))
            model_name = model_artifact.get("model_name", "unknown")

            preprocessor = preprocessor_artifact["pipeline"]
            raw_feature_columns = preprocessor_artifact["feature_columns"]

            transformed_feature_columns = list(preprocessor.get_feature_names_out())

        except Exception as e:
            raise RuntimeError(f"Error loading model artifact: {str(e)}")

def artifacts_status() -> dict:
    try:
        _load_artifacts()
        return {
            "model_loaded": model is not None,
            "preprocessor_loaded": preprocessor is not None,
            "model_name": model_name,
            "threshold": threshold,
            "raw_feature_columns_available": raw_feature_columns is not None,
            "transformed_feature_columns_available": transformed_feature_columns is not None,
        }
    except Exception:
        return {
            "model_loaded": False,
            "preprocessor_loaded": False,
            "model_name": None,
            "threshold": None,
            "raw_feature_columns_available": False,
            "transformed_feature_columns_available": False,
        }

def _prepare_model_input(raw_df: pd.DataFrame) -> pd.DataFrame:
    validated_df, _ = clean_raw_dataframe(raw_df)
    feature_source_df, _ = prepare_feature_inputs(validated_df)

    # Align theo raw feature columns trước khi transform
    model_input_df = feature_source_df.reindex(columns=raw_feature_columns, fill_value=0)

    transformed = preprocessor.transform(model_input_df)

    transformed_df = pd.DataFrame(
        transformed,
        columns=transformed_feature_columns,
        index=model_input_df.index,
    )

    return transformed_df

def predict(data: CustomerInput) -> PredictionOutput:
    _load_artifacts()
    try:
        raw_df = pd.DataFrame([data.model_dump()])
        model_input_df = _prepare_model_input(raw_df)

        churn_prob = float(model.predict_proba(model_input_df)[0][1])
        churn_label = int(churn_prob >= threshold)

        return PredictionOutput(
            churn_probability=churn_prob,
            prediction=churn_label,
            threshold=threshold,
            model_name=model_name,
        )

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def batch_predict(data_list: list[CustomerInput]) -> list[PredictionOutput]:
    _load_artifacts()
    try:
        raw_df = pd.DataFrame([item.model_dump() for item in data_list])
        model_input_df = _prepare_model_input(raw_df)

        proba = model.predict_proba(model_input_df)[:, 1]
        labels = (proba >= threshold).astype(int)

        return [
            PredictionOutput(
                churn_probability=float(prob),
                prediction=int(label),
                threshold=threshold,
                model_name=model_name,
            )
            for prob, label in zip(proba, labels)
        ]

    except Exception as e:
        raise ValueError(f"Batch prediction failed: {str(e)}")

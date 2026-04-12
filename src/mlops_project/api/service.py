import joblib
import pandas as pd

from src.mlops_project.api.schema import CustomerInput, PredictionOutput
from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs

MODEL_PATH = "artifacts/models/Netflix_Prediction_final.pkl"
PREPROCESSOR_PATH = "artifacts/preprocessors/preprocessor.pkl"

try:
    model = joblib.load(MODEL_PATH)
    preprocessor_artifact = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or preprocessor: {str(e)}")

preprocessor = preprocessor_artifact["pipeline"]
feature_columns = preprocessor_artifact["feature_columns"]

def artifacts_status() -> dict:
    return {
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

def predict(data: CustomerInput) -> PredictionOutput:
    try:
        # raw_df = pd.DataFrame([data.dict()])  #  Pydantic v1
        raw_df = pd.DataFrame([data.model_dump()])
        
        validated_df, _ = clean_raw_dataframe(raw_df)
        feature_source_df, _ = prepare_feature_inputs(validated_df)

        model_input_df = feature_source_df.reindex(columns=feature_columns)
        transformed = preprocessor.transform(model_input_df)

        ##
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        # Binary classification: probability of positive class (churn = 1)
        proba = model.predict_proba(transformed_df)
        churn_prob = proba[0][1]
        churn_label = int(churn_prob > 0.5)

        return PredictionOutput(
            churn_probability=float(churn_prob),
            prediction=churn_label
        )

    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

def batch_predict(data_list: list[CustomerInput]) -> list[PredictionOutput]:
    try:
        raw_df = pd.DataFrame([item.model_dump() for item in data_list])

        validated_df, _ = clean_raw_dataframe(raw_df)
        feature_source_df, _ = prepare_feature_inputs(validated_df)

        model_input_df = feature_source_df.reindex(columns=feature_columns)
        transformed = preprocessor.transform(model_input_df)

        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        proba = model.predict_proba(transformed_df)[:, 1]
        labels = (proba > 0.5).astype(int)

        results = [
            PredictionOutput(
                churn_probability=float(prob),
                prediction=int(label)
            )
            for prob, label in zip(proba, labels)
        ]

        return results

    except Exception as e:
        raise ValueError(f"Batch prediction failed: {str(e)}")

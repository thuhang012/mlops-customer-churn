import joblib
import pandas as pd

from src.mlops_project.api.schema import CustomerInput
from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs

model = joblib.load("artifacts/models/Netflix_Prediction_final.pkl")
preprocessor_artifact = joblib.load("artifacts/preprocessors/preprocessor.pkl")

preprocessor = preprocessor_artifact["pipeline"]
feature_columns = preprocessor_artifact["feature_columns"]

def predict(data: CustomerInput) -> float:
    try:
        # raw_df = pd.DataFrame([data.dict()])  # đổi sang model_dump() nếu Pydantic v2
        raw_df = pd.DataFrame([data.dict()])
        
        validated_df, _ = clean_raw_dataframe(raw_df)
        feature_source_df, _ = prepare_feature_inputs(validated_df)

        model_input_df = feature_source_df.reindex(columns=feature_columns)
        transformed = preprocessor.transform(model_input_df)

        ##
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        proba = model.predict_proba(transformed_df)
        # proba = model.predict_proba(transformed)
        churn_prob = proba[0][1]

        churn_label = int(churn_prob > 0.5)

        return {
            "churn_probability": float(churn_prob),
            "prediction": churn_label
        }


        return float(churn_prob)
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")
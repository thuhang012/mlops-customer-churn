import os
from datetime import datetime, timezone
from typing import Any

import joblib
import pandas as pd

from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs

INFERENCE_LOG_PATH = "data/processed/inference_log.csv"
INFERENCE_LOG_RAW_PATH = "data/processed/inference_log_raw.csv"
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessors/preprocessor.pkl")


_LOG_PREPROCESSOR = None
_RAW_FEATURE_COLUMNS = None
_TRANSFORMED_FEATURE_COLUMNS = None


def _load_log_preprocessor() -> tuple[Any, list[str], list[str]]:
    global _LOG_PREPROCESSOR, _RAW_FEATURE_COLUMNS, _TRANSFORMED_FEATURE_COLUMNS

    if _LOG_PREPROCESSOR is None or _RAW_FEATURE_COLUMNS is None:
        artifact = joblib.load(PREPROCESSOR_PATH)
        _LOG_PREPROCESSOR = artifact["pipeline"]
        _RAW_FEATURE_COLUMNS = artifact["feature_columns"]
        _TRANSFORMED_FEATURE_COLUMNS = list(_LOG_PREPROCESSOR.get_feature_names_out())

    return _LOG_PREPROCESSOR, _RAW_FEATURE_COLUMNS, _TRANSFORMED_FEATURE_COLUMNS


def _prepare_log_features(input_data: dict[str, Any]) -> dict[str, Any]:
    raw_df = pd.DataFrame([input_data])
    validated_df, _ = clean_raw_dataframe(raw_df)
    feature_source_df, _ = prepare_feature_inputs(validated_df)

    preprocessor, raw_feature_columns, transformed_feature_columns = _load_log_preprocessor()
    model_input_df = feature_source_df.reindex(columns=raw_feature_columns, fill_value=0)
    transformed = preprocessor.transform(model_input_df)

    transformed_df = pd.DataFrame(
        transformed,
        columns=transformed_feature_columns,
        index=model_input_df.index,
    )
    return transformed_df.iloc[0].to_dict()


def _append_log_entry(path: str, log_entry: dict[str, Any]) -> None:
    df = pd.DataFrame([log_entry])

    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def log_inference(input_data: dict, prediction):
    timestamp = datetime.now(timezone.utc).isoformat()

    raw_log_entry = {
        **input_data,
        "prediction": prediction,
        "timestamp": timestamp,
    }

    try:
        processed_input = _prepare_log_features(input_data)
    except Exception:
        processed_input = input_data.copy()

    processed_log_entry = {
        **processed_input,
        "prediction": prediction,
        "timestamp": timestamp,
    }

    _append_log_entry(INFERENCE_LOG_RAW_PATH, raw_log_entry)
    _append_log_entry(INFERENCE_LOG_PATH, processed_log_entry)

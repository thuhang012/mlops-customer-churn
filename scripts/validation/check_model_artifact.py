from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.utils.validation import check_is_fitted

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mlops_project.data.validate_data import clean_raw_dataframe
from src.mlops_project.features.build_features import prepare_feature_inputs


DEFAULT_MODEL_PATH = Path("artifacts/models/Netflix_Prediction_final.pkl")
DEFAULT_PREPROCESSOR_PATH = Path("artifacts/preprocessors/preprocessor.pkl")
DEFAULT_RAW_DATA_PATH = Path("data/raw/netflix_large.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether the real churn model artifact can be loaded and used for inference."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the model artifact (.pkl).",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=DEFAULT_PREPROCESSOR_PATH,
        help="Path to the preprocessor artifact (.pkl).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_RAW_DATA_PATH,
        help="Path to the real raw dataset CSV.",
    )
    return parser.parse_args()


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    loaded = joblib.load(model_path)
    threshold = 0.5

    if isinstance(loaded, dict):
        if "model" not in loaded:
            raise ValueError(f"Model artifact dict missing 'model' key: {list(loaded.keys())}")
        threshold = float(loaded.get("threshold", threshold))
        model = loaded["model"]
    else:
        model = loaded

    return model, threshold


def load_preprocessor(preprocessor_path: Path):
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    bundle = joblib.load(preprocessor_path)
    if "pipeline" not in bundle or "feature_columns" not in bundle:
        raise ValueError("Preprocessor artifact must contain 'pipeline' and 'feature_columns'.")

    return bundle["pipeline"], list(bundle["feature_columns"])


def load_real_input(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {data_path}")

    raw_df = pd.read_csv(data_path)
    validated_df, _ = clean_raw_dataframe(raw_df)
    feature_source_df, _ = prepare_feature_inputs(validated_df)
    return feature_source_df


def main() -> int:
    args = parse_args()

    try:
        model, threshold = load_model(args.model)
        preprocessor, feature_columns = load_preprocessor(args.preprocessor)
        feature_source_df = load_real_input(args.data)

        print(f"Model artifact: {args.model}")
        print(f"Preprocessor artifact: {args.preprocessor}")
        print(f"Raw data: {args.data}")
        print(f"Loaded model type: {type(model).__name__}")
        print(f"Decision threshold: {threshold}")

        try:
            check_is_fitted(model)
            print("Model fitted status: FITTED")
        except Exception:
            print("Model fitted status: NOT FITTED")
            return 1

        missing_columns = sorted(set(feature_columns) - set(feature_source_df.columns))
        if missing_columns:
            print("Feature schema mismatch: missing columns from input data")
            print(missing_columns)
            return 1

        model_input_df = feature_source_df.reindex(columns=feature_columns)
        transformed = preprocessor.transform(model_input_df.head(1))
        feature_names = preprocessor.get_feature_names_out()
        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        if not hasattr(model, "predict_proba"):
            print("Model does not expose predict_proba")
            return 1

        proba = model.predict_proba(transformed_df)[:, 1]
        print(f"Sample churn probability: {float(proba[0]):.6f}")
        print("Inference check: PASSED")
        return 0

    except Exception as exc:
        print("Inference check: FAILED")
        print(f"Reason: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_TARGET_COLUMN = "churn_status"
DEFAULT_SPLIT_COLUMN = "data_split"
DEFAULT_TEST_LABEL = "test"
DEFAULT_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained churn model and export reports."
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to trained model .pkl file.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to processed dataset CSV containing data_split column.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory to save evaluation reports.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default=DEFAULT_TARGET_COLUMN,
        help="Target column name.",
    )
    parser.add_argument(
        "--split-column",
        type=str,
        default=DEFAULT_SPLIT_COLUMN,
        help="Split column name.",
    )
    parser.add_argument(
        "--test-label",
        type=str,
        default=DEFAULT_TEST_LABEL,
        help="Value in split column used for test rows.",
    )
    parser.add_argument(
        "--default-threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Default threshold for final metrics export.",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.10,
        help="Threshold search start.",
    )
    parser.add_argument(
        "--threshold-stop",
        type=float,
        default=0.91,
        help="Threshold search stop (exclusive in np.arange).",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        help="Threshold search step.",
    )
    parser.add_argument(
        "--recall-min",
        type=float,
        default=0.80,
        help="Minimum recall for recall-priority threshold table.",
    )
    return parser.parse_args()


def validate_inputs(
    df: pd.DataFrame,
    target_column: str,
    split_column: str,
    test_label: str,
) -> None:
    missing_columns = [c for c in [target_column, split_column] if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns in dataset: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )

    if df[df[split_column] == test_label].empty:
        raise ValueError(
            f"No rows found with {split_column} == '{test_label}'. "
            "Cannot evaluate on empty test set."
        )


def get_test_split(
    df: pd.DataFrame,
    target_column: str,
    split_column: str,
    test_label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    test_df = df[df[split_column] == test_label].copy()
    x_test = test_df.drop(columns=[target_column, split_column])
    y_test = test_df[target_column]
    return x_test, y_test

def extract_model_and_threshold(loaded_object, default_threshold: float) -> tuple[object, float]:
    if isinstance(loaded_object, dict):
        if "model" not in loaded_object:
            raise ValueError(
                "Loaded dict object does not contain 'model' key. "
                f"Available keys: {list(loaded_object.keys())}"
            )

        model = loaded_object["model"]
        threshold = float(loaded_object.get("threshold", default_threshold))
        return model, threshold

    return loaded_object, float(default_threshold)

def predict_probabilities(model, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)[:, 1]
        return np.asarray(y_proba, dtype=float)

    raise ValueError(
        "Loaded model does not support predict_proba. "
        f"Model type: {type(model)}"
    )

def compute_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "predicted_positive_rate": float(y_pred.mean()),
        "threshold": float(threshold),
    }
    return metrics


def build_threshold_table(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold_start: float,
    threshold_stop: float,
    threshold_step: float,
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    thresholds = np.arange(threshold_start, threshold_stop, threshold_step)

    for threshold in thresholds:
        threshold = round(float(threshold), 2)
        y_pred = (y_proba >= threshold).astype(int)

        rows.append(
            {
                "threshold": threshold,
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
                "predicted_positive_rate": float(y_pred.mean()),
            }
        )

    return pd.DataFrame(rows)


def ensure_reports_dir(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)


def save_json(data: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    model_path = args.model
    data_path = args.data
    reports_dir = args.reports_dir

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ensure_reports_dir(reports_dir)

    loaded_object = joblib.load(model_path)
    model, stored_threshold = extract_model_and_threshold(
        loaded_object=loaded_object,
        default_threshold=args.default_threshold,
    )
    artifact_model_name = None
    if isinstance(loaded_object, dict):
        artifact_model_name = loaded_object.get("model_name")

    model_name = artifact_model_name or type(model).__name__

    df = pd.read_csv(data_path)
    # model = joblib.load(model_path)
    # df = pd.read_csv(data_path)

    validate_inputs(
        df=df,
        target_column=args.target_column,
        split_column=args.split_column,
        test_label=args.test_label,
    )

    x_test, y_test = get_test_split(
        df=df,
        target_column=args.target_column,
        split_column=args.split_column,
        test_label=args.test_label,
    )

    y_proba = predict_probabilities(model, x_test)

    final_metrics = compute_metrics(
        y_true=y_test,
        y_proba=y_proba,
        threshold=stored_threshold,
    )

    final_metrics_payload = {
        "project": "Netflix_Prediction",
        "model_name": model_name,
        "selection_metric": "pr_auc",
        "test_set_label": args.test_label,
        "stored_threshold": float(stored_threshold),
        "metrics": final_metrics,
    }

    threshold_df = build_threshold_table(
        y_true=y_test,
        y_proba=y_proba,
        threshold_start=args.threshold_start,
        threshold_stop=args.threshold_stop,
        threshold_step=args.threshold_step,
    )

    threshold_df = threshold_df.sort_values(
        by="f1_score",
        ascending=False,
    ).reset_index(drop=True)

    recall_priority_df = threshold_df[threshold_df["recall"] >= args.recall_min].copy()
    recall_priority_df = recall_priority_df.sort_values(
        by=["precision", "f1_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    best_threshold_row = threshold_df.iloc[0].to_dict()
    best_threshold_payload = {
        "best_threshold_by_f1": float(best_threshold_row["threshold"]),
        "f1_score": float(best_threshold_row["f1_score"]),
        "precision": float(best_threshold_row["precision"]),
        "recall": float(best_threshold_row["recall"]),
        "predicted_positive_rate": float(best_threshold_row["predicted_positive_rate"]),
    }

    threshold_dir = reports_dir / "threshold"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    final_metrics_path = reports_dir / "final_metrics.json"
    threshold_metrics_path = threshold_dir / "threshold_metrics.csv"
    threshold_recall_priority_path = threshold_dir / "threshold_recall_priority.csv"
    best_threshold_path = threshold_dir / "best_threshold.json"

    save_json(final_metrics_payload, final_metrics_path)
    threshold_df.to_csv(threshold_metrics_path, index=False)
    recall_priority_df.to_csv(threshold_recall_priority_path, index=False)
    save_json(best_threshold_payload, best_threshold_path)

    print("Evaluation completed successfully.")
    print(f"Saved: {final_metrics_path}")
    print(f"Saved: {threshold_metrics_path}")
    print(f"Saved: {threshold_recall_priority_path}")
    print(f"Saved: {best_threshold_path}")


if __name__ == "__main__":
    main()

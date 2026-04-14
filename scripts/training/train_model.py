"""
Model Training Script
Trains model from a real processed dataset.
"""

import joblib
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd


def train_model(output_dir: str = "artifacts", data_path: str = "data/processed/cleaned_data.csv") -> dict:
    """
    Train model and save metrics

    Args:
        output_dir: Directory to save model and metrics
        data_path: Path to processed CSV containing features and churn_status

    Returns:
        Dictionary with model_id, f1_score, roc_auc
    """
    try:
        # Setup directories
        output_dir = Path(output_dir)
        models_dir = output_dir / "models"
        metrics_dir = output_dir / "metrics"

        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        print("Training model from real processed data...")
        model_id = f"model-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        dataset_path = Path(data_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found at {dataset_path}. Run preprocessing and DVC pull first.")

        df = pd.read_csv(dataset_path)
        if df.empty:
            raise ValueError("Processed dataset is empty")
        if "churn_status" not in df.columns:
            raise ValueError("Missing required target column: churn_status")

        X = df.drop(columns=["churn_status", "data_split"], errors="ignore")
        y = df["churn_status"]

        if X.empty:
            raise ValueError("No feature columns available after dropping target columns")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        model_path = models_dir / "model.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

        # Calculate metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            roc_auc = 0.0

        # Save metrics
        metrics = {"f1_score": float(f1), "roc_auc": float(roc_auc), "n_samples": len(X_test), "n_features": X.shape[1]}

        metrics_path = metrics_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved: {metrics_path}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")

        _write_outputs(model_id, f1, roc_auc)
        return {"model_id": model_id, "f1_score": f1, "roc_auc": roc_auc, "model_path": str(model_path)}

    except Exception as e:
        print(f"ERROR: Model training failed: {e}")
        _write_outputs("", 0, 0)
        sys.exit(1)


def _write_outputs(model_id: str, f1_score_value: float, roc_auc_value: float) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a") as f:
            f.write(f"model_id={model_id}\n")
            f.write(f"f1_score={f1_score_value}\n")
            f.write(f"roc_auc={roc_auc_value}\n")


if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
    input_data = sys.argv[2] if len(sys.argv) > 2 else "data/processed/cleaned_data.csv"
    results = train_model(output_dir, input_data)

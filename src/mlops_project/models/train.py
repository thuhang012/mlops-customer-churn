import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import yaml  # Quay lại sử dụng yaml
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import logging
from mlflow.tracking import MlflowClient

import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Churn model with YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to best_model_config.yaml")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to processed CSV with data_split column",
    )
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns_local"),
        help="MLflow tracking URI",
    )
    return parser.parse_args()


def get_model_instance(name, params):
    model_map = {
        "RandomForest": RandomForestClassifier,
        "XGBoost": XGBClassifier,
        "LightGBM": LGBMClassifier,
        "CatBoost": CatBoostClassifier,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}. Check 'best_model_overall' in YAML.")
    return model_map[name](**params)


def main(args):
    # 1. Load YAML config (Cấu trúc phẳng)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Trích xuất thông tin từ cấu trúc YAML của bạn
    model_name = config.get("project", "Netflix_Churn_Model")
    best_algo = config.get("best_model_overall")
    params = config.get("hyperparameters")

    if not best_algo or not params:
        raise KeyError("YAML file missing 'best_model_overall' or 'hyperparameters'")

    # Add safe defaults for CatBoost final training
    if best_algo == "CatBoost":
        params = {
            **params,
            "loss_function": "Logloss",
            "verbose": 0,
            "allow_writing_files": False,
            "random_state": 42,
        }

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(model_name)

    # 2. Load data & Split theo cấu trúc của bạn
    df = pd.read_csv(args.data)
    target = "churn_status"  # Tên cột mục tiêu
    split_col = "data_split"

    required_cols = {target, split_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    train_set = df[df[split_col] == "train"]
    test_set = df[df[split_col] == "test"]

    if train_set.empty or test_set.empty:
        raise ValueError("Train set or test set is empty. Check 'data_split' column.")

    X_train = train_set.drop(columns=[target, split_col])
    y_train = train_set[target]

    X_test = test_set.drop(columns=[target, split_col])
    y_test = test_set[target]

    # 3. Khởi tạo mô hình
    model = get_model_instance(best_algo, params)
    SELECTED_THRESHOLD = float(config.get("threshold", 0.5))
    # SELECTED_THRESHOLD = 0.5

    with mlflow.start_run(run_name=f"final_training_{best_algo}"):
        logger.info(f"Training {best_algo} on custom split...")
        model.fit(X_train, y_train)

        # Dự đoán để tính metrics
        # y_pred = model.predict(X_test)
        # y_proba = model.predict_proba(X_test)[:, 1]
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= SELECTED_THRESHOLD).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
            "predicted_positive_rate": float(y_pred.mean()),
        }

        # Handle metrics that fail with single class in y_test
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            metrics["roc_auc"] = 0.0

        try:
            metrics["pr_auc"] = float(average_precision_score(y_test, y_proba))
        except Exception:
            metrics["pr_auc"] = 0.0

        try:
            metrics["log_loss"] = float(log_loss(y_test, y_proba, labels=[0, 1]))
        except Exception:
            metrics["log_loss"] = 0.0

        # Log tham số và kết quả lên MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("selected_threshold", SELECTED_THRESHOLD)

        # 4. Log Model dựa trên thuật toán
        explicit_reqs = [
            "mlflow",
            "numpy",
            "pandas",
            "scikit-learn",
            "cloudpickle",
            "xgboost",
            "lightgbm",
            "catboost",
        ]

        if best_algo == "LightGBM":
            mlflow.lightgbm.log_model(model, "tuned_model", pip_requirements=explicit_reqs)
        elif best_algo == "XGBoost":
            mlflow.xgboost.log_model(model, "tuned_model", pip_requirements=explicit_reqs)
        else:
            mlflow.sklearn.log_model(model, "tuned_model", pip_requirements=explicit_reqs)

        # 5. Đăng ký vào Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/tuned_model"
        client = MlflowClient()

        try:
            client.create_registered_model(model_name)
        except Exception:
            pass

        mv = None
        try:
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
            client.transition_model_version_stage(name=model_name, version=mv.version, stage="Staging")
        except Exception as e:
            logger.warning(f"Skipping model registry operations: {e}")

        # 6. Lưu file .pkl cục bộ

        save_path = f"{args.models_dir}/{model_name}_final.pkl"
        # joblib.dump(model, save_path)
        joblib.dump(
            {
                "model": model,
                "threshold": SELECTED_THRESHOLD,
                "model_name": best_algo,
                "feature_columns": list(X_train.columns),
            },
            save_path,
        )

        logger.info("-" * 30)
        if mv is not None:
            logger.info(f"DONE: Model {model_name} registered (v{mv.version})")
        else:
            logger.info(f"DONE: Model {model_name} trained and logged without registry version")
        logger.info(f"FILE SAVED: {save_path}")
        logger.info(f"Decision threshold saved: {SELECTED_THRESHOLD}")
        logger.info(f"PR-AUC: {metrics['pr_auc']:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

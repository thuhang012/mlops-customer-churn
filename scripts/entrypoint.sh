#!/bin/bash
set -eo pipefail

echo "============================================================"
echo "[Production Boot] Initializing MLOps Pipeline..."
echo "============================================================"

# Paths setup
DATASET_PATH="data/raw/netflix_large.csv"
MODEL_PATH="artifacts/models/Netflix_Prediction_final.pkl"

echo "[Boot: DVC] Configuring Credentials for DagsHub..."
if [ -z "$DAGSHUB_USERNAME" ] || [ -z "$DAGSHUB_TOKEN" ]; then
    echo "❌ FATAL ERROR: Missing DAGSHUB_USERNAME or DAGSHUB_TOKEN."
    exit 1
fi

# Step 1: DVC Auth Configuration
# Ensure DVC local config uses no-scm mode because .git is ignored in Docker
dvc config core.no_scm true
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USERNAME"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"

echo "[Boot: DVC] Synchronizing Data..."
# Step 2: Data Sync using DVC Pull
if ! dvc pull -r origin -v -f; then
    echo "❌ FATAL ERROR: Failed to pull data from DVC remote 'origin'."
    exit 1
fi

if [ ! -f "$DATASET_PATH" ]; then
    echo "❌ FATAL ERROR: Dataset not found at $DATASET_PATH after pulling."
    exit 1
fi
echo "✅ DVC Sync successful. Clean remote dataset located."

echo "[Boot: MLflow] Connecting to Tracking URI: ${MLFLOW_TRACKING_URI:-Unknown}"

# Step 3: Conditional Training Logic
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️ Target model not found at $MODEL_PATH. Initiating training..."
    
    echo "[Boot: Pipeline] Phase 1 - Preprocessing..."
    if ! python -m src.mlops_project.data.preprocess; then
        echo "❌ FATAL ERROR: Data Preprocessing failed."
        exit 1
    fi
    
    echo "[Boot: Pipeline] Phase 2 - Executing Core Training..."
    if ! python src/mlops_project/models/train.py \
        --config "artifacts/models/best_model_config.yaml" \
        --data "data/processed/cleaned_data.csv" \
        --models-dir "artifacts/models" \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}"; then
        echo "❌ FATAL ERROR: Model Training Process failed."
        exit 1
    fi
    
    # We verify the models have actually been produced
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ FATAL ERROR: Training succeeded but model file $MODEL_PATH is missing."
        exit 1
    fi
    echo "✅ Model successfully generated."
else
    echo "✅ Active Model found at $MODEL_PATH. Bypassing training sequence."
fi

echo "============================================================"
echo "[Production Boot] System Health: OK. Serving API... 🚀"
echo "👉 Swagger UI is available at: http://localhost:8000/docs"
echo "============================================================"

# Step 4: Start Serving
exec uvicorn mlops_project.api.serve:app --host 0.0.0.0 --port 8000

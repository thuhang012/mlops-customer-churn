#!/bin/bash
set -eo pipefail

echo "============================================================"
echo "[Production Boot] Initializing MLOps Pipeline..."
echo "============================================================"

# Paths setup
DVC_FILE="data/raw/telcom_churn.csv.dvc"
MODEL_PATH="artifacts/models/Telco_Churn_Model_final.pkl"

# Phase 1: DVC
echo "[Boot: DVC] Synchronizing Data..."
if [ -z "$DAGSHUB_USERNAME" ] || [ -z "$DAGSHUB_TOKEN" ]; then
    echo "❌ FATAL ERROR: Missing DAGSHUB_USERNAME or DAGSHUB_TOKEN."
    exit 1
fi

dvc config core.no_scm true
dvc config core.analytics false
export DVC_NO_ANALYTICS=true
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USERNAME"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"

if ! dvc pull "$DVC_FILE" -v -f; then
    echo "❌ FATAL ERROR: DVC pull failed for $DVC_FILE."
    exit 1
fi
echo "✅ DVC Sync successful."

# Phase 2: Conditional Training Logic
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Active Model found at $MODEL_PATH. Bypassing training sequence."
else
    echo "⚠️ Target model not found at $MODEL_PATH. Initiating training..."
    
    echo "[Boot: Pipeline] Phase 2 - Preprocessing..."
    # Ensure it generates the 3 required outputs
    if ! python -m src.mlops_project.data.preprocess; then
        echo "❌ FATAL ERROR: Data Preprocessing failed."
        exit 1
    fi
    
    echo "[Boot: Pipeline] Phase 2 - Executing Core Training..."
    mkdir -p artifacts/models
    if ! python src/mlops_project/models/train.py \
        --config "configs/best_model.yaml" \
        --data "data/processed/cleaned_data_tree.csv" \
        --models-dir "artifacts/models" \
        --mlflow-tracking-uri "${MLFLOW_TRACKING_URI}"; then
        echo "❌ FATAL ERROR: Model Training Process failed."
        exit 1
    fi
    
    # We verify the model has actually been produced
    if [ ! -f "$MODEL_PATH" ]; then
        echo "❌ FATAL ERROR: Training succeeded but model file $MODEL_PATH is missing."
        exit 1
    fi
    echo "✅ Model successfully generated."
fi

echo "============================================================"
echo "[Production Boot] System Health: OK. Serving API... 🚀"
echo "============================================================"

# Phase 4: Serving
if [ "$#" -gt 0 ]; then
    echo "[Boot: Serving] Starting custom command: $*"
    exec "$@"
fi

if [ "$APP_ROLE" = "ui" ]; then
    echo "[Boot: Serving] Starting Streamlit UI..."
    exec streamlit run streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
else
    echo "[Boot: Serving] Starting FastAPI API..."
    exec uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000
fi

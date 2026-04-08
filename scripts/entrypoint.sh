#!/bin/bash
set -e

# ============================================================
# M4 - Entrypoint Bootstrap Script
# Mục đích: Đảm bảo API có đủ file model trước khi khởi chạy.
# Nếu file .pkl chưa tồn tại → tự động train từ mock data.
# ============================================================

MODEL_PATH="artifacts/models/Netflix_Prediction_final.pkl"
PREPROCESSOR_PATH="artifacts/preprocessors/preprocessor.pkl"
MOCK_DATA_PATH="data/raw/mock_data.csv"
CONFIG_PATH="artifacts/models/best_model_config.yaml"
CLEANED_DATA_PATH="data/processed/cleaned_data.csv"
MODELS_DIR="artifacts/models"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://mlflow:5000}"

# ----------------------------------------------------------
# Bước 1: Kiểm tra file model đã tồn tại chưa
# ----------------------------------------------------------
if [ -f "$MODEL_PATH" ] && [ -f "$PREPROCESSOR_PATH" ]; then
    echo "[M4-Bootstrap] Model và Preprocessor đã sẵn sàng. Bỏ qua bước train."
else
    echo "[M4-Bootstrap] Thiếu file model. Bắt đầu tự động khởi tạo..."

    # ----------------------------------------------------------
    # Bước 2: Đợi MLflow server sẵn sàng (tối đa 60 giây)
    # Sử dụng kiểm tra TCP socket (không phụ thuộc API endpoint)
    # ----------------------------------------------------------
    echo "[M4-Bootstrap] Đang chờ MLflow server..."

    # Trích xuất host và port từ URI
    MLFLOW_HOST=$(echo "$MLFLOW_URI" | sed -E 's|https?://||' | cut -d: -f1)
    MLFLOW_PORT=$(echo "$MLFLOW_URI" | sed -E 's|https?://||' | cut -d: -f2 | cut -d/ -f1)
    MLFLOW_PORT=${MLFLOW_PORT:-5000}

    RETRIES=30
    WAIT_SECONDS=2
    MLFLOW_READY=false

    for i in $(seq 1 $RETRIES); do
        if python -c "
import socket, sys
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(2)
try:
    s.connect(('$MLFLOW_HOST', $MLFLOW_PORT))
    s.close()
    sys.exit(0)
except:
    s.close()
    sys.exit(1)
" 2>/dev/null; then
            echo "[M4-Bootstrap] MLflow đã sẵn sàng tại $MLFLOW_URI!"
            MLFLOW_READY=true
            break
        fi
        echo "[M4-Bootstrap] MLflow chưa phản hồi, thử lại lần $i/$RETRIES..."
        sleep $WAIT_SECONDS
    done

    if [ "$MLFLOW_READY" = false ]; then
        echo "[M4-Bootstrap] CẢNH BÁO: MLflow không phản hồi sau ${RETRIES} lần thử."
        echo "[M4-Bootstrap] Tiếp tục train ở chế độ offline (local SQLite)..."
        MLFLOW_URI="sqlite:///mlflow_local.db"
    fi

    # ----------------------------------------------------------
    # Bước 3: Chạy tiền xử lý dữ liệu (Preprocess)
    # ----------------------------------------------------------
    if [ ! -f "$CLEANED_DATA_PATH" ]; then
        echo "[M4-Bootstrap] Đang chạy tiền xử lý dữ liệu..."
        if ! python -m src.mlops_project.data.preprocess --input "$MOCK_DATA_PATH" 2>&1; then
            echo "[M4-Bootstrap] LỖI: Tiền xử lý thất bại."
            echo "[M4-Bootstrap] Vui lòng đảm bảo file dữ liệu tồn tại tại: $MOCK_DATA_PATH"
            exit 1
        fi
        echo "[M4-Bootstrap] Tiền xử lý hoàn tất."
    else
        echo "[M4-Bootstrap] Dữ liệu processed đã tồn tại. Bỏ qua."
    fi

    # ----------------------------------------------------------
    # Bước 4: Train model (có xử lý lỗi)
    # ----------------------------------------------------------
    echo "[M4-Bootstrap] Đang train model..."
    if python src/mlops_project/models/train.py \
        --config "$CONFIG_PATH" \
        --data "$CLEANED_DATA_PATH" \
        --models-dir "$MODELS_DIR" \
        --mlflow-tracking-uri "$MLFLOW_URI" 2>&1; then
        echo "[M4-Bootstrap] Train model hoàn tất!"
    else
        echo "[M4-Bootstrap] CẢNH BÁO: Training pipeline gặp lỗi (có thể do dữ liệu mock quá nhỏ)."
        echo "[M4-Bootstrap] Tạo model placeholder để API có thể khởi động..."

        # Tạo model + preprocessor khớp nhau bằng Python thuần
        python -c "
import joblib, os
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

os.makedirs('artifacts/models', exist_ok=True)
os.makedirs('artifacts/preprocessors', exist_ok=True)

# Đọc số features thực tế từ preprocessor đã tạo (Bước 3 đã chạy)
n_features = 10  # fallback
feature_columns = []
if os.path.exists('$PREPROCESSOR_PATH'):
    artifact = joblib.load('$PREPROCESSOR_PATH')
    feature_columns = artifact.get('feature_columns', [])
    n_features = len(feature_columns) if feature_columns else 10
elif os.path.exists('$CLEANED_DATA_PATH'):
    df = pd.read_csv('$CLEANED_DATA_PATH')
    exclude = ['churn_status', 'data_split']
    feature_columns = [c for c in df.columns if c not in exclude]
    n_features = len(feature_columns)

print(f'[M4-Bootstrap] Tạo placeholder model với {n_features} features.')

# Tạo dummy data đúng số chiều để model khớp với preprocessor
X_dummy = np.zeros((4, n_features))
X_dummy[1, :] = 1
X_dummy[3, :] = 1
y_dummy = np.array([0, 1, 0, 1])

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_dummy, y_dummy)
joblib.dump(model, '$MODEL_PATH')

print('[M4-Bootstrap] Model placeholder đã được tạo thành công.')
print('[M4-Bootstrap] LƯU Ý: Đây là model tạm. Hãy train lại với dữ liệu thật để có kết quả chính xác.')
"
    fi

    # Kiểm tra lần cuối
    if [ ! -f "$MODEL_PATH" ] || [ ! -f "$PREPROCESSOR_PATH" ]; then
        echo "[M4-Bootstrap] LỖI NGHIÊM TRỌNG: Không thể tạo file model."
        echo "[M4-Bootstrap] Vui lòng liên hệ M2 để cung cấp file:"
        echo "  - $MODEL_PATH"
        echo "  - $PREPROCESSOR_PATH"
        exit 1
    fi
fi

# ----------------------------------------------------------
# Bước 5: Khởi chạy API Server
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo "[M4-Bootstrap] API IS READY! 🚀"
echo "[M4-Bootstrap] Local API Docs: http://localhost:8000/docs"
echo "[M4-Bootstrap] MLflow UI:       http://localhost:5555"
echo "============================================================"
echo ""

exec uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000

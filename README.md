# MLOpsProject

End-to-end MLOps learning project for churn prediction, built with Agile/Kanban collaboration.

Current status:

- Sprint 1 foundation is available (mock API, CI scaffold, DVC tracking).
- Some production modules are still placeholders and will be completed in later sprints.

---

## 1) Project Goals

- Build a reproducible MLOps workflow from data to serving and monitoring.
- Keep quality gates active from day 1 (lint + tests via CI).
- Enable concurrent engineering (team members can build in parallel using contracts and mocks).

Primary stack:

- Python 3.10+
- FastAPI (serving)
- DVC + DagsHub (data versioning)
- Docker / Docker Compose (packaging and runtime)
- GitHub Actions (CI)

---

## 2) Repository Structure

```text
MLOpsProject/
├── .github/workflows/
│   └── ci.yml
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
├── src/mlops_project/
│   ├── api/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── monitoring/
│   └── utils/
├── tests/
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml
├── pyproject.toml
├── requirements.txt
├── README_DVC.md
└── README_M4_DOCKER.md
```

---

## 3) Quick Start: Complete Setup Guide

Hướng dẫn chi tiết này sẽ giúp bạn setup toàn bộ dự án từ A-Z.

### 3.1 Clone Repository

```powershell
git clone https://github.com/thuhang012/mlops-customer-churn.git
cd mlops-customer-churn
```

### 3.2 Create Virtual Environment (Optional for Local Development)

**Note:** Nếu Python 3.12 của bạn gặp lỗi SSL, hãy bỏ qua bước này và chạy Docker trực tiếp (Docker có Python 3.11 riêng).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.3 Install Dependencies (Local Only)

Chỉ cần nếu bạn chạy local, không cần cho Docker:

```powershell
pip install -r requirements.txt
```

### 3.4 Configure DVC Credentials for Docker

**QUAN TRỌNG:** Docker sẽ tự động pull data từ DagsHub. Bạn cần tạo file `.env`:

```powershell
# Copy .env.example -> .env
Copy-Item .env.example .env
```

Mở file `.env` và kiểm tra (hoặc chỉnh sửa) credentials:

```dotenv
DAGSHUB_USER=bich-le
DAGSHUB_TOKEN=42faca3ef5d1242b9f72c7f5be0f09e97acd4c82
```

**Lưu ý bảo mật:** 
- Token này đã được expose trong `.env.example` - nên rotate token sau khi setup.
- Đừng commit `.env` vào Git, chỉ commit `.env.example`.

---

## 4) Run with Docker Compose (Recommended)

Docker Compose sẽ:
1. Tự động pull data từ DagsHub (dùng credentials từ `.env`)
2. Tự động train model (nếu chưa có model file)
3. Khởi động MLflow Tracking Server
4. Khởi động FastAPI Server

### 4.1 Build & Start Services

```powershell
docker-compose up -d --build
```

**Output mong đợi:**
```
✔ Image mlops-customer-churn-api Built
✔ Container mlflow_tracker       Healthy
✔ Container churn_api_container  Started
```

### 4.2 Xác Nhận Services Đang Chạy

```powershell
docker-compose ps
```

Hoặc kiểm tra health check:

```powershell
curl http://127.0.0.1:8000/health
```

### 4.3 Truy Cập Các Service

Sau khi Docker chạy, bạn có thể truy cập:

| Service | URL | Mô Tả |
|---------|-----|-------|
| FastAPI Swagger UI | http://localhost:8000/docs | API documentation & testing |
| FastAPI Health Check | http://localhost:8000/health | Kiểm tra server status |
| MLflow Tracking Server | http://localhost:5555 | Xem model metrics, parameters, artifacts |

### 4.4 Test API Prediction

```powershell
# Predict endpoint
$body = @{
    tenure = 5
    monthly_charges = 90
    contract_type = "monthly"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

---

## 5) Local Development (Không dùng Docker)

### 5.1 Activate venv & Install Dependencies

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
dvc[s3] dagshub
```

### 5.2 Configure DVC Credentials

```powershell
dvc remote modify origin --local auth basic
dvc remote modify origin --local user bich-le
dvc remote modify origin --local password 42faca3ef5d1242b9f72c7f5be0f09e97acd4c82
```

### 5.3 Pull Data from DVC

```powershell
dvc pull -r origin
```

Sau lệnh này, file `data/raw/netflix_large.csv` sẽ xuất hiện.

### 5.4 Run Tests Locally

```powershell
python -m ruff check src tests --select E,F,W
python -m pytest tests/
```

### 5.5 Run API Locally

```powershell
uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000
```

---

## 6) Stop & Cleanup Docker

### Stop Services

```powershell
docker-compose down
```

### Complete Reset (Xóa Database)

```powershell
docker-compose down -v
```

---

## 7) Troubleshooting

### Docker Container Exiting

Kiểm tra logs:

```powershell
docker-compose logs -f churn_api_container
docker-compose logs -f mlflow_tracker
```

### DVC Pull Failed

Kiểm tra credentials trong `.env`:

```powershell
docker-compose down
# Sửa .env
docker-compose up -d --build
```

### Port Already in Use

Nếu port 8000 hoặc 5555 bị dùng, chỉnh sửa `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Đổi từ 8000 sang 8001
```

---

## 8) CI Overview

### CI: .github/workflows/ci.yml

Triggers: push to main, pull_request to main

Steps:
1. Checkout code
2. Set up Python 3.10
3. Install runtime & dev dependencies
4. Run Ruff lint checks
5. Run pytest

## 9) Team Roles (M1-M6)

- M1: data pipeline and DVC
- M2: training and MLflow
- M3: API and serving contract
- M4: Docker and infrastructure
- M5: CI quality gates and automation
- M6: monitoring and drift reporting

---

## 10) DVC Workflow for Team

Khi dataset thay đổi:

```powershell
dvc add data/raw/your_data.csv
dvc push -r origin
git add data/raw/your_data.csv.dvc .gitignore
git commit -m "Update dataset"
git push
```

**Important:**
- Never git add large raw data files directly.
- Commit .dvc pointers, not heavy artifacts.

---

## 11) Security Notes

- Do not hardcode secrets or tokens in source code.
- Keep credentials in `.env` or CI secret managers.
- Rotate tokens immediately if exposed.
- Never commit `.env` to Git.

---

## 12) Known Gaps and Next Steps

- dvc.yaml pipeline stages not finalized yet
- Model training/evaluation scripts in progress
- Monitoring flow and retraining trigger pending
- Cloud deployment target pending

---

## 13) Useful Files

- [README_DVC.md](README_DVC.md): DVC setup & data versioning guide
- [README_M4_DOCKER.md](README_M4_DOCKER.md): Docker infrastructure & deployment SOP
- [.github/workflows/ci.yml](.github/workflows/ci.yml): CI checks

# MLOpsProject

End-to-end MLOps learning project for churn prediction, built with Agile/Kanban collaboration.

Current status:

- Sprint 1 foundation is available (mock API, CI, CD scaffold, DVC tracking).
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
- GitHub Actions (CI/CD)

---

## 2) Repository Structure

```text
MLOpsProject/
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
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

## 3) Quick Start (Local)

### 3.1 Create and activate virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3.2 Install dependencies

```powershell
pip install -r requirements.txt
```

### 3.3 Pull data from DVC remote

See full setup in README_DVC.md. Minimal command:

```powershell
dvc pull -r origin
```

Expected file after success:

- data/raw/netflix_large.csv

---

## 4) Run API

Current API is a Sprint 1 mock service.

### Run locally

```powershell
uvicorn src.mlops_project.api.serve:app --host 0.0.0.0 --port 8000
```

### Endpoints

- GET / -> service message
- GET /health -> health check
- POST /predict -> mock churn probability

Sample request:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body '{"tenure":5,"monthly_charges":90,"contract_type":"monthly"}'
```

---

## 5) Testing and Quality Gates

### Local checks

```powershell
python -m ruff check src tests --select E,F,W
python -m pytest tests/
```

### Existing tests

- tests/test_api.py: health endpoint smoke test
- tests/test_data.py: raw data existence + required columns
- tests/test_model.py: placeholder test (to be replaced in later sprint)

Ruff and pytest config are defined in pyproject.toml.

---

## 6) CI/CD Overview

### CI: .github/workflows/ci.yml

Triggers:

- push to main
- pull_request to main

Steps:

1. Checkout code
2. Set up Python 3.10
3. Install runtime and dev dependencies
4. Run Ruff lint checks
5. Run pytest

### CD (Scaffold): .github/workflows/cd.yml

Current flow:

1. Build Docker image
2. Run container and smoke-test /health
3. Write deployment summary in GitHub Actions

Note:

- Offline-first: smoke-test only, no image push to GHCR.
- Docker is for local validation; CD does not require online registry.

---

## 7) Docker Usage

### Build and run with Docker Compose

```powershell
docker compose up -d --build
```

Check health:

```powershell
curl http://127.0.0.1:8000/health
```

Stop services:

```powershell
docker compose down
```

---

## 8) DVC Workflow (Team)

When dataset changes:

```powershell
dvc add data/raw/your_data.csv
dvc push -r origin
git add data/raw/your_data.csv.dvc .gitignore
git commit -m "Update dataset"
git push
```

Important:

- Never git add large raw data files directly.
- Commit .dvc pointers, not heavy artifacts.

---

## 9) Team Roles (M1-M6)

- M1: data pipeline and DVC
- M2: training and MLflow
- M3: API and serving contract
- M4: Docker and infrastructure
- M5: CI/CD quality gates and automation
- M6: monitoring and drift reporting

Sprint progress notes are maintained as local working notes and are not part of the shared repository documentation.

---

## 10) Known Gaps and Next Steps

- dvc.yaml pipeline stages are not finalized yet.
- Model training/evaluation scripts are still in progress.
- Monitoring flow and retraining trigger are pending.
- Cloud deployment target is pending final integration.

---

## 11) Security Notes

- Do not hardcode secrets or tokens in source code.
- Keep credentials in local config or CI secret managers.
- Rotate tokens immediately if exposed in shared docs/history.

---

## 12) Useful Files

- README_DVC.md: data setup and DVC usage
- README_DVC.md: DVC setup and data versioning guide
- .github/workflows/ci.yml: CI checks
- .github/workflows/cd.yml: CD scaffold

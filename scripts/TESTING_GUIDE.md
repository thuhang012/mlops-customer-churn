# Testing Guide

This guide explains how to run tests locally. All commands below assume you've activated your virtual environment.

## Quick Start

```powershell
# Install dependencies
pip install -r requirements.txt

# Run fast tests (CI equivalent)
python -m pytest -m fast tests/

# Run all tests including model quality (integration tests)
python -m pytest tests/
```

## Test Categories

### 1. **Fast Tests (CI)** - Lint + Quick Unit Tests

These tests run in CI on every push and should be fast (<1 minute).

```powershell
# Lint with Ruff
python -m ruff check src tests --select E,F

# Run fast tests only (excludes heavy integration tests)
python -m pytest -m fast tests/ -v
```

**Expected Result:** 7 passed, 7 deselected (fast tests only)

---

### 2. **Integration Tests (CT)** - Continuous Training Tests

These tests use real data and real model artifacts. They're slower and reserved for CT workflow.

```powershell
# Run CT model quality tests only
python -m pytest -m ct tests/test_model_quality.py -v
```

**Note:** CT tests require:
- Real dataset at `data/raw/netflix_large.csv`
- Trained model at `artifacts/models/Netflix_Prediction_final.pkl`
- These are populated by CT workflow or manual training

---

## Individual Test Modules

### Test Data Validation
```powershell
python scripts/validation/validate_data.py data/raw/netflix_large.csv
```

### Test API Health
```powershell
python -m pytest tests/test_api.py -v
```

### Test Data Structure
```powershell
python -m pytest tests/test_data.py -v
```

### Test Model Quality (Integration)
```powershell
python -m pytest tests/test_model_quality.py -v
```

---

## Full CI Simulation (Local)

To run the full CI pipeline locally:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Lint
python -m ruff check src tests --select E,F

# 3. Run fast tests
python -m pytest -m fast tests/ -v

# Expected: all pass, quick execution (<1 min)
```

---

## Full CT Simulation (With Real Data)

To run the CT pipeline locally (requires DVC credentials):

```powershell
# 1. Set DVC credentials
$env:DAGSHUB_USERNAME = "your-username"
$env:DAGSHUB_TOKEN = "your-token"

# 2. Configure DVC
dvc remote modify --local origin auth basic
dvc remote modify --local origin user $env:DAGSHUB_USERNAME
dvc remote modify --local origin password $env:DAGSHUB_TOKEN

# 3. Pull real data
dvc pull data/raw/netflix_large.csv.dvc

# 4. Validate data
python scripts/validation/validate_data.py data/raw/netflix_large.csv

# 5. Train model (if needed)
python src/mlops_project/models/train.py \
  --config artifacts/models/best_model_config.yaml \
  --data data/processed/cleaned_data.csv \
  --models-dir artifacts/models

# 6. Run all tests including CT
python -m pytest tests/ -v
```

---

## Pytest Markers

Tests are organized using pytest markers:

| Marker | Purpose | Runs In |
|--------|---------|---------|
| `fast` | Fast unit tests and API/data checks | CI fast path |
| `ct` | Continuous Training integration tests | CT workflow only |

### Run by Marker
```powershell
# Fast tests only
python -m pytest -m fast tests/

# Integration tests only
python -m pytest -m ct tests/test_model_quality.py

# All tests
python -m pytest tests/
```

---

## Pytest Options

```powershell
# Run with verbose output
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test file
python -m pytest tests/test_api.py -v

# Run specific test function
python -m pytest tests/test_api.py::test_api_health_reports_ok_and_loaded_artifacts -v

# Stop on first failure
python -m pytest tests/ -x

# Run only failed tests from last run
python -m pytest tests/ --lf
```

---

## Troubleshooting

### **Tests fail with "Model artifact is not fitted"**
- This is expected: the model file in the repo is intentionally unfitted for learning purposes
- CT tests will fail until you train a real model via the CT workflow
- CI fast tests skip this check and pass normally

### **DVC pull fails**
- Ensure `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are set
- Check that you have access to the DagsHub repository

### **Module not found errors**
- Activate your virtual environment: `.venv\Scripts\Activate.ps1`
- Reinstall dependencies: `pip install -r requirements.txt`

---

## Docker Testing

To test the full Docker setup:

```powershell
# Build and run with docker-compose
docker compose up -d --build

# Check health
curl http://127.0.0.1:8000/health

# View logs
docker logs churn_api_container

# Stop
docker compose down
```

---

## CI/CD Workflow Testing

### **Test CI locally**
```powershell
python -m ruff check src tests --select E,F
python -m pytest -m fast tests/ -v
```

### **Test CD locally**
```powershell
docker build -t mlops-project-api:test .
docker run -d --name test-api -p 8000:8000 -e CI_SMOKE_MODE=true mlops-project-api:test
sleep 3
curl http://127.0.0.1:8000/health
docker rm -f test-api
```

### **Test CT locally**
Requires real DVC data and working model. See "Full CT Simulation" above.

---

## Notes

- **CI tests** are lightweight and should pass on every branch
- **CT tests** are integration-heavy and only run on `main` branch in GitHub Actions
- **CD tests** are smoke tests that validate the Docker container can start
- **Monitoring workflow** is separate and runs on schedule; not part of core CI/CD/CT

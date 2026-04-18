# Testing Guide

This guide explains how to run tests locally. All commands below assume you've activated your virtual environment.

## Quick Start

### Step 0: Use Python 3.11 (IMPORTANT!)
```powershell
# Check your Python version
python --version

# If you see Python 3.12+, you MUST use Python 3.11!
# Either:
# 1. Install Python 3.11 from https://www.python.org/downloads/
# 2. Or use: C:\Python311\python.exe -m venv .venv
```

### Step 1: Create & Activate Virtual Environment
```powershell
# Create venv if it doesn't exist
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install All Dependencies (REQUIRED!)
```powershell
# Install all project dependencies including pytest, ruff, etc.
pip install -r requirements.txt
```

**What gets installed:**
- Test framework: `pytest`, `pytest-cov`
- Linting: `ruff`, `bandit`, `pip-audit`
- ML libraries: `scikit-learn`, `pandas`, `numpy`, `xgboost`, `catboost`, `lightgbm`
- API: `fastapi`, `uvicorn`
- Tracking: `mlflow`, `dvc`, `dagshub`

### Step 3: Run Tests
```powershell
# Run fast tests (CI equivalent)
python -m pytest -m fast tests/

# Run all tests including model quality (integration tests)
python -m pytest tests/
```

---

## Fixing venv Issues

### **Python Version Error: "module 'ssl' has no attribute 'SSLSession'"**
This means you're using **Python 3.12+** but the project requires **Python 3.11**.

**Solution:**
```powershell
# 1. Install Python 3.11 from https://www.python.org/downloads/

# 2. Delete current venv
Remove-Item .venv -Recurse -Force

# 3. Create venv with Python 3.11 explicitly
C:\Python311\python.exe -m venv .venv

# 4. Activate it
.\.venv\Scripts\Activate.ps1

# 5. INSTALL dependencies
pip install -r requirements.txt
```

⚠️ **If you don't have Python 3.11 installed:**
- Download from https://www.python.org/downloads/release/python-3119/ (or latest 3.11.x)
- During install, check "Add Python to PATH"
- Then follow steps 2-5 above

### **"No module named pytest" after installing**
Make sure you:
1. ✅ Used **Python 3.11** (not 3.12+)
2. ✅ Created venv with that Python version
3. ✅ Activated the venv
4. ✅ Ran `pip install -r requirements.txt`

Verify:
```powershell
# Should show venv Python 3.11.x
python --version

# Should show pytest installed
python -m pytest --version
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

**Expected Result:** 9 passed, 7 deselected (fast tests only)

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

### Test API Health & Predictions (6 tests, fast)
```powershell
python -m pytest tests/test_api.py -v
```
Tests:
- API root endpoint returns welcome message
- API health check reports OK status and loaded artifacts
- Single prediction respects model fit state
- Batch prediction respects model fit state
- API rejects missing required fields
- API rejects invalid field types

### Test Data Structure (1 test, fast)
```powershell
python -m pytest tests/test_data.py -v
```
Tests:
- Raw CSV contains all required columns

### Test Monitoring (2 tests, fast)
```powershell
python -m pytest tests/test_monitoring.py -v
```
Tests:
- Monitoring detects drift and degradation when thresholds exceeded
- Monitoring skips checks gracefully when input files missing

### Test Model Quality (7 tests, CT integration only)
```powershell
python -m pytest tests/test_model_quality.py -v
```
Tests:
- Model, preprocessor, and raw dataset exist on disk
- Preprocessor columns compatible with engineered features
- Saved estimator is fitted for inference
- Predicted probabilities within unit interval [0, 1]
- Metrics meet baseline quality gate
- Classifier predictions identical on repeated forward pass
- ROC-AUC within valid numeric bounds

**⚠️ Note:** CT tests require real artifacts (trained model, preprocessor, dataset)

---

## Full CI Simulation (Local)

To run the full CI pipeline locally:

```powershell
# 1. Create and activate venv (if not already)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies (CRITICAL!)
pip install -r requirements.txt

# 3. Lint
python -m ruff check src tests --select E,F

# 4. Run fast tests
python -m pytest -m fast tests/ -v

# Expected: all pass, quick execution (<1 min)
```

**Expected Output:**
- **Ruff:** ❌ 38 E501 errors (line-too-long) - These will fail CI on GitHub!
- **Pytest:** 9 passed, 7 deselected

⚠️ **CRITICAL:** The E501 errors must be fixed before CI passes! See "Fixing E501 Linting Errors" below.

---

## Fixing E501 Linting Errors

Currently **38 E501 errors** (lines longer than 88 characters) exist in the codebase:

**Affected Files:**
- `src/mlops_project/api/service.py` - 2 errors
- `src/mlops_project/models/evaluate.py` - 6 errors  
- `src/mlops_project/models/train.py` - 8 errors
- `src/mlops_project/monitoring/drift_detector.py` - 22 errors

**How to fix manually:**
1. Break long lines into multiple lines
2. Split long f-strings across lines  
3. Use parentheses for line continuation

**Example:**
```python
# ❌ Before (too long)
raise ValueError(f"Error message with {variable}. Another message: {list(data.keys())}")

# ✅ After (proper)
raise ValueError(
    f"Error message with {variable}. "
    f"Another message: {list(data.keys())}"
)
```

**Auto-fix attempt:**
```powershell
python -m ruff check src --select E501 --fix
```

---

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

| Marker | Count | Purpose | Runs In |
|--------|-------|---------|---------|
| `fast` | 9 tests | Fast unit tests (API, data, monitoring) | CI fast path |
| `ct` | 7 tests | Integration tests requiring real artifacts, data, model | CT workflow only |
| **(total)** | **16 tests** | All tests | Local development |

### Test File Mapping

| File | Tests | Marker | Purpose |
|------|-------|--------|---------|
| test_api.py | 6 | fast | API health, predictions, validation |
| test_data.py | 1 | fast | Raw data structure validation |
| test_monitoring.py | 2 | fast | Drift/degradation detection |
| test_model_quality.py | 7 | ct | Model fit, performance, reproducibility |
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

### **ERROR: AttributeError: module 'ssl' has no attribute 'SSLSession'**
```
AttributeError: module 'ssl' has no attribute 'SSLSession'
```
**Cause:** You're using Python 3.12+, but the project requires **Python 3.11**

**Solution:**
1. Install Python 3.11 from https://www.python.org/downloads/release/python-3119/
2. Delete venv: `Remove-Item .venv -Recurse -Force`
3. Create venv with Python 3.11: `C:\Python311\python.exe -m venv .venv`
4. Activate: `.\.venv\Scripts\Activate.ps1`
5. Install: `pip install -r requirements.txt`

### **"No module named pytest" error**
```
D:\...\python.exe: No module named pytest
```
**Solution:** You skipped installing dependencies OR using wrong Python version!
```powershell
# 1. Check Python version (must be 3.11!)
python --version

# 2. Make sure venv is activated
.\.venv\Scripts\Activate.ps1

# 3. Install requirements
pip install -r requirements.txt

# 4. Verify pytest works
python -m pytest --version
```

### **"Module not found" for sklearn, pandas, etc.**
Same solution - you need to:
1. ✅ Use **Python 3.11** (not 3.12+)
2. ✅ Run `pip install -r requirements.txt`

### **Tests fail with "Model artifact is not fitted"**
- This is expected: the model file in the repo is intentionally unfitted for learning purposes
- CT tests will fail until you train a real model via the CT workflow
- CI fast tests skip this check and pass normally

### **DVC pull fails**
- Ensure `DAGSHUB_USERNAME` and `DAGSHUB_TOKEN` are set
- Check that you have access to the DagsHub repository

---

## Docker Testing

To test the full Docker setup:

```powershell
# Build and run with docker-compose
docker compose up -d --build

# Check health
curl http://127.0.0.1:8000/health

# Run linting in container
docker-compose exec -T api python -m ruff check src --select E,F

# Run fast tests in container (note: tests/ not in production image)
docker-compose exec -T api python -m pytest -m fast tests/  # ❌ Will fail (tests not in image)

# View logs
docker logs api  # or docker logs churn_api_container

# Stop
docker compose down
```

⚠️ **Note:** Production Docker image does NOT include tests/ directory. Use local venv for testing.

---

## CI/CT Workflow Testing

### **Test CI locally**
```powershell
python -m ruff check src tests --select E,F
python -m pytest -m fast tests/ -v
```

### **Test CT locally**
Requires real DVC data and working model. See "Full CT Simulation" above.

---

## Notes

- **CI tests** are lightweight and should pass on every branch
- **CT tests** are integration-heavy and only run on `main` branch in GitHub Actions
- **Monitoring workflow** is separate and runs on schedule; not part of core CI/CT

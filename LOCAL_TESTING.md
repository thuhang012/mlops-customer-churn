# 🧪 Local Testing Guide

Complete guide to test the CI/CD pipeline locally before pushing to GitHub.

## ⚙️ Setup

### 1. Install Requirements

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 2. Create Necessary Directories

```bash
mkdir -p artifacts/models
mkdir -p artifacts/metrics
mkdir -p artifacts/baseline
mkdir -p reports/monitoring
mkdir -p logs
```

---

## 🚀 Testing Individual Components

### Option A: Using Make (Recommended)

```bash
# View all available targets
make help

# Test individual components
make test-data
make test-train
make test-quality
make test-monitor

# Run all tests
make test-all

# Clean artifacts
make clean
```

### Option B: Running Scripts Directly

#### 1️⃣ Data Validation

```bash
python scripts/validation/validate_data.py data/raw/netflix_large.csv
```

**Expected output:**
```
✅ Data validation passed
- File: data/raw/netflix_large.csv
- Rows: 10000
- Columns: 15
```

---

#### 2️⃣ Model Training

```bash
python scripts/training/train_model.py artifacts
```

**Expected output:**
```
🎯 Training model...
✅ Training completed
- Model saved: artifacts/models/model.pkl
- F1 Score: 0.87
- ROC-AUC: 0.92
- Runtime: 45 seconds
```

**Generated files:**
- `artifacts/models/model.pkl` - Serialized model
- `artifacts/metrics/metrics.json` - Training metrics

---

#### 3️⃣ Quality Gate Check

```bash
python scripts/validation/quality_gate.py \
  artifacts/metrics/metrics.json \
  artifacts/baseline/metrics.json \
  0.95
```

**Expected output (first run):**
```
⚠️  Baseline not found, creating from current metrics
✅ Quality gate passed (baseline created)
```

**Expected output (subsequent runs):**
```
✅ Quality gate passed
- Current F1: 0.87 (Baseline: 0.85, Δ +2.4%)
- Current AUC: 0.92 (Baseline: 0.90, Δ +2.2%)
```

---

#### 4️⃣ Model Registration

```bash
python scripts/training/register_model.py \
  artifacts/metrics/metrics.json \
  artifacts/baseline/metrics.json
```

**Expected output:**
```
✅ Model registered
- Baseline saved: artifacts/baseline/metrics.json
- F1 Score: 0.87
- ROC-AUC: 0.92
- Timestamp: 2026-04-12T10:30:00Z
```

---

#### 5️⃣ Monitoring Checks

```bash
python scripts/monitoring/checks.py
```

**Expected output:**
```
🔍 Running monitoring checks...
==================================================
📊 Monitoring Results (2026-04-12T10:35:00)
--------------------------------------------------
Drift Detected: ✅ NO
Degradation: ✅ NO

Detailed Checks:
  - feature_drift: ✅ OK
  - performance_degradation: ✅ OK
  - data_quality: ✅ OK
==================================================
✅ Monitoring checks completed successfully
```

---

## 🔄 Testing Full Pipeline Simulation

### Complete CT Pipeline (Local)

```bash
#!/bin/bash
set -e  # Exit on error

echo "🚀 Starting CT Pipeline Simulation..."
echo ""

# Step 1: Data Validation
echo "1️⃣  Data Validation"
python scripts/validation/validate_data.py data/raw/netflix_large.csv
echo ""

# Step 2: Training
echo "2️⃣  Model Training"
python scripts/training/train_model.py artifacts
echo ""

# Step 3: Quality Gate
echo "3️⃣  Quality Gate"
python scripts/validation/quality_gate.py \
  artifacts/metrics/metrics.json \
  artifacts/baseline/metrics.json \
  0.95
QUALITY_PASSED=$?
echo ""

# Step 4: Registration (only if quality passed)
if [ $QUALITY_PASSED -eq 0 ]; then
  echo "4️⃣  Model Registration"
  python scripts/training/register_model.py \
    artifacts/metrics/metrics.json \
    artifacts/baseline/metrics.json
  echo ""
  echo "✅ CT Pipeline completed successfully!"
else
  echo "❌ Quality gate failed, skipping registration"
  exit 1
fi
```

Save as `test_ct_locally.sh` and run:

```bash
chmod +x test_ct_locally.sh
./test_ct_locally.sh
```

---

## 🧬 Testing Model Quality Tests

```bash
# Run all model quality tests
pytest tests/test_model_quality.py -v

# Run specific test class
pytest tests/test_model_quality.py::TestModelTraining -v

# Run with coverage
pytest tests/test_model_quality.py --cov=src/mlops_project --cov-report=html
```

---

## 📊 Checking Outputs

After running tests, check generated files:

```bash
# View metrics
cat artifacts/metrics/metrics.json

# View baseline (created after registration)
cat artifacts/baseline/metrics.json

# View monitoring results
cat reports/monitoring/latest_check.json

# View all artifacts
find artifacts -type f -name "*.json" -o -name "*.pkl"
```

---

## 🔐 Testing with Environment Variables

For GitHub Actions variable output testing:

```bash
# Create a dummy GITHUB_OUTPUT file
export GITHUB_OUTPUT=$(mktemp)

# Run script (will write variables to file)
python scripts/training/train_model.py artifacts

# Check outputs
cat $GITHUB_OUTPUT

# Clean up
rm $GITHUB_OUTPUT
```

---

## 🐛 Debugging Failed Tests

### If data validation fails:
```bash
# Check if data file exists
ls -la data/raw/netflix_large.csv

# Check file size
du -h data/raw/netflix_large.csv

# Check data format
head data/raw/netflix_large.csv
```

### If training fails:
```bash
# Check dependencies
pip list | grep scikit-learn

# Test import
python -c "import sklearn; print(sklearn.__version__)"

# Check artifacts directory
ls -la artifacts/
```

### If quality gate fails:
```bash
# Check baseline exists
ls -la artifacts/baseline/metrics.json

# Compare metrics
python -c "
import json
with open('artifacts/metrics/metrics.json') as f: current = json.load(f)
with open('artifacts/baseline/metrics.json') as f: baseline = json.load(f)
print('Current:', current)
print('Baseline:', baseline)
"
```

---

## ✅ Checklist Before Pushing

- [ ] `make test-data` passes
- [ ] `make test-train` passes and creates artifacts
- [ ] `make test-quality` passes
- [ ] `make test-monitor` completes
- [ ] All generated files in artifacts/ are valid
- [ ] No errors in test output
- [ ] `.gitignore` includes `artifacts/` (for local testing)

```bash
# Verify .gitignore
grep -c "artifacts/" .gitignore || echo "Add artifacts/ to .gitignore"
```

---

## 📝 Example Test Session

```bash
$ make test-all

🧪 Testing data validation...
✅ Data validation passed
- File: data/raw/netflix_large.csv
- Rows: 10000
- Columns: 15

🧪 Testing model training...
🎯 Training model...
✅ Training completed
- Model saved: artifacts/models/model.pkl
- F1 Score: 0.87
- ROC-AUC: 0.92

🧪 Testing quality gate...
⚠️  Baseline not found, creating from current metrics
✅ Quality gate passed (baseline created)

🧪 Running model quality tests...
tests/test_model_quality.py::TestModelTraining::test_model_type PASSED
tests/test_model_quality.py::TestModelMetrics::test_f1_score PASSED
...
========================= 14 passed in 2.34s ==========================

🧪 Testing monitoring checks...
🔍 Running monitoring checks...
✅ Monitoring checks completed successfully

✅ All tests completed!
```

---

## 🚀 Next Steps

After local testing succeeds:

1. **Commit changes:**
   ```bash
   git add scripts/ .github/workflows/ Makefile
   git commit -m "refactor: restructure CI/CD scripts for better testing"
   ```

2. **Create PR:**
   ```bash
   git push origin feature/refactor-cicd
   git pull-request
   ```

3. **Verify CI/CD:** Watch GitHub Actions run all workflows

4. **Merge to main:** After all checks pass

---

**Last Updated:** 2026-04-12

# 📚 Scripts Directory

Organized scripts for CI/CD pipeline operations.

## 📁 Structure

```
scripts/
├── validation/          # Data & model validation
│   ├── validate_data.py     # Check data quality
│   └── quality_gate.py       # Quality gate checks
├── training/            # Model registration utilities
│   └── register_model.py      # Register trained model

Note: production monitoring code is in src/mlops_project/monitoring/.
```

## 🔧 Usage

### Data Validation
```bash
python scripts/validation/validate_data.py <data_path>
```

**Example:**
```bash
python scripts/validation/validate_data.py data/raw/netflix_large.csv
```

**Output:**
- ✅ Validates file exists
- ✅ Checks structure (rows, columns)
- ❌ Exits with error if invalid

---

### Model Training
```bash
python src/mlops_project/models/train.py --config <config_path> --data <data_path> --models-dir <models_dir>
```

**Example:**
```bash
python src/mlops_project/models/train.py \
  --config artifacts/models/best_model_config.yaml \
  --data data/processed/cleaned_data.csv \
  --models-dir artifacts/models
```

**Outputs:**
- `artifacts/models/Netflix_Prediction_final.pkl` - Trained model bundle
- `artifacts/metrics/metrics.json` - Metrics (F1, ROC-AUC)
- `$GITHUB_OUTPUT` - Variables for GitHub Actions

---

### Quality Gate
```bash
python scripts/validation/quality_gate.py <current_metrics> <baseline_metrics> <threshold>
```

**Example:**
```bash
python scripts/validation/quality_gate.py \
  artifacts/metrics/metrics.json \
  artifacts/baseline/metrics.json \
  0.95
```

**Checks:**
- Current F1 ≥ 95% of baseline
- Current ROC-AUC ≥ 95% of baseline
- Exit code 0 if pass, 1 if fail

---

### Model Artifact Check
```bash
python scripts/validation/check_model_artifact.py
```

**Example:**
```bash
python scripts/validation/check_model_artifact.py \
  --model artifacts/models/Netflix_Prediction_final.pkl \
  --preprocessor artifacts/preprocessors/preprocessor.pkl \
  --data data/raw/netflix_large.csv
```

**Checks:**
- Model artifact exists and can be loaded
- Model is fitted
- Preprocessor artifact exists and can be loaded
- Real raw data can be cleaned, featurized, transformed, and scored
- Exits 0 if inference works, 1 if any real artifact step fails

---

### Model Registration
```bash
python scripts/training/register_model.py <metrics_path> <baseline_output_path>
```

**Example:**
```bash
python scripts/training/register_model.py \
  artifacts/metrics/metrics.json \
  artifacts/baseline/metrics.json
```

**Output:**
- Saves current metrics as new baseline
- Logs registered model info

---

### Monitoring Checks
```bash
python src/mlops_project/monitoring/checks.py
```

**Outputs:**
- Drift detection results
- Performance degradation status
- `$GITHUB_OUTPUT` - Variables for auto-retrain trigger

---

## 🧪 Local Testing

Using Makefile:

```bash
# Test individual components
make test-data      # Test data validation
make test-train     # Test training
make test-quality   # Test quality gate
make test-monitor   # Test monitoring

# Run all tests
make test-all

# Clean artifacts
make clean
```

## 📊 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success ✅ |
| 1 | Failed ❌ |

Scripts return appropriate exit codes for CI/CD pipeline flow control.

## 🔐 Environment Variables

For GitHub Actions integration, scripts output variables using:
```bash
echo "variable_name=value" >> $GITHUB_OUTPUT
```

**Available in workflows:**
- `${{ steps.train.outputs.f1_score }}`
- `${{ steps.train.outputs.roc_auc }}`
- `${{ steps.check.outputs.drift_detected }}`
- `${{ steps.check.outputs.degradation_detected }}`

## 🐞 Debugging

Run scripts locally to test before committing:

```bash
# Run with verbose output
python src/mlops_project/models/train.py \
  --config artifacts/models/best_model_config.yaml \
  --data data/processed/cleaned_data.csv \
  --models-dir artifacts/models

# Check outputs
cat artifacts/metrics/metrics.json
```

---

**Last Updated:** 2026-04-12

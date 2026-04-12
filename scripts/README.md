# 📚 Scripts Directory

Organized scripts for CI/CD pipeline operations.

## 📁 Structure

```
scripts/
├── validation/          # Data & model validation
│   ├── validate_data.py     # Check data quality
│   └── quality_gate.py       # Quality gate checks
├── training/            # Model training & registration
│   ├── train_model.py        # Train model
│   └── register_model.py      # Register trained model
└── monitoring/          # Production monitoring
    └── checks.py             # Drift & performance checks
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
python scripts/training/train_model.py <output_dir>
```

**Example:**
```bash
python scripts/training/train_model.py artifacts
```

**Outputs:**
- `artifacts/models/model.pkl` - Trained model
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
python scripts/monitoring/checks.py
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
python scripts/training/train_model.py artifacts

# Check outputs
cat artifacts/metrics/metrics.json
```

---

**Last Updated:** 2026-04-12

# 🔄 GitHub Actions Workflows

CI/CD automation for MLOps pipeline.

## 📋 Workflows Overview

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| **CI** (`ci.yml`) | Push, PR to main | Lint & unit tests |
| **CT** (`ct.yml`) | Data/code changes, schedule, manual | Continuous training |
| **CD** (`cd.yml`) | Push to main | Docker build & push |
| **Monitoring** (`scheduled-monitoring.yml`) | Every 6h, manual | Drift detection & auto-retrain |

---

## 🚀 CI Workflow

**File:** `ci.yml`

**Triggers:**
- Push to `main` branch
- Pull request to `main` branch

**Jobs:**
1. Lint with Ruff
2. Run pytest tests

**Duration:** ~2-3 minutes

---

## 🔄 CT (Continuous Training) Pipeline

**File:** `ct.yml`

**Flow:**
```
Data Validation → Model Training → Quality Gate → Registration
```

**Triggers:**
- Push to `main` with data/code changes
- Weekly schedule (Monday 2 AM)
- Manual via `workflow_dispatch` with reason selection

**Jobs:**

### 1. Data Processing
- Validates data exists and structure is correct
- Outputs: data hash for versioning

### 2. Model Training
- Trains model using `scripts/training/train_model.py`
- Calculates metrics (F1, ROC-AUC)
- Uploads artifacts

### 3. Model Evaluation
- Quality gate check via `scripts/validation/quality_gate.py`
- Threshold: 95% of baseline
- Fails if degradation > 5%

### 4. Model Registration
- Registers model via `scripts/training/register_model.py`
- Saves baseline metrics
- Generates summary report

**Duration:** ~5-10 minutes

---

## 📦 CD Workflow

**File:** `cd.yml`

**Flow:**
```
Build Image → Smoke Test → Push to GHCR → Deployment Report
```

**Triggers:**
- Push to `main` branch
- Manual trigger

**Jobs:**

### 1. Build & Smoke Test
- Builds Docker image
- Tests `/health` endpoint
- Cleans up test container

### 2. Push Image
- Logs into GitHub Container Registry
- Pushes with commit hash tag + latest
- Only on main branch

### 3. Deployment Report
- Generates summary
- Shows image details and commit

**Duration:** ~3-5 minutes

---

## 📊 Monitoring Workflow

**File:** `scheduled-monitoring.yml`

**Flow:**
```
Monitoring Checks → (Drift/Degradation Detected?) → Auto-Trigger CT
```

**Triggers:**
- Every 6 hours (cron: 0 */6 * * *)
- Manual trigger

**Jobs:**

### 1. Monitoring Checks
- Runs `scripts/monitoring/checks.py`
- Outputs: drift_detected, degradation_detected

### 2. Auto-Retrain (conditional)
- Only if drift OR degradation detected
- Triggers CT pipeline with reason
- Workflow_dispatch to ct.yml

### 3. Monitoring Report
- Generates summary artifact
- Logs check results

**Duration:** ~1-2 minutes

---

## 🔑 Required Secrets

### GitHub Secrets (Settings → Secrets & Variables)

For **CT Pipeline** (if using DVC + MLflow):
- `DAGSHUB_USERNAME` - DVC remote username
- `DAGSHUB_TOKEN` - DVC remote token
- `MLFLOW_TRACKING_URI` - MLflow server URI (optional)

For **CD Pipeline**:
- `GITHUB_TOKEN` - Auto-created by GitHub (no setup needed)

**Setup:**
1. Go to repository Settings
2. Click "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add each secret

---

## 📈 Artifact Retention

| Name | Retention | Purpose |
|------|-----------|---------|
| processed-data | 30 days | Training data |
| trained-model | 90 days | Model artifacts |
| evaluation-report | 365 days | Performance metrics |
| monitoring-results | 30 days | Monitoring checks |

---

## 🔗 Workflow Dependencies

```
CI (lint + test)
  ↓
  → CT (training) - On data/code changes
  ↓
  → CD (deployment) - After training succeeds
  ↓
  → Monitoring (scheduled) - Every 6 hours
  ↓ (if drift detected)
  → CT (retrain) - Auto-triggered
```

---

## 📝 Logs & Debugging

View workflow runs:
1. GitHub repo → "Actions" tab
2. Click workflow name
3. Click run attempt
4. Click job to see detailed logs

**Useful logs:**
- Data validation output
- Training metrics (F1, AUC)
- Quality gate results
- Docker build output
- Monitoring check results

---

## ⚙️ Customization

### Change CT Schedule
Edit `ct.yml` line:
```yaml
- cron: '0 2 * * 1'  # Change this
```

Cron syntax: `minute hour day month day-of-week`

### Change Monitoring Interval
Edit `scheduled-monitoring.yml` line:
```yaml
- cron: '0 */6 * * *'  # Change this
```

### Change Quality Gate Threshold
Edit `ct.yml` quality-gate step args:
```bash
python scripts/validation/quality_gate.py ... 0.95  # Change 0.95
```

---

## 🚨 Troubleshooting

| Issue | Solution |
|-------|----------|
| Secrets not found | Verify in repo Settings → Secrets |
| Data validation fails | Check data file path and format |
| Quality gate fails | Model degradation > 5%, retrain needed |
| Docker push fails | Check GHCR authentication |
| Monitoring not triggered | Check cron schedule, run manual |

---

**Last Updated:** 2026-04-12

# PROJECT_PROGRESS

## Document Purpose

- This is the central progress tracker for the full MLOpsProject.
- It tracks:
  - CI/CD progress (M5 scope)
  - Overall system progress (M1-M6)
- It should be updated regularly by sprint and by important technical milestones.

## How To Read Quickly

- If you are a beginner, read in this order:
  1. Current snapshot
  2. CI/CD status (Done, In Progress, Next)
  3. Overall system status (Done, Blockers, Next)
  4. Latest update log

---

## 1) Current Snapshot (Sprint 1)

- Overall state: source structure and mock API are available; data and infrastructure are still incomplete.
- M5 state: CI foundation is complete, and a CD scaffold exists for image build and publish flow.
- Data/DVC blocker: required dataset object is missing on DVC remote storage.

## 2) CI/CD (M5) Status

### Completed

- CI workflow created:
  - .github/workflows/ci.yml
  - Triggers: push and pull_request to main
  - Runner: ubuntu-latest
  - Python: 3.10
  - Steps: install dependencies, Ruff lint, run tests
- Development dependencies split:
  - requirements-dev.txt
  - Includes: ruff, pytest, pytest-cov, httpx
- Unified project config:
  - pyproject.toml (Ruff and pytest config)
- Minimum smoke tests added:
  - tests/test_api.py: GET /health returns 200
  - tests/test_data.py: checks presence/columns of data/raw/netflix_large.csv (dataset-dependent, not a dummy test)
  - tests/test_model.py: dummy test
- CD scaffold created:
  - .github/workflows/cd.yml
  - Flow: build image -> smoke test /health -> push GHCR image -> deployment summary

### Locally Verified

- Ruff check: PASS (0 errors)
- Pytest: PASS (3 tests)

### Open / Next

- Push PR and confirm CI is green on GitHub Actions.
- Validate CD workflow end-to-end on GitHub.
- Connect deployment target (Render or AWS) for real environment deployment.

## 3) Overall System (M1-M6) Status

### Available

- Sprint 1 mock API is available:
  - src/mlops_project/api/schema.py
  - src/mlops_project/api/service.py
  - src/mlops_project/api/serve.py
- DVC metadata exists:
  - data/raw/netflix_large.csv.dvc
- Temporary local CSV exists for testing:
  - data/raw/netflix_large_dataset_cleaned.csv
- Base container files now exist:
  - Dockerfile
  - docker-compose.yml

### Blockers / Risks

- Data blocker:
  - dvc pull -r origin fails because remote cache is missing object for data/raw/netflix_large.csv
- Security risk:
  - README_DVC.md includes a plaintext token example and should be replaced with a safe approach
- Remaining gaps:
  - dvc.yaml is empty
  - README.md is empty

### Next Actions (Agile/Kanban)

- M5:
  - Open PR for CI/CD setup and request at least one approval
  - Ensure CI and CD are green on GitHub Actions
- M1:
  - Sync missing DVC object to remote so team can pull the official dataset
- M2-M4-M6:
  - Continue implementation using existing mock interfaces without waiting for final data flow

---

## 4) Update Log

### 2026-03-29 - Update #3 (CD Scaffold)

- Added CD workflow: .github/workflows/cd.yml
- Added initial CD flow: build image -> smoke test /health -> push GHCR -> deployment summary
- Added minimal Dockerfile for FastAPI app
- Added minimal docker-compose.yml for API service
- Fixed invalid Ruff option in CI workflow

### 2026-03-29 - Update #2 (Ruff Upgrade)

- Migrated from flake8 to Ruff (current best practice)
- Added pyproject.toml as unified config for Ruff and pytest
- Updated ci.yml to run Ruff instead of flake8
- Fixed W292, W191, E402 issues in tests/test_api.py
- Verified locally: Ruff PASS and pytest PASS

### 2026-03-29 - Update #1

- Confirmed M5 Sprint 1 scope
- Set up base CI (lint + tests) and verified locally
- Verified DVC pull failure root cause is missing remote object, not local command setup
- Downloaded temporary local CSV for testing and learning only

---

## 5) Definition of Done Tracking

### M5 Sprint 1 DoD

- [x] CI workflow runs automatically on push and PR
- [x] Lint quality gate is in place
- [x] At least one smoke/unit test exists for core API logic
- [ ] CI confirmed green on GitHub Actions after PR
- [ ] At least one teammate review and approval

### Team Baseline DoD

- [ ] DVC pull works on a clean machine
- [ ] Training script integrated with MLflow
- [ ] API fully dockerized
- [ ] Docker Compose can run the service stack
- [ ] Drift monitoring report available

---

## 6) Update Rules For This File

- For each important change, update:
  - Section 2 for M5 CI/CD changes
  - Section 3 for overall system changes
  - Section 4 with a new dated log entry
- Keep entries short, factual, verified, and clearly marked as Done, In Progress, or Blocked.

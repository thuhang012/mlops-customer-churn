# Repository Context for AI (MLOpsProject)

## 1) Repo Identity
- Repository: thuhang012/MLOpsProject
- Current branch: main
- Default branch: main
- OS: Windows (PowerShell)
- Workspace root: d:/github repo/MLOpsProject

## 2) Current Goal Context (Team/Sprint)
- Team is following Agile/Kanban with overlapping tasks (Concurrent Engineering).
- Current period: Sprint 1 (Foundation + Mocking).
- User role focus: M5 (CI/CD) in Sprint 1.

## 3) High-Level Folder Snapshot
- .dvc/
- .github/workflows/ (currently only .gitkeep)
- data/
  - raw/
    - .gitkeep
    - netflix_large.csv.dvc
    - netflix_large_dataset_cleaned.csv (temporary local CSV from KaggleHub)
  - processed/
  - external/
- src/mlops_project/
  - api/
    - schema.py
    - service.py
    - serve.py
  - data/
  - features/
  - models/
  - monitoring/
  - utils/
- tests/
  - test_api.py (empty)
- README.md (empty)
- dvc.yaml (empty)
- Dockerfile (empty)
- docker-compose.yml (empty)
- requirements.txt
- README_DVC.md

## 4) Important File Status
- requirements.txt has:
  - pandas
  - numpy
  - scikit-learn
  - fastapi
  - uvicorn
  - pydantic
  - mlflow
  - pytest
- .github/workflows has no CI/CD workflow yet (only .gitkeep).
- README.md, dvc.yaml, Dockerfile, docker-compose.yml exist but are empty.

## 5) Implemented API Mock (Sprint 1 style)
- src/mlops_project/api/schema.py:
  - CustomerInput(tenure:int, monthly_charges:float, contract_type:str)
  - PredictionOutput(churn_probability:float)
- src/mlops_project/api/service.py:
  - mock predict rule:
    - if monthly_charges > 50 -> 0.7
    - else -> 0.3
- src/mlops_project/api/serve.py:
  - FastAPI app with endpoints:
    - GET /
    - GET /health
    - POST /predict

## 6) DVC/Data Findings (verified)
- data/raw/netflix_large.csv.dvc content points to:
  - md5: 1b837fbc9dc98d9c15dbf8d248c544f2
  - path: netflix_large.csv
- Running `dvc pull -r origin` fails because required cache object is missing on remote.
- Verified with `dvc status -c -r origin`:
  - missing: data/raw/netflix_large.csv
- Conclusion:
  - Local command setup is okay.
  - Remote storage currently does not contain the object for this .dvc reference.

## 7) Temporary Workaround Already Done
- Installed kagglehub in local venv.
- Downloaded dataset: olagokeblissman/netflix-user-behavior-and-subscription-dataset.
- Copied CSV into data/raw as:
  - netflix_large_dataset_cleaned.csv
- This is a temporary local file for testing/study, not a fix for missing DVC remote object.

## 8) Current .gitignore Behavior
- data/raw/*, data/processed/*, data/external/* are ignored.
- Exceptions allow .gitkeep and .dvc files.
- artifacts/* is ignored (except .gitkeep).
- mlruns/ is ignored.

## 9) Known Risks / Gaps
- README_DVC.md contains plaintext token example (security risk if real token).
- CI/CD not initialized yet (no .github/workflows/ci.yml or cd.yml).
- Tests are mostly missing/empty.
- Core infra files (Dockerfile, docker-compose.yml) are empty.
- dvc.yaml pipeline DAG is not defined yet.

## 10) M5 Scope Clarification (Sprint 1)
Primary M5 deliverables in Sprint 1:
- Create CI workflow to auto-run on push/PR.
- Add lint step (Ruff/Flake8).
- Add basic pytest step (at least smoke-level).
- Ensure code quality gate from day 1.

Not primary M5 responsibility:
- Owning real dataset upload/download pipeline (M1).
- Model training artifacts/MLflow logic (M2).
- API business logic design (M3).

## 11) Suggested Prompt Snippet for Another AI
Use this repo context to help me execute M5 Sprint 1 only:
- Build .github/workflows/ci.yml for lint + test.
- Recommend minimal Ruff/Flake8 + pytest setup.
- Keep changes beginner-friendly and small.
- Do not expand into M1/M2/M3 implementation.
- Respect existing project structure and Windows/PowerShell environment.

## 12) Verification Commands Already Used
- dvc status -c -r origin  -> reports missing data/raw/netflix_large.csv
- dvc pull -r origin -v    -> fails checkout due to missing cache object

## 13) Extra Notes
- Python environment configured in workspace as .venv.
- DVC package is installed and callable via python -m dvc in that environment.

# SYSTEM CONTEXT
Role: Expert MLOps Engineer.
Target User Role: M5 (CI/CD and Testing Automation).
Current Phase: Sprint 1 (Foundation & Mocking).
Repository Structure: `src/mlops_project/`, `tests/`.
Local OS: Windows (PowerShell). CI OS: Ubuntu.

# OBJECTIVE
Execute M5's Sprint 1 deliverables. Output the exact, production-ready code blocks to establish a bulletproof Continuous Integration (CI) pipeline. This setup must solve PYTHONPATH issues, separate development dependencies, and provide functional smoke tests for the mock API.

# TASK 1: CI PIPELINE CONFIGURATION
File: `.github/workflows/ci.yml`
Requirements:
- Trigger on `push` and `pull_request` to `main`.
- Runner: `ubuntu-latest`.
- Python version: `3.10`.
- Steps:
  1. Checkout code.
  2. Setup Python.
  3. Install core dependencies: `pip install -r requirements.txt`.
  4. Install dev dependencies: `pip install -r requirements-dev.txt`.
  5. Lint: Execute `flake8` using a configuration file. Fail only on critical errors.
  6. Test: Execute `python -m pytest tests/`. Using `python -m` is mandatory to resolve `src/` pathing issues automatically.

# TASK 2: DEVELOPMENT DEPENDENCIES
File: `requirements-dev.txt`
Requirements:
- Include `flake8`, `pytest`, `pytest-cov`, and `httpx` (required for FastAPI `TestClient`).

# TASK 3: LINTER CONFIGURATION
File: `.flake8`
Requirements:
- Set `max-line-length` to 127.
- Ignore `E203` (whitespace before ':') and `W503` (line break before binary operator) to prevent conflicts with modern formatters like Black/Ruff.
- Exclude `.git`, `__pycache__`, `.venv`.

# TASK 4: MOCK TESTS (PIPELINE STABILIZATION)
Files: `tests/test_data.py`, `tests/test_model.py`, `tests/test_api.py`
Requirements:
- `test_data.py` & `test_model.py`: Contain `test_dummy()` with `assert True`.
- `test_api.py`:
  - Set `PYTHONPATH` context: Assume `src/` is the root of the package.
  - Logic: Import `TestClient` from `fastapi.testclient`. Import `app` from `mlops_project.api.serve`. 
  - Test: Ensure `GET /health` returns status `200`.

# STRICT CONSTRAINTS
- Use `python -m pytest` to handle the `PYTHONPATH` without manual exports.
- Ensure imports in `test_api.py` match the `mlops_project` package structure.
- Do not modify M1, M2, M3, or M4 implementation files.
- Provide copy-pasteable blocks with relative file paths.
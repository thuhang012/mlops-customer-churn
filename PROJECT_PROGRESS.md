# PROJECT_PROGRESS

## Muc tieu tai lieu
- Day la file tong hop tien do trung tam cho toan bo MLOpsProject.
- Theo doi rieng:
  - CI/CD (vai tro M5)
  - Tien do he thong tong the (M1-M6)
- Duoc cap nhat dinh ky theo sprint va theo su kien ky thuat quan trong.

## Cach doc nhanh
- Neu ban la beginner, doc theo thu tu:
  1. Tong quan hien trang
  2. CI/CD (M5) - Da xong / Dang lam / Tiep theo
  3. He thong tong the - Da xong / Blocker / Tiep theo
  4. Nhat ky cap nhat gan nhat

---

## 1) Tong quan hien trang (Sprint 1)
- Trang thai chung: Da co khung source va API mock; chua hoan tat pipeline data va infra.
- M5 (CI/CD): Da dung xong bo nen tang CI va co them CD mau de build/push image.
- Data/DVC: Dang blocker do object du lieu thieu tren remote DVC.

## 2) CI/CD (M5) Status

### Da hoan thanh
- Tao workflow CI:
  - .github/workflows/ci.yml
  - Trigger: push + pull_request vao nhanh main
  - Runner: ubuntu-latest
  - Python: 3.10
  - Steps: install dependencies, Ruff lint, run tests
- Tach dev dependencies:
  - requirements-dev.txt
  - Goi gom: ruff, pytest, pytest-cov, httpx
- Cau hinh project (unified):
  - pyproject.toml (Ruff + pytest config)
- Tao smoke tests toi thieu:
  - tests/test_api.py: test GET /health tra 200
  - tests/test_data.py: dummy test
  - tests/test_model.py: dummy test

### Da xac minh local
- Ruff check: PASS (0 loi)
- pytest: PASS (3 tests)

### Dang mo / can tiep tuc
- Chua push PR de xac nhan GitHub Actions chay xanh tren remote.
- Da co CD mau, can xac nhan chay that tren GitHub va ket noi dich vu deploy (Render/AWS).

## 3) He thong tong the (M1-M6) Status tom tat

### Da co san
- API mock sprint 1:
  - src/mlops_project/api/schema.py
  - src/mlops_project/api/service.py
  - src/mlops_project/api/serve.py
- DVC metadata cho file raw:
  - data/raw/netflix_large.csv.dvc
- Co file CSV tam local de thu nghiem:
  - data/raw/netflix_large_dataset_cleaned.csv

### Blockers / Ruil ro hien tai
- Blocker Data:
  - dvc pull -r origin fail vi thieu object cache tren remote cho data/raw/netflix_large.csv
- Bao mat:
  - README_DVC.md dang co vi du token plaintext, can thay bang cach an toan hon
- Tien do chung:
  - dvc.yaml rong
  - Dockerfile da co ban API
  - docker-compose.yml da co service API co ban
  - README.md rong

### Viec tiep theo theo huong Agile/Kanban
- M5:
  - Tao PR cho CI setup va xin 1 approval
  - Dam bao CI xanh tren GitHub Actions
- M1:
  - Dong bo lai DVC remote object de team pull duoc data chuan
- M2-M4-M6:
  - Tiep tuc theo mock interfaces da co, khong doi data that moi moi bat dau

---

## 4) Nhat ky cap nhat

### 2026-03-29 - Dot cap nhat #3 (CD Scaffold)
- Them workflow CD: .github/workflows/cd.yml.
- Luong CD mau: build image -> smoke test /health -> push GHCR -> deployment summary.
- Them Dockerfile toi thieu cho FastAPI app.
- Them docker-compose.yml toi thieu de chay service API.
- Sửa ci.yml bo option Ruff khong hop le (--show-source).

### 2026-03-29 - Dot cap nhat #2 (Upgrade Ruff)
- Upgrade flake8 sang Ruff (best practice 2026).
- Tao pyproject.toml tong hop cau hinh Ruff + pytest.
- Cap nhat ci.yml de goi ruff check thay flake8.
- Fix: W292 (newline), W191 (tabs), E402 (import position) trong tests/test_api.py.
- Xac nhan local: Ruff check PASS + pytest PASS.
- File .flake8 khong con can, config nam trong pyproject.toml.

### 2026-03-29 - Dot cap nhat #1
- Da khoanh vung dung scope M5 Sprint 1.
- Da setup CI co ban (lint + test) va verify pass local.
- Da xac minh nguyen nhan loi DVC pull la thieu object tren remote, khong phai loi lenh local.
- Da tai tam 1 file CSV local phuc vu hoc/tam test, khong thay the duoc DVC remote chuan.

---

## 5) Dinh nghia Hoan thanh (tracking de check)

### DoD M5 Sprint 1
- [x] Co workflow CI auto run tren push/PR
- [x] Co lint gate (critical errors)
- [x] Co it nhat 1 unit/smoke test cho logic cot loi API
- [ ] CI xanh tren GitHub Actions (sau khi push PR)
- [ ] Co it nhat 1 review/approve PR

### DoD he thong muc co ban (toan doi)
- [ ] DVC pull duoc data chuan tren may moi
- [ ] Train script + MLflow
- [ ] API dockerized
- [ ] Docker compose run duoc cum dich vu
- [ ] Monitoring drift co report

---

## 6) Quy uoc cap nhat file nay
- Moi lan co thay doi quan trong, cap nhat:
  - Muc 2 (CI/CD) neu lien quan M5
  - Muc 3 (Tong the) neu lien quan he thong
  - Them 1 dong vao Muc 4 (Nhat ky)
- Nguyen tac: ngan, dung su that da verify, co trang thai ro rang (Done/In progress/Blocked).

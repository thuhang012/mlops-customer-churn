# MLOps CI/CD Testing & Development

.PHONY: help test-data test-train test-quality test-monitor test-all clean \
	ci-install ci-lint ci-test ci-fast ci-dvc-pull ci-data-validate ci-all

help:
	@echo "📚 MLOps CI/CD Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  test-data       - Test data validation"
	@echo "  test-train      - Test model training"
	@echo "  test-quality    - Test quality gate"
	@echo "  test-monitor    - Test monitoring checks"
	@echo "  test-all        - Run all tests"
	@echo "  test-models     - Run model quality tests"
	@echo ""
	@echo "CI targets:"
	@echo "  ci-install          - Install dependencies for CI"
	@echo "  ci-lint             - Run Ruff like CI (E,F on src/tests)"
	@echo "  ci-test             - Run pytest like CI"
	@echo "  ci-fast             - Run lint + test (CI fast gate)"
	@echo "  ci-dvc-pull         - Pull real dataset with DVC (requires DagsHub env vars)"
	@echo "  ci-data-validate    - Validate pulled real dataset"
	@echo "  ci-all              - Run full local CI simulation"
	@echo "  clean           - Clean artifacts"
	@echo ""

# =========================
# CI convenience targets
# =========================
ci-install:
	@echo "Installing CI dependencies..."
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

ci-lint:
	@echo "Running Ruff (CI mode: E,F)..."
	python -m ruff check src tests --select E,F

ci-test:
	@echo "Running pytest..."
	python -m pytest tests/

ci-fast: ci-lint ci-test
	@echo "CI fast gate passed"

ci-dvc-pull:
	@echo "Pulling real dataset with DVC..."
	@if [ -z "$$DAGSHUB_USERNAME" ] || [ -z "$$DAGSHUB_TOKEN" ]; then \
		echo "ERROR: set DAGSHUB_USERNAME and DAGSHUB_TOKEN first"; \
		exit 1; \
	fi
	dvc remote modify --local origin auth basic
	dvc remote modify --local origin user "$$DAGSHUB_USERNAME"
	dvc remote modify --local origin password "$$DAGSHUB_TOKEN"
	dvc pull data/raw/netflix_large.csv.dvc

ci-data-validate:
	@echo "Validating real dataset..."
	@test -f data/raw/netflix_large.csv
	python scripts/validation/validate_data.py data/raw/netflix_large.csv

ci-all: ci-install ci-fast ci-dvc-pull ci-data-validate
	@echo "Full local CI simulation passed"

# Test data validation script
test-data:
	@echo "🧪 Testing data validation..."
	python scripts/validation/validate_data.py data/raw/netflix_large.csv

# Test model training script
test-train:
	@echo "🧪 Testing model training..."
	python scripts/training/train_model.py artifacts

# Test quality gate script
test-quality:
	@echo "🧪 Testing quality gate..."
	python scripts/validation/quality_gate.py artifacts/metrics/metrics.json artifacts/baseline/metrics.json 0.95

# Test monitoring script
test-monitor:
	@echo "🧪 Testing monitoring checks..."
	python scripts/monitoring/checks.py

# Test model quality tests
test-models:
	@echo "🧪 Running model quality tests..."
	pytest tests/test_model_quality.py -v

# Run all tests
test-all: test-data test-train test-quality test-models test-monitor
	@echo ""
	@echo "✅ All tests completed!"

# Clean artifacts
clean:
	@echo "🧹 Cleaning artifacts..."
	rm -rf artifacts/
	rm -rf reports/
	rm -rf logs/
	@echo "✅ Cleanup complete"

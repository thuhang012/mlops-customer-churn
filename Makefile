# MLOps CI/CD Testing & Development

.PHONY: help test-data test-train test-quality test-monitor test-all clean

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
	@echo "  clean           - Clean artifacts"
	@echo ""

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

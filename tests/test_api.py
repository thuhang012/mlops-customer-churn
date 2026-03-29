import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure imports work in local runs and CI without manual PYTHONPATH export.
ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mlops_project.api.serve import app  # noqa: E402


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

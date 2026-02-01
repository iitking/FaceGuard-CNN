import io
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)

def test_read_root_dashboard():
    """Verify that the home route serves the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "FaceGuard" in response.text
    assert "text/html" in response.headers["content-type"]

def test_health_endpoint():
    """Verify the health check endpoint returns 200 with structured status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "architecture" in data

def test_api_info_endpoint():
    """Verify the model metadata endpoint."""
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert data["app_name"] == "FaceGuard-CNN"
    assert "classes" in data
    assert "metrics" in data
    assert data["metrics"]["test_accuracy"] == 0.9828

def test_invalid_image_upload():
    """Verify that uploading invalid file types returns a 400 error."""
    response = client.post(
        "/api/predict",
        files={"file": ("test.txt", b"plain text is not an image", "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]

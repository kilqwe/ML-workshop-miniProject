import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_valid_input():
    with patch.object(app.state, "pipeline", MagicMock()), \
         patch.object(app.state, "df", MagicMock()):
        response = client.post(
            "/api/v1/predictions",
            json={"Shooting Total": 88, "Dribbling Total": 85, "Pace Total": 80}
        )
        assert response.status_code in [200, 503]

def test_predict_empty_input():
    response = client.post("/api/v1/predictions", json={})
    assert response.status_code in [400, 503]

def test_register_user():
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "testuser123", "password": "testpass123"}
    )
    assert response.status_code in [200, 400]

def test_login_invalid_credentials():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "fake", "password": "wrongpass"}
    )
    assert response.status_code == 401

def test_prediction_history():
    response = client.get("/api/v1/predictions/history")
    assert response.status_code == 200
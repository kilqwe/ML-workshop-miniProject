import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Override env vars before importing app
os.environ["DATABASE_URL"] = "postgresql://postgres:password@localhost:5432/test_db"
os.environ["REDIS_URL"] = "redis://localhost:6379"
os.environ["SECRET_KEY"] = "aslkdh54643kjg5j52l"
os.environ["DEBUG"] = "True"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
from app.db.session import Base, engine

# Create tables before tests run
Base.metadata.create_all(bind=engine)

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_ml_state():
    app.state.pipeline = MagicMock()
    app.state.df = MagicMock()
    yield
    del app.state.pipeline
    del app.state.df

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

def test_predict_empty_input():
    response = client.post("/api/v1/predictions", json={})
    assert response.status_code == 400

def test_predict_valid_input():
    from app.services.prediction_service import predict_player, find_similar_players
    import pandas as pd

    mock_similar = pd.DataFrame([{
        "Full Name": "Test Player",
        "Overall": 85,
        "Best Position": "ST",
        "Club Name": "Test FC",
        "Pace Total": 80.0,
        "Shooting Total": 88.0,
        "Passing Total": 75.0,
        "Dribbling Total": 85.0,
        "Defending Total": 40.0,
        "Physicality Total": 70.0,
    }])

    app.state.pipeline.__getitem__ = MagicMock(side_effect=lambda k: {
        "core_stats": ["Pace Total", "Shooting Total", "Passing Total",
                       "Dribbling Total", "Defending Total", "Physicality Total"],
        "gk_stats": ["Goalkeeper Diving", "Goalkeeper Handling",
                     "Goalkeeper Kicking", "Goalkeeper Positioning", "Goalkeeper Reflexes"],
        "raw_centroids": {"ST": {"Pace Total": 80}},
    }[k])

    with patch("app.api.v1.routes.predictions.predict_player",
               return_value=("FWD", 85, "ST")), \
         patch("app.api.v1.routes.predictions.find_similar_players",
               return_value=mock_similar), \
         patch("app.api.v1.routes.predictions.get_cache",
               return_value=None), \
         patch("app.api.v1.routes.predictions.set_cache"):
        response = client.post(
            "/api/v1/predictions",
            json={"Shooting Total": 88, "Dribbling Total": 85, "Pace Total": 80}
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_rating" in data
        assert "predicted_group" in data

def test_register_user():
    response = client.post(
        "/api/v1/auth/register",
        json={"username": "testuser123", "password": "testpass123"}
    )
    assert response.status_code in [200, 400]

def test_login_invalid_credentials():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "nobody", "password": "wrongpass"}
    )
    assert response.status_code == 401

def test_prediction_history():
    response = client.get("/api/v1/predictions/history")
    assert response.status_code == 200
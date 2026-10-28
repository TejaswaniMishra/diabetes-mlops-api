"""Automated tests for the Diabetes Prediction API."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    """The health endpoint should report that the service is up."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_valid_response():
    """A valid patient payload should return a well formed prediction."""
    payload = {
        "Pregnancies": 6,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["Diabetes", "No diabetes"]
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_invalid_input():
    """Negative values should be rejected with a 422 status code."""
    payload = {
        "Pregnancies": -1,
        "Glucose": 148.0,
        "BloodPressure": 72.0,
        "SkinThickness": 35.0,
        "Insulin": 0.0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
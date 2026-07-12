"""
Unit tests for FastAPI.
"""

from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "message": "Heart Disease Prediction API is running!"
    }


def test_prediction():

    sample = {

        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1

    }

    response = client.post(
        "/predict",
        json=sample
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "probability" in data

    assert "diagnosis" in data
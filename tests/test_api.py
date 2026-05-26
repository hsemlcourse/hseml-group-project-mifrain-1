from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app


def test_health_endpoint_loads_model() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_endpoint_returns_price() -> None:
    client = TestClient(app)
    payload = {
        "dataset_part": "used",
        "brand": "BMW",
        "model": "X5",
        "trim_or_variant": "BMW X5 2020 A/T Turbo SUV",
        "model_year": 2020,
        "snapshot_date": "2023-03-27",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_price_egp"] > 0
    assert body["model_name"] == "hist_gradient_boosting"
    assert body["primary_metric"] == "mae"

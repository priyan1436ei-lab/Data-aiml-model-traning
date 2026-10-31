"""
Unit Tests for FastAPI Explorer Endpoints
"""
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Universal Global Coding Dataset Engine"

def test_datasets_endpoint():
    response = client.get("/datasets")
    assert response.status_code in [200, 404]  # 404 if discover hasn't been run yet


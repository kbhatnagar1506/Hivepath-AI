"""Shared fixtures.

Tests run against a known-empty credential set so nothing reaches the network.
Environment variables take precedence over ``.env`` in pydantic-settings, so
clearing them here neutralises any real ``.env`` a developer has locally.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from hivepath.api.application import create_app
from hivepath.config import Settings, get_settings
from hivepath.domain import Depot, Stop, Vehicle
from hivepath.ml.service_time import get_service_time_model
from hivepath.storage import reset_repositories

CREDENTIAL_VARS = (
    "GOOGLE_MAPS_API_KEY",
    "GOOGLE_STREET_VIEW_API_KEY",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch):
    """Blank credentials, reset caches and shared state around every test."""
    for name in CREDENTIAL_VARS:
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    get_settings.cache_clear()
    get_service_time_model.cache_clear()
    reset_repositories()
    yield
    get_settings.cache_clear()
    get_service_time_model.cache_clear()
    reset_repositories()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def base_time() -> datetime:
    return datetime(2026, 8, 8, 8, 0, 0, tzinfo=UTC)


@pytest.fixture
def depot() -> Depot:
    return Depot(id="depot", lat=42.3601, lng=-71.0589)


@pytest.fixture
def stops() -> list[Stop]:
    """Eight stops spread around the depot, each within one vehicle's capacity."""
    return [
        Stop(
            id=f"s{i}",
            lat=42.3601 + (i - 4) * 0.008,
            lng=-71.0589 + ((i % 3) - 1) * 0.008,
            demand=40,
            priority=1,
        )
        for i in range(8)
    ]


@pytest.fixture
def vehicles() -> list[Vehicle]:
    return [
        Vehicle(id="v1", capacity=200, fuel_type="diesel"),
        Vehicle(id="v2", capacity=200, fuel_type="ev"),
    ]


@pytest.fixture
def optimize_payload() -> dict:
    """A valid request body for POST /api/v1/optimize/routes."""
    return {
        "run_id": "test-run",
        "depot": {"id": "depot", "lat": 42.3601, "lng": -71.0589},
        "vehicles": [
            {"id": "v1", "capacity": 200, "fuel_type": "diesel"},
            {"id": "v2", "capacity": 200, "fuel_type": "ev"},
        ],
        "stops": [
            {
                "id": f"s{i}",
                "lat": 42.3601 + (i - 4) * 0.008,
                "lng": -71.0589 + ((i % 3) - 1) * 0.008,
                "demand": 40,
            }
            for i in range(8)
        ],
        # Keep unit tests fast and hermetic.
        "time_limit_sec": 2,
        "use_access_analysis": False,
    }


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())

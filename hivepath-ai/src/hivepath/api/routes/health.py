"""Liveness, readiness, and capability reporting."""

from __future__ import annotations

from fastapi import APIRouter

from hivepath import __version__
from hivepath.api.schemas import HealthResponse
from hivepath.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report status and which optional features are actually available.

    The feature flags reflect configuration, so a deployment can be checked
    without guessing which integrations have credentials.
    """
    settings = get_settings()
    return HealthResponse(
        status="healthy",
        service=settings.service_name,
        version=__version__,
        environment=settings.environment.value,
        features={
            "google_maps_distances": settings.has_maps_credentials,
            "street_view_imagery": settings.has_street_view_credentials,
            "accessibility_analysis": settings.has_vlm_credentials
            and settings.has_street_view_credentials,
        },
    )


@router.get("/ready", tags=["health"])
def ready() -> dict[str, bool]:
    """Readiness probe. The solver has no external dependencies to await."""
    return {"ready": True}

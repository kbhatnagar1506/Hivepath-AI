"""Kerbside accessibility assessment endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hivepath.accessibility import evaluate
from hivepath.accessibility.analyzer import AccessibilityAnalyzer
from hivepath.api.schemas import AccessibilityRequest
from hivepath.config import get_settings
from hivepath.integrations.street_view import DEFAULT_HEADINGS

router = APIRouter(prefix="/accessibility", tags=["accessibility"])

_analyzer: AccessibilityAnalyzer | None = None


def _get_analyzer() -> AccessibilityAnalyzer:
    """Lazily construct the analyzer so its cache is shared across requests."""
    global _analyzer
    if _analyzer is None:
        _analyzer = AccessibilityAnalyzer()
    return _analyzer


@router.post("/analyze", summary="Assess kerbside access at a location")
async def analyze(request: AccessibilityRequest) -> dict[str, Any]:
    """Score how hard it is to stop, park, and unload at a location.

    Requires Street View and vision-model credentials; returns 503 rather than
    a misleading neutral score when they are absent.
    """
    settings = get_settings()
    if not (settings.has_street_view_credentials and settings.has_vlm_credentials):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "accessibility analysis needs GOOGLE_MAPS_API_KEY and OPENAI_API_KEY; "
                "see .env.example"
            ),
        )

    analysis = await _get_analyzer().analyze(
        request.lat,
        request.lng,
        headings=request.headings or DEFAULT_HEADINGS,
        vehicle_desc=request.vehicle_desc,
    )
    decision = evaluate(analysis)

    return {
        "stop_id": request.stop_id,
        "lat": request.lat,
        "lng": request.lng,
        "analysis": analysis,
        "decision": decision.to_dict(),
    }

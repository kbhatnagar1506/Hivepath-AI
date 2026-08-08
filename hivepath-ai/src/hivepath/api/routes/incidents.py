"""Disruption reporting and automatic replanning."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hivepath.api.schemas import IncidentRequest
from hivepath.logging_config import get_logger
from hivepath.planning import replan
from hivepath.storage import get_incident_repository

logger = get_logger(__name__)

router = APIRouter(prefix="/incidents", tags=["incidents"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Report an incident")
async def report_incident(incident: IncidentRequest) -> dict[str, Any]:
    """Block a stop, and optionally replan a run that included it."""
    repository = get_incident_repository()
    repository.block(incident.stop_id, incident.ttl_minutes)

    response: dict[str, Any] = {
        "ok": True,
        "incident_id": incident.id,
        "blocked_stops": sorted(repository.active_ids()),
    }

    if incident.replan_from_run_id and incident.new_run_id:
        plan = await replan(incident.replan_from_run_id, incident.new_run_id)
        if plan is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"cannot replan: no stored request for {incident.replan_from_run_id!r}",
            )
        response["new_run_id"] = incident.new_run_id
        response["plan"] = plan.to_dict()

    return response


@router.get("", summary="List active blocks")
def list_incidents() -> dict[str, Any]:
    active = get_incident_repository().active()
    return {
        "ok": True,
        "blocked": [
            {"stop_id": stop_id, "expires_at": expiry} for stop_id, expiry in sorted(active.items())
        ],
    }


@router.delete("/{stop_id}", summary="Clear a block")
def clear_incident(stop_id: str) -> dict[str, Any]:
    removed = get_incident_repository().unblock(stop_id)
    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"stop {stop_id!r} is not blocked"
        )
    return {"ok": True, "stop_id": stop_id, "unblocked": True}

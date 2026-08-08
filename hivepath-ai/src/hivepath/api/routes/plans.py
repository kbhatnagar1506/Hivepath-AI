"""Retrieval and metrics for stored plans."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from hivepath.storage import get_plan_repository

router = APIRouter(prefix="/plans", tags=["plans"])


@router.get("/{run_id}", summary="Fetch a stored plan")
def get_plan(run_id: str) -> dict[str, Any]:
    plan = get_plan_repository().get(run_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no plan for run_id {run_id!r}"
        )
    return plan


@router.get("/{run_id}/metrics", summary="Summary metrics for a plan")
def plan_metrics(run_id: str) -> dict[str, Any]:
    """Aggregate distance, time, emissions, and coverage for a plan."""
    plan = get_plan_repository().get(run_id)
    if plan is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"no plan for run_id {run_id!r}"
        )
    if not plan.get("ok"):
        return {"ok": False, "run_id": run_id, "error": plan.get("error")}

    summary = plan.get("summary", {})
    routes = plan.get("routes", [])
    served = summary.get("served_stops", 0)
    dropped = summary.get("dropped_stops", 0)
    total = served + dropped

    return {
        "ok": True,
        "run_id": run_id,
        "routes": len(routes),
        "vehicles_used": sum(1 for r in routes if len(r.get("stops", [])) > 1),
        "total_distance_km": summary.get("total_distance_km", 0.0),
        "total_drive_min": summary.get("total_drive_min", 0),
        "total_co2_kg": summary.get("total_co2_kg", 0.0),
        "served_stops": served,
        "dropped_stops": dropped,
        "service_rate": round(served / total, 4) if total else 0.0,
        "matrix_source": plan.get("telemetry", {}).get("matrix_source"),
    }

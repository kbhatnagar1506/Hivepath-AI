from fastapi import APIRouter
from services.plan_store import get_plan

router = APIRouter()

@router.get("/metrics/plan")
def metrics(run_id: str):
    pl = get_plan(run_id)
    if not pl or not pl.get("ok"):
        return {"ok": False}
    co2 = sum(r["co2_kg"] for r in pl["routes"])
    return {
        "ok": True,
        "run_id": run_id,
        "routes": len(pl["routes"]),
        "total_distance_km": pl["summary"]["total_distance_km"],
        "total_drive_min": pl["summary"]["total_drive_min"],
        "total_co2_kg": round(co2, 2)
    }

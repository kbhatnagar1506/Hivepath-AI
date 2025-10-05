from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from services.incident_store import block_stop, active_blocks
from services.request_store import get_request
from services.plan_store import save_plan, get_plan
from services.ortools_solver import solve_vrp, plan_to_routes_for_warm_start

router = APIRouter()

class Incident(BaseModel):
    id: str
    type: str
    target: dict     # e.g. {"stop_id": "C3"}
    severity: float = 1.0
    ttl_minutes: int = 90
    note: Optional[str] = None
    replan_from_run_id: Optional[str] = None
    new_run_id: Optional[str] = None

@router.post("/ingest")
def ingest_incident(inc: Incident):
    stop_id = inc.target.get("stop_id")
    if stop_id:
        block_stop(stop_id, inc.ttl_minutes)

        if inc.replan_from_run_id and inc.new_run_id:
            req = get_request(inc.replan_from_run_id)
            if req:
                blocked = {stop_id} if stop_id else set()
                # Try warm start from previous plan
                prev = get_plan(inc.replan_from_run_id)
                warm = plan_to_routes_for_warm_start(prev) if prev and prev.get("ok") else None
                plan = solve_vrp(
                    req["depot"], req["stops"], req["vehicles"], req.get("speed_kmph", 40.0),
                    blocked_stop_ids=blocked,
                    initial_routes=warm,
                    num_workers=8,
                    time_limit_sec=6
                )
                plan["run_id"] = inc.new_run_id
                save_plan(inc.new_run_id, plan)
                return {"ok": True, "blocked": list(active_blocks().keys()), "new_run_id": inc.new_run_id, "plan": plan}
    return {"ok": True, "blocked": list(active_blocks().keys())}

@router.get("/active")
def active():
    return {"ok": True, "blocked": active_blocks()}

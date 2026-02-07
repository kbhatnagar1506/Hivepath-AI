from fastapi import APIRouter
from services.plan_store import get_plan

router = APIRouter()

@router.get("/plan/{run_id}")
def get_plan_endpoint(run_id: str):
    plan = get_plan(run_id)
    return plan or {"ok": False, "error": "not_found"}

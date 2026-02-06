from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import json
import time

router = APIRouter()

class ActualsRequest(BaseModel):
    run_id: str
    vehicle_id: str
    stop_id: str
    arrived_min: int              # minutes since plan start
    departed_min: int
    walk_m: Optional[int] = None  # extra walking distance
    blocked_flag: bool = False

@router.post("/actuals")
def log_actuals(actuals: ActualsRequest):
    """Log actual delivery data for ML training."""
    # In production, this would write to a proper database
    # For now, we'll just log to a file
    log_entry = {
        "timestamp": time.time(),
        "run_id": actuals.run_id,
        "vehicle_id": actuals.vehicle_id,
        "stop_id": actuals.stop_id,
        "arrived_min": actuals.arrived_min,
        "departed_min": actuals.departed_min,
        "service_min": actuals.departed_min - actuals.arrived_min,
        "walk_m": actuals.walk_m,
        "blocked_flag": actuals.blocked_flag
    }
    
    # Append to actuals log file
    with open("actuals.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    
    return {"ok": True, "logged": True}

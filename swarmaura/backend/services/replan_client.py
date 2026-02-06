import os, httpx, uuid
from typing import Dict, Any
BASE = os.getenv("BASE_BACKEND_URL", "http://localhost:8000")
async def replan_from(run_id: str, stop_id: str, severity: float) -> Dict[str, Any]:
    new_run = f"{run_id}-replan-{uuid.uuid4().hex[:6]}"
    payload = {
        "id": f"obs-{uuid.uuid4().hex[:6]}",
        "type": "blocked_dock",
        "target": {"stop_id": stop_id},
        "severity": severity,
        "ttl_minutes": 120,
        "replan_from_run_id": run_id,
        "new_run_id": new_run
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(f"{BASE}/api/v1/incidents/ingest", json=payload)
        r.raise_for_status()
        return r.json()
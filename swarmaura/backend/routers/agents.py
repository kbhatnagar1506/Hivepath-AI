from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from services.streetview_client import fetch_streetview_image
from services.vlm_client import call_vlm_with_images
from services.access_policy import decision_from_scores
from services.replan_client import replan_from

router = APIRouter()

class AnalyzeRequest(BaseModel):
    run_id: str = Field(..., description="Existing plan run id")
    stop_id: str
    lat: float
    lng: float
    vehicle_desc: str = "26-ft box truck"
    headings: Optional[List[int]] = None
    autoincident: bool = True

@router.post("/streetview-analyze")
async def streetview_analyze(req: AnalyzeRequest):
    heads = req.headings or [0, 90, 180, 270]
    images = []
    for h in heads[:4]:
        try:
            img = await fetch_streetview_image(req.lat, req.lng, heading=h)
            images.append(img)
        except Exception:
            pass
    if not images:
        raise HTTPException(status_code=422, detail="no_streetview_images")

    try:
        result = await call_vlm_with_images(images, req.lat, req.lng, req.vehicle_desc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vlm_error: {e}")

    scores = result.get("analysis", {})
    decision = decision_from_scores(scores)

    out = {
        "stop_id": req.stop_id,
        "lat": req.lat, "lng": req.lng,
        "scores": scores,
        "recommendation": result.get("recommendation", {}),
        "notes": result.get("notes", ""),
        "decision": decision
    }

    if req.autoincident and decision["block"]:
        repl = await replan_from(req.run_id, req.stop_id, decision["severity"])
        out["replan"] = repl

    return out

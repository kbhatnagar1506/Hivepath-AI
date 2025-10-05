from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx, asyncio, os, json
from math import radians, cos, sin
import time

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY = os.getenv("GOOGLE_STREET_VIEW_API_KEY")

HEADINGS_COARSE = [0, 90, 180, 270]

class AnalyzeReq(BaseModel):
    lat: float
    lng: float
    headings: Optional[List[int]] = None
    fov: int = 90
    pitch: int = -10
    size: str = "640x640"

class Finding(BaseModel):
    label: str
    present: bool
    confidence: float

class Hazard(BaseModel):
    label: str
    severity: str  # "minor|major|critical"

class Evidence(BaseModel):
    heading: int
    notes: str

class AnalyzeResp(BaseModel):
    access_score: int
    pred_service_time_sec: int
    findings: List[Finding]
    hazards: List[Hazard]
    evidence: List[Evidence]

def sv_url(lat, lng, heading, pitch, fov, size):
    return (f"https://maps.googleapis.com/maps/api/streetview"
            f"?size={size}&location={lat},{lng}&heading={heading}"
            f"&pitch={pitch}&fov={fov}&key={GOOGLE_KEY}")

async def fetch_image(client, url):
    r = await client.get(url, timeout=15)
    r.raise_for_status()
    # Return image bytes for VLM processing
    return r.content

async def analyze_with_vlm(image_bytes_list, headings, lat, lng):
    """Analyze Street View images with VLM using Colmena's strict JSON pattern."""
    prompt = f"""Analyze curbside access for logistics at ({lat},{lng}).
Return ONLY JSON: {{"access_score":0-100,"pred_service_time_sec":int,
"findings":[{{"label":str,"present":bool,"confidence":0-1}}],
"hazards":[{{"label":str,"severity":"minor|major|critical"}}],
"evidence":[{{"heading":int,"notes":str}}]}}.

Scoring rubric:
- Dedicated loading zone, curb cutouts, wide shoulders, legal parking nearby → +40..+60
- Clear driveway/loading bay within 30m → +20..+30
- Bus lane/no-stopping/hydrant, heavy obstruction, narrow lane, stairs-only → −30..−60
- Good visibility/turning radius/signage → +5..+15
- Severe hazards (bike lane conflict, blocked entrance) → cap access_score ≤ 35

Service time model (defaults):
- Base 240s (4 min)
- −60s if dedicated loading zone; −45s if legal near-kerb parking; +90s if no legal stopping within 60m; +60s if stairs; +45s for heavy pedestrian/traffic conflict."""
    
    try:
        from openai import OpenAI
        oai = OpenAI(api_key=OPENAI_API_KEY)
        
        # Prepare content with images
        content = [{"type": "text", "text": prompt}]
        for i, img_bytes in enumerate(image_bytes_list):
            import base64
            b64_img = base64.b64encode(img_bytes).decode('utf-8')
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_img}",
                    "detail": "low"
                }
            })
        
        resp = oai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an access inspector. Return ONLY valid JSON."},
                {"role": "user", "content": content}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(resp.choices[0].message.content)
        
        # Ensure evidence includes heading information
        if "evidence" not in result:
            result["evidence"] = []
        for i, heading in enumerate(headings):
            if i < len(result["evidence"]):
                result["evidence"][i]["heading"] = heading
            else:
                result["evidence"].append({"heading": heading, "notes": f"View {heading}°"})
        
        return result
        
    except Exception as e:
        # Fallback to default values if VLM fails
        return {
            "access_score": 50,
            "pred_service_time_sec": 240,
            "findings": [{"label": "unknown", "present": True, "confidence": 0.5}],
            "hazards": [],
            "evidence": [{"heading": h, "notes": f"View {h}° (analysis failed)"} for h in headings]
        }

async def analyze_location(req: AnalyzeReq) -> AnalyzeResp:
    """Analyze a single location for curbside access."""
    if not GOOGLE_KEY:
        raise ValueError("GOOGLE_STREET_VIEW_API_KEY not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    headings = req.headings or HEADINGS_COARSE
    
    async with httpx.AsyncClient() as client:
        try:
            # Fetch images concurrently
            image_tasks = [
                fetch_image(client, sv_url(req.lat, req.lng, h, req.pitch, req.fov, req.size))
                for h in headings
            ]
            image_bytes_list = await asyncio.gather(*image_tasks)
            
            # Analyze with VLM
            result = await analyze_with_vlm(image_bytes_list, headings, req.lat, req.lng)
            
            # Convert to response model
            return AnalyzeResp(
                access_score=result["access_score"],
                pred_service_time_sec=result["pred_service_time_sec"],
                findings=[Finding(**f) for f in result["findings"]],
                hazards=[Hazard(**h) for h in result["hazards"]],
                evidence=[Evidence(**e) for e in result["evidence"]]
            )
            
        except Exception as e:
            # Return default response on error
            return AnalyzeResp(
                access_score=50,
                pred_service_time_sec=240,
                findings=[Finding(label="error", present=True, confidence=0.0)],
                hazards=[Hazard(label="analysis_failed", severity="minor")],
                evidence=[Evidence(heading=h, notes=f"Error: {str(e)}") for h in headings]
            )

async def batch_analyze_locations(requests: List[AnalyzeReq], max_concurrent: int = 8) -> List[AnalyzeResp]:
    """Analyze multiple locations with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def analyze_with_semaphore(req):
        async with semaphore:
            return await analyze_location(req)
    
    tasks = [analyze_with_semaphore(req) for req in requests]
    return await asyncio.gather(*tasks)




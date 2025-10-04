from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.streetscout import AnalyzeReq, AnalyzeResp, analyze_location, batch_analyze_locations

router = APIRouter()

class BatchAnalyzeReq(BaseModel):
    locations: List[AnalyzeReq]
    max_concurrent: int = 8

class ScanRadiusReq(BaseModel):
    lat: float
    lng: float
    radius_m: float = 50.0
    grid_size: int = 3
    headings: Optional[List[int]] = None

@router.post("/analyze")
async def analyze(req: AnalyzeReq) -> AnalyzeResp:
    """Analyze a single location for curbside access."""
    try:
        return await analyze_location(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/score-stops")
async def score_stops(req: BatchAnalyzeReq) -> List[AnalyzeResp]:
    """Batch analyze multiple stops with concurrency control."""
    try:
        return await batch_analyze_locations(req.locations, req.max_concurrent)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.post("/scan-radius")
async def scan_radius(req: ScanRadiusReq) -> List[AnalyzeResp]:
    """Grid sample locations within radius for access analysis."""
    import math
    
    # Generate grid points within radius
    locations = []
    step = req.radius_m / req.grid_size
    
    for i in range(-req.grid_size, req.grid_size + 1):
        for j in range(-req.grid_size, req.grid_size + 1):
            if i == 0 and j == 0:
                continue  # Skip center point
                
            # Convert to lat/lng offset
            lat_offset = (i * step) / 111000  # rough conversion
            lng_offset = (j * step) / (111000 * math.cos(math.radians(req.lat)))
            
            new_lat = req.lat + lat_offset
            new_lng = req.lng + lng_offset
            
            # Check if within radius
            distance = math.sqrt((i * step)**2 + (j * step)**2)
            if distance <= req.radius_m:
                locations.append(AnalyzeReq(
                    lat=new_lat,
                    lng=new_lng,
                    headings=req.headings
                ))
    
    # Add center point
    locations.append(AnalyzeReq(
        lat=req.lat,
        lng=req.lng,
        headings=req.headings
    ))
    
    try:
        return await batch_analyze_locations(locations, max_concurrent=8)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Radius scan failed: {str(e)}")

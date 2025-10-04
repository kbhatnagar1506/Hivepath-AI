from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional
from services.ortools_solver import solve_vrp
from services.plan_store import save_plan
from services.request_store import save_request

router = APIRouter()

class Depot(BaseModel):
    id: str
    lat: float
    lng: float

class Vehicle(BaseModel):
    id: str
    capacity: int = 1000
    fuel_type: Optional[str] = "diesel"

class Stop(BaseModel):
    id: str
    lat: float
    lng: float
    demand: int = 0
    priority: int = 1
    time_window: Optional[dict] = None  # {"start": ISO, "end": ISO}

class RouteRequest(BaseModel):
    run_id: str = Field(..., description="Plan identifier")
    depot: Depot
    vehicles: List[Vehicle]
    stops: List[Stop]
    speed_kmph: float = 40.0
    # NEW (all optional)
    time_limit_sec: int = 8
    num_workers: int = 0
    default_service_min: int = 5
    allow_drop: bool = True
    drop_penalty_per_priority: int = 5000
    debug_log: bool = False
    preset: Optional[str] = None  # "fast" | "balanced" | "quality"
    # NEW ML toggles
    use_service_time_model: bool = True
    use_warmstart: bool = True
    # Access-aware toggles
    use_access_analysis: bool = True
    access_penalty_weight: float = 0.002
    drop_penalty_weight: float = 0.02
    # Google Maps integration
    use_google_maps: bool = True  # DEFAULT: Use Google Maps for real distances

@router.post("/routes")
async def optimize_routes(req: RouteRequest):
    stops = [s.model_dump() for s in req.stops]

    # (A) service time predictions
    if req.use_service_time_model:
        try:
            from services.service_time_model import predict_minutes
            svc = predict_minutes(stops)
            for s in stops:
                s["service_min"] = int(svc.get(s["id"], s.get("service_min", 5)))
        except Exception:
            pass  # fallback to default service times

    # (B) access analysis
    if req.use_access_analysis:
        try:
            from services.access_enricher import enrich_stops_with_access
            stops = await enrich_stops_with_access(stops)
        except Exception:
            pass  # fallback to stops without access scores

    # (C) optional warm-start routes
    warm_routes = None
    if req.use_warmstart:
        try:
            from services.warmstart_policy import build_initial_routes
            warm_routes = build_initial_routes(req.depot.model_dump(), stops, [v.model_dump() for v in req.vehicles], req.speed_kmph)
        except Exception:
            warm_routes = None

    # Apply presets if specified - ULTRA-OPTIMIZED for speed
    if req.preset:
        n_stops = len(req.stops)
        presets = {
            "ultra_fast": dict(
                time_limit_sec=1 if n_stops <= 8 else 2 if n_stops <= 15 else 3,
                num_workers=8, 
                allow_drop=True, 
                drop_penalty_per_priority=25000,  # Increased penalty to serve more stops
                use_service_time_model=False,  # Skip ML for max speed
                use_warmstart=False,  # Skip warmstart for max speed
                use_access_analysis=False,  # Skip access analysis for max speed
                use_google_maps=False  # Use Haversine for max speed
            ),
            "fast":     dict(time_limit_sec=2 if n_stops <= 8 else 4, num_workers=8, allow_drop=True, drop_penalty_per_priority=20000, use_google_maps=True),  # Increased
            "balanced": dict(time_limit_sec=4 if n_stops <= 8 else 6, num_workers=8, allow_drop=True, drop_penalty_per_priority=25000, use_google_maps=True),  # Increased
            "quality":  dict(time_limit_sec=8 if n_stops <= 8 else 10, num_workers=8, allow_drop=False, use_google_maps=True),
            "high_quality": dict(
                time_limit_sec=6 if n_stops <= 8 else 8, 
                num_workers=8, 
                allow_drop=True, 
                drop_penalty_per_priority=50000,  # Very high penalty to serve maximum stops
                use_service_time_model=True,  # Enable ML for better decisions
                use_warmstart=True,  # Enable warmstart for better initial solutions
                use_access_analysis=True,  # Enable access analysis
                use_google_maps=True  # Use Google Maps for maximum accuracy
            )
        }
        kw = presets.get(req.preset, {})
        # Override request values with preset values
        time_limit_sec = kw.get("time_limit_sec", req.time_limit_sec)
        num_workers = kw.get("num_workers", req.num_workers)
        allow_drop = kw.get("allow_drop", req.allow_drop)
        drop_penalty_per_priority = kw.get("drop_penalty_per_priority", req.drop_penalty_per_priority)
        
        # Override ML features and Google Maps for presets
        if req.preset in ["ultra_fast", "fast", "balanced", "quality", "high_quality"]:
            req.use_service_time_model = kw.get("use_service_time_model", req.use_service_time_model)
            req.use_warmstart = kw.get("use_warmstart", req.use_warmstart)
            req.use_access_analysis = kw.get("use_access_analysis", req.use_access_analysis)
            req.use_google_maps = kw.get("use_google_maps", req.use_google_maps)
    else:
        time_limit_sec = req.time_limit_sec
        num_workers = req.num_workers
        allow_drop = req.allow_drop
        drop_penalty_per_priority = req.drop_penalty_per_priority
    
    # Add warm start if available
    if warm_routes:
        time_limit_sec = time_limit_sec  # could reduce time limit with warm start
    
    plan = solve_vrp(
        req.depot.model_dump(),
        stops,
        [v.model_dump() for v in req.vehicles],
        req.speed_kmph,
        time_limit_sec=time_limit_sec,
        num_workers=num_workers,
        default_service_min=req.default_service_min,
        allow_drop=allow_drop,
        drop_penalty_per_priority=drop_penalty_per_priority,
        debug_log=req.debug_log,
        initial_routes=warm_routes,
        use_access_scores=req.use_access_analysis,
        access_penalty_weight=req.access_penalty_weight,
        drop_penalty_weight=req.drop_penalty_weight,
        use_google_maps=req.use_google_maps
    )
    plan["run_id"] = req.run_id
    save_plan(req.run_id, plan)
    save_request(req.run_id, req.model_dump())
    return plan

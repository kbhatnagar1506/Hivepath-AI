from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from services.ortools_solver import solve_vrp
from services.plan_store import save_plan
from services.request_store import save_request

router = APIRouter()

class Location(BaseModel):
    """Enhanced location model for multi-location routing"""
    id: str
    lat: float
    lng: float
    location_type: Literal["depot", "pickup", "delivery", "service", "waypoint"] = "service"
    demand: int = 0  # Positive for pickup, negative for delivery
    priority: int = 1
    time_window: Optional[dict] = None  # {"start": ISO, "end": ISO}
    service_min: int = 5
    capacity_required: int = 0  # Space needed at this location
    dependencies: List[str] = []  # Location IDs that must be visited before this one
    group_id: Optional[str] = None  # Group locations together (e.g., same customer)
    access_constraints: Optional[dict] = None  # Vehicle type restrictions, etc.

class Vehicle(BaseModel):
    id: str
    capacity: int = 1000
    fuel_type: Optional[str] = "diesel"
    max_stops: Optional[int] = None  # Maximum stops per route
    allowed_location_types: List[str] = ["pickup", "delivery", "service"]  # What this vehicle can handle
    start_location: Optional[str] = None  # Specific start location ID
    end_location: Optional[str] = None  # Specific end location ID

class RouteSequence(BaseModel):
    """Predefined sequence of locations for a specific vehicle"""
    vehicle_id: str
    location_sequence: List[str]  # Ordered list of location IDs
    priority: int = 1  # Higher priority sequences are preferred

class MultiLocationRequest(BaseModel):
    run_id: str = Field(..., description="Plan identifier")
    locations: List[Location]  # All locations (depots, pickups, deliveries, etc.)
    vehicles: List[Vehicle]
    speed_kmph: float = 40.0
    
    # Routing constraints
    time_limit_sec: int = 8
    num_workers: int = 0
    allow_drop: bool = True
    drop_penalty_per_priority: int = 5000
    
    # Multi-location specific
    enforce_sequences: bool = False  # Whether to enforce predefined sequences
    sequences: List[RouteSequence] = []  # Predefined sequences
    pickup_delivery_pairs: List[Dict[str, str]] = []  # [{"pickup": "P1", "delivery": "D1"}]
    
    # ML and optimization
    use_service_time_model: bool = True
    use_warmstart: bool = True
    use_access_analysis: bool = True
    access_penalty_weight: float = 0.002
    drop_penalty_weight: float = 0.02
    
    # Preset for quick configuration
    preset: Optional[str] = None  # "pickup_delivery" | "multi_depot" | "service_routes"

@router.post("/multi-location-routes")
async def optimize_multi_location_routes(req: MultiLocationRequest):
    """
    Optimize routes for complex multi-location scenarios:
    - Pickup and delivery pairs
    - Multiple depots
    - Service routes with waypoints
    - Location dependencies
    - Vehicle-specific constraints
    """
    
    # Convert multi-location request to standard VRP format
    vrp_request = _convert_to_vrp_format(req)
    
    # Apply presets if specified
    if req.preset:
        vrp_request = _apply_multi_location_preset(req, vrp_request)
    
    # Run the optimized VRP solver
    plan = solve_vrp(
        depot=vrp_request["depot"],
        stops=vrp_request["stops"],
        vehicles=vrp_request["vehicles"],
        speed_kmph=req.speed_kmph,
        time_limit_sec=req.time_limit_sec,
        num_workers=req.num_workers,
        allow_drop=req.allow_drop,
        drop_penalty_per_priority=req.drop_penalty_per_priority,
        use_access_scores=req.use_access_analysis,
        access_penalty_weight=req.access_penalty_weight,
        drop_penalty_weight=req.drop_penalty_weight
    )
    
    # Convert result back to multi-location format
    multi_location_plan = _convert_from_vrp_format(plan, req)
    
    # Save the plan
    plan["run_id"] = req.run_id
    save_plan(req.run_id, multi_location_plan)
    save_request(req.run_id, req.model_dump())
    
    return multi_location_plan

def _convert_to_vrp_format(req: MultiLocationRequest) -> Dict[str, Any]:
    """Convert multi-location request to standard VRP format"""
    
    # Find depot locations
    depots = [loc for loc in req.locations if loc.location_type == "depot"]
    if not depots:
        raise ValueError("At least one depot location is required")
    
    # Use first depot as main depot (could be enhanced for multi-depot)
    main_depot = depots[0]
    
    # Convert other locations to stops
    stops = []
    for loc in req.locations:
        if loc.location_type != "depot":
            stop = {
                "id": loc.id,
                "lat": loc.lat,
                "lng": loc.lng,
                "demand": abs(loc.demand),  # VRP uses positive demand
                "priority": loc.priority,
                "time_window": loc.time_window,
                "service_min": loc.service_min,
                "location_type": loc.location_type,
                "group_id": loc.group_id,
                "dependencies": loc.dependencies
            }
            stops.append(stop)
    
    # Convert vehicles
    vehicles = []
    for veh in req.vehicles:
        vehicle = {
            "id": veh.id,
            "capacity": veh.capacity,
            "fuel_type": veh.fuel_type,
            "max_stops": veh.max_stops,
            "allowed_location_types": veh.allowed_location_types,
            "start_location": veh.start_location,
            "end_location": veh.end_location
        }
        vehicles.append(vehicle)
    
    return {
        "depot": {
            "id": main_depot.id,
            "lat": main_depot.lat,
            "lng": main_depot.lng
        },
        "stops": stops,
        "vehicles": vehicles
    }

def _apply_multi_location_preset(req: MultiLocationRequest, vrp_request: Dict[str, Any]) -> Dict[str, Any]:
    """Apply preset configurations for different multi-location scenarios"""
    
    presets = {
        "pickup_delivery": {
            "time_limit_sec": 10,
            "drop_penalty_per_priority": 30000,  # High penalty to avoid dropping pairs
            "use_service_time_model": True,
            "use_warmstart": True,
            "use_access_analysis": True
        },
        "multi_depot": {
            "time_limit_sec": 12,
            "drop_penalty_per_priority": 20000,
            "use_service_time_model": True,
            "use_warmstart": True,
            "use_access_analysis": False
        },
        "service_routes": {
            "time_limit_sec": 6,
            "drop_penalty_per_priority": 15000,
            "use_service_time_model": True,
            "use_warmstart": False,
            "use_access_analysis": True
        }
    }
    
    preset_config = presets.get(req.preset, {})
    
    # Apply preset overrides
    for key, value in preset_config.items():
        if hasattr(req, key):
            setattr(req, key, value)
    
    return vrp_request

def _convert_from_vrp_format(plan: Dict[str, Any], req: MultiLocationRequest) -> Dict[str, Any]:
    """Convert VRP result back to multi-location format"""
    
    if not plan.get("ok"):
        return plan
    
    # Enhanced plan with multi-location information
    enhanced_plan = plan.copy()
    enhanced_plan["location_info"] = {
        "total_locations": len(req.locations),
        "location_types": {
            "depot": len([l for l in req.locations if l.location_type == "depot"]),
            "pickup": len([l for l in req.locations if l.location_type == "pickup"]),
            "delivery": len([l for l in req.locations if l.location_type == "delivery"]),
            "service": len([l for l in req.locations if l.location_type == "service"]),
            "waypoint": len([l for l in req.locations if l.location_type == "waypoint"])
        },
        "pickup_delivery_pairs": req.pickup_delivery_pairs,
        "enforced_sequences": req.sequences
    }
    
    # Add location type information to each route
    for route in enhanced_plan.get("routes", []):
        for stop in route.get("stops", []):
            if stop["node"] > 0:  # Not depot
                stop_idx = stop["node"] - 1
                if stop_idx < len(req.locations):
                    location = req.locations[stop_idx]
                    stop["location_type"] = location.location_type
                    stop["group_id"] = location.group_id
                    stop["dependencies"] = location.dependencies
    
    return enhanced_plan

@router.post("/pickup-delivery-routes")
async def optimize_pickup_delivery_routes(req: MultiLocationRequest):
    """
    Specialized endpoint for pickup and delivery optimization
    """
    req.preset = "pickup_delivery"
    return await optimize_multi_location_routes(req)

@router.post("/multi-depot-routes")
async def optimize_multi_depot_routes(req: MultiLocationRequest):
    """
    Specialized endpoint for multi-depot routing
    """
    req.preset = "multi_depot"
    return await optimize_multi_location_routes(req)



#!/usr/bin/env python3
"""
DATA EXTRACTION API
Comprehensive API endpoints for frontend data extraction
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uvicorn

# Add backend to path
sys.path.append("/Users/krishnabhatnagar/hackharvard/swarmaura/backend")

# Import backend services
try:
    from backend.services.ortools_solver import solve_vrp
    from backend.services.service_time_model import predictor_singleton
    from backend.services.risk_shaper import risk_shaper_singleton
    from backend.services.warmstart import warmstart_singleton
    from unified_data_system import UnifiedDataSystem
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend services not available: {e}")
    BACKEND_AVAILABLE = False

app = FastAPI(
    title="SwarmAura Data Extraction API",
    description="Comprehensive API for frontend data extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize unified data system
if BACKEND_AVAILABLE:
    uds = UnifiedDataSystem()

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SwarmAura Data Extraction API",
        "version": "1.0.0",
        "status": "operational",
        "backend_available": BACKEND_AVAILABLE,
        "endpoints": {
            "locations": "/api/locations",
            "vehicles": "/api/vehicles",
            "routes": "/api/routes",
            "analytics": "/api/analytics",
            "predictions": "/api/predictions",
            "environmental": "/api/environmental",
            "accessibility": "/api/accessibility",
            "health": "/api/health"
        }
    }

# ==================== LOCATION DATA ENDPOINTS ====================

@app.get("/api/locations")
async def get_all_locations():
    """Get all locations with complete data"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        locations = uds.master_data["locations"]
        return {
            "status": "success",
            "count": len(locations),
            "data": locations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/{location_id}")
async def get_location_by_id(location_id: str):
    """Get specific location by ID"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        locations = uds.master_data["locations"]
        location = next((loc for loc in locations if loc["id"] == location_id), None)
        
        if not location:
            raise HTTPException(status_code=404, detail="Location not found")
        
        return {
            "status": "success",
            "data": location,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/locations/type/{location_type}")
async def get_locations_by_type(location_type: str):
    """Get locations by type (depot, stop, etc.)"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        locations = uds.master_data["locations"]
        filtered_locations = [loc for loc in locations if loc.get("type") == location_type]
        
        return {
            "status": "success",
            "type": location_type,
            "count": len(filtered_locations),
            "data": filtered_locations,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== VEHICLE DATA ENDPOINTS ====================

@app.get("/api/vehicles")
async def get_all_vehicles():
    """Get all vehicles with complete data"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        vehicles = uds.master_data["vehicles"]
        return {
            "status": "success",
            "count": len(vehicles),
            "data": vehicles,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vehicles/{vehicle_id}")
async def get_vehicle_by_id(vehicle_id: str):
    """Get specific vehicle by ID"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        vehicles = uds.master_data["vehicles"]
        vehicle = next((veh for veh in vehicles if veh["id"] == vehicle_id), None)
        
        if not vehicle:
            raise HTTPException(status_code=404, detail="Vehicle not found")
        
        return {
            "status": "success",
            "data": vehicle,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/vehicles/type/{vehicle_type}")
async def get_vehicles_by_type(vehicle_type: str):
    """Get vehicles by type (truck, van, etc.)"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        vehicles = uds.master_data["vehicles"]
        filtered_vehicles = [veh for veh in vehicles if veh.get("type") == vehicle_type]
        
        return {
            "status": "success",
            "type": vehicle_type,
            "count": len(filtered_vehicles),
            "data": filtered_vehicles,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ROUTING DATA ENDPOINTS ====================

@app.post("/api/routes/optimize")
async def optimize_routes(route_request: Dict[str, Any]):
    """Optimize routes with given parameters"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        result = solve_vrp(
            depot=route_request["depot"],
            stops=route_request["stops"],
            vehicles=route_request["vehicles"],
            time_limit_sec=route_request.get("time_limit_sec", 10),
            drop_penalty_per_priority=route_request.get("drop_penalty_per_priority", 2000),
            use_access_scores=route_request.get("use_access_scores", True)
        )
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/routes/current")
async def get_current_routes():
    """Get current active routes"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        # Get routing data and solve
        routing_data = uds.get_routing_data()
        service_data = uds.get_service_time_data()
        
        # Add service times
        for stop in routing_data["stops"]:
            for service in service_data:
                if service["id"] == stop["id"]:
                    stop["service_min"] = service["historical_avg"]
                    break
        
        result = solve_vrp(
            depot=routing_data["depot"],
            stops=routing_data["stops"],
            vehicles=routing_data["vehicles"],
            time_limit_sec=10,
            drop_penalty_per_priority=2000,
            use_access_scores=True
        )
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ANALYTICS ENDPOINTS ====================

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get system analytics overview"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        master_data = uds.master_data
        
        # Calculate metrics
        total_locations = len(master_data["locations"])
        total_vehicles = len(master_data["vehicles"])
        total_demand = sum(loc.get("demand", 0) for loc in master_data["locations"])
        total_capacity = sum(veh["capacity"] for veh in master_data["vehicles"])
        capacity_utilization = (total_demand / total_capacity * 100) if total_capacity > 0 else 0
        
        return {
            "status": "success",
            "data": {
                "total_locations": total_locations,
                "total_vehicles": total_vehicles,
                "total_demand": total_demand,
                "total_capacity": total_capacity,
                "capacity_utilization": round(capacity_utilization, 2),
                "system_status": "operational",
                "last_updated": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance")
async def get_performance_metrics():
    """Get performance metrics"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        # Get routing data and solve for performance metrics
        routing_data = uds.get_routing_data()
        service_data = uds.get_service_time_data()
        
        # Add service times
        for stop in routing_data["stops"]:
            for service in service_data:
                if service["id"] == stop["id"]:
                    stop["service_min"] = service["historical_avg"]
                    break
        
        start_time = time.time()
        result = solve_vrp(
            depot=routing_data["depot"],
            stops=routing_data["stops"],
            vehicles=routing_data["vehicles"],
            time_limit_sec=10,
            drop_penalty_per_priority=2000,
            use_access_scores=True
        )
        solve_time = time.time() - start_time
        
        routes = result.get("routes", [])
        summary = result.get("summary", {})
        
        return {
            "status": "success",
            "data": {
                "solve_time": round(solve_time, 3),
                "total_routes": len(routes),
                "total_distance": summary.get("total_distance_km", 0),
                "total_time": summary.get("total_time_min", 0),
                "served_stops": summary.get("served_stops", 0),
                "served_rate": summary.get("served_rate", 0),
                "vehicle_efficiency": len([r for r in routes if r.get("stops", [])]) / len(routing_data["vehicles"]) * 100,
                "average_route_length": sum(len(r.get("stops", [])) for r in routes) / len(routes) if routes else 0
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== PREDICTIONS ENDPOINTS ====================

@app.get("/api/predictions/service-times")
async def get_service_time_predictions():
    """Get service time predictions"""
    if not BACKEND_AVAILABLE or not predictor_singleton:
        raise HTTPException(status_code=503, detail="Service time predictor not available")
    
    try:
        service_data = uds.get_service_time_data()
        predictions = predictor_singleton.predict_minutes(service_data)
        
        results = []
        for i, (service, pred) in enumerate(zip(service_data, predictions)):
            loc = next(loc for loc in uds.master_data["locations"] if loc["id"] == service["id"])
            confidence = 1.0 - (abs(pred - service["historical_avg"]) / service["historical_avg"])
            
            results.append({
                "location_id": service["id"],
                "location_name": loc["name"],
                "predicted_time": round(pred, 1),
                "historical_avg": round(service["historical_avg"], 1),
                "confidence": round(confidence, 2),
                "model_type": predictor_singleton.mode,
                "factors": {
                    "demand": service["demand"],
                    "access_score": service["access_score"],
                    "weather_risk": service["weather_risk"],
                    "traffic_risk": service["traffic_risk"],
                    "peak_hour": service["peak_hour"]
                }
            })
        
        return {
            "status": "success",
            "count": len(results),
            "data": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predictions/risk-assessment")
async def get_risk_assessment():
    """Get risk assessment for all routes"""
    if not BACKEND_AVAILABLE or not risk_shaper_singleton:
        raise HTTPException(status_code=503, detail="Risk shaper not available")
    
    try:
        locations = uds.master_data["locations"]
        stops_order = [loc["id"] for loc in locations]
        
        # Create OSRM matrix
        import numpy as np
        n = len(locations)
        osrm_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    osrm_matrix[i][j] = np.random.uniform(5, 25)
        
        # Features
        features = {}
        for loc in locations:
            features[loc["id"]] = {
                "risk": loc["crime_risk"],
                "light": loc["lighting_score"],
                "cong": loc["congestion_score"]
            }
        
        # Get risk multipliers
        multipliers = risk_shaper_singleton.shape(stops_order, osrm_matrix.tolist(), 14, 2, features)
        
        results = []
        for i, src in enumerate(locations):
            for j, dst in enumerate(locations):
                if i != j:
                    risk_mult = multipliers[i][j]
                    base_time = osrm_matrix[i][j]
                    adjusted_time = base_time * (1 + risk_mult)
                    
                    results.append({
                        "src_id": src["id"],
                        "src_name": src["name"],
                        "dst_id": dst["id"],
                        "dst_name": dst["name"],
                        "base_time": round(base_time, 1),
                        "risk_multiplier": round(risk_mult, 3),
                        "adjusted_time": round(adjusted_time, 1),
                        "time_increase": round(adjusted_time - base_time, 1),
                        "risk_level": "high" if risk_mult > 0.3 else "medium" if risk_mult > 0.2 else "low"
                    })
        
        # Sort by risk
        results.sort(key=lambda x: x["risk_multiplier"], reverse=True)
        
        return {
            "status": "success",
            "count": len(results),
            "data": results,
            "statistics": {
                "average_risk": round(np.mean([r["risk_multiplier"] for r in results]), 3),
                "max_risk": round(max([r["risk_multiplier"] for r in results]), 3),
                "high_risk_routes": len([r for r in results if r["risk_multiplier"] > 0.3])
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ENVIRONMENTAL DATA ENDPOINTS ====================

@app.get("/api/environmental/weather")
async def get_weather_data():
    """Get weather data"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        env_data = uds.get_environmental_data()
        weather = env_data["weather"]
        
        return {
            "status": "success",
            "data": {
                "temperature": weather["temperature"],
                "condition": weather["condition"],
                "humidity": weather["humidity"],
                "wind_speed": weather["wind_speed"],
                "visibility": weather["visibility"],
                "precipitation": weather["precipitation"],
                "pressure": weather.get("pressure", 1013.25),
                "uv_index": weather.get("uv_index", 0),
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/environmental/traffic")
async def get_traffic_data():
    """Get traffic data"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        env_data = uds.get_environmental_data()
        traffic = env_data["traffic"]
        
        return {
            "status": "success",
            "data": {
                "overall_congestion": traffic["overall_congestion"],
                "incidents": traffic["incidents"],
                "construction_zones": traffic["construction_zones"],
                "rush_hour_multiplier": traffic["rush_hour_multiplier"],
                "average_speed": traffic.get("average_speed", 30),
                "timestamp": datetime.now().isoformat()
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== ACCESSIBILITY DATA ENDPOINTS ====================

@app.get("/api/accessibility")
async def get_accessibility_data():
    """Get accessibility data for all locations"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        access_data = uds.get_accessibility_data()
        
        results = []
        for loc in access_data:
            features = loc["features"]
            hazards = loc["hazards"]
            
            # Calculate accessibility score
            base_score = loc["access_score"] * 100
            feature_bonus = 0
            if "elevator" in features:
                feature_bonus += 15
            if "ramp" in features:
                feature_bonus += 10
            if "wide_doors" in features:
                feature_bonus += 5
            
            sidewalk_bonus = min(10, loc["sidewalk_width"] * 2)
            curb_bonus = min(10, loc["curb_cuts"] * 2)
            hazard_penalty = len(hazards) * 10
            
            final_score = base_score + feature_bonus + sidewalk_bonus + curb_bonus - hazard_penalty
            final_score = max(0, min(100, final_score))
            
            results.append({
                "location_id": loc["id"],
                "base_score": round(base_score, 0),
                "feature_bonus": feature_bonus,
                "sidewalk_bonus": round(sidewalk_bonus, 1),
                "curb_bonus": curb_bonus,
                "hazard_penalty": hazard_penalty,
                "final_score": round(final_score, 0),
                "features": features,
                "hazards": hazards,
                "sidewalk_width": loc["sidewalk_width"],
                "curb_cuts": loc["curb_cuts"],
                "parking_spaces": loc["parking_spaces"],
                "loading_docks": loc["loading_docks"],
                "lighting_score": loc["lighting_score"],
                "accessibility_level": "excellent" if final_score >= 80 else "good" if final_score >= 60 else "poor"
            })
        
        return {
            "status": "success",
            "count": len(results),
            "data": results,
            "statistics": {
                "average_score": round(sum(r["final_score"] for r in results) / len(results), 1),
                "excellent_locations": len([r for r in results if r["final_score"] >= 80]),
                "good_locations": len([r for r in results if 60 <= r["final_score"] < 80]),
                "poor_locations": len([r for r in results if r["final_score"] < 60])
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HEALTH CHECK ENDPOINT ====================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend_available": BACKEND_AVAILABLE,
        "timestamp": datetime.now().isoformat(),
        "services": {
            "unified_data_system": BACKEND_AVAILABLE,
            "service_time_predictor": BACKEND_AVAILABLE and predictor_singleton is not None,
            "risk_shaper": BACKEND_AVAILABLE and risk_shaper_singleton is not None,
            "warmstart_clusterer": BACKEND_AVAILABLE and warmstart_singleton is not None
        }
    }

# ==================== BULK DATA ENDPOINT ====================

@app.get("/api/bulk/all")
async def get_all_data():
    """Get all data in one request"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        # Get all data
        master_data = uds.master_data
        env_data = uds.get_environmental_data()
        access_data = uds.get_accessibility_data()
        
        # Get service time predictions
        service_predictions = []
        if predictor_singleton:
            service_data = uds.get_service_time_data()
            predictions = predictor_singleton.predict_minutes(service_data)
            for i, (service, pred) in enumerate(zip(service_data, predictions)):
                loc = next(loc for loc in master_data["locations"] if loc["id"] == service["id"])
                confidence = 1.0 - (abs(pred - service["historical_avg"]) / service["historical_avg"])
                service_predictions.append({
                    "location_id": service["id"],
                    "location_name": loc["name"],
                    "predicted_time": round(pred, 1),
                    "confidence": round(confidence, 2)
                })
        
        return {
            "status": "success",
            "data": {
                "locations": master_data["locations"],
                "vehicles": master_data["vehicles"],
                "environmental": env_data,
                "accessibility": access_data,
                "service_predictions": service_predictions,
                "system_info": {
                    "total_locations": len(master_data["locations"]),
                    "total_vehicles": len(master_data["vehicles"]),
                    "total_demand": sum(loc.get("demand", 0) for loc in master_data["locations"]),
                    "total_capacity": sum(veh["capacity"] for veh in master_data["vehicles"])
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Change to project directory
    os.chdir("/Users/krishnabhatnagar/hackharvard/swarmaura")
    
    # Start the server
    uvicorn.run(
        "data_extraction_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

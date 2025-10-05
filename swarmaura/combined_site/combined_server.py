#!/usr/bin/env python3
"""
Combined Site Server
Serves both frontend and backend in one application
"""
import os
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

# Import backend services
try:
    from backend.services.ortools_solver import solve_vrp
    from backend.services.unified_data_system import UnifiedDataSystem
    BACKEND_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backend services not available: {e}")
    BACKEND_AVAILABLE = False

app = FastAPI(title="SwarmAura Combined Site", version="1.0.0")

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
async def serve_frontend():
    """Serve the combined frontend"""
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "backend_available": BACKEND_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/optimize")
async def optimize_routes(request: Request):
    """Optimize routes endpoint"""
    if not BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Backend services not available")
    
    try:
        data = await request.json()
        
        # Use unified data system for optimization
        result = solve_vrp(
            depot=data["depot"],
            stops=data["stops"],
            vehicles=data["vehicles"],
            time_limit_sec=10,
            drop_penalty_per_priority=2000,
            use_access_scores=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics():
    """Get system metrics"""
    if not BACKEND_AVAILABLE:
        return {"error": "Backend services not available"}
    
    try:
        # Get metrics from unified data system
        master_data = uds.master_data
        
        return {
            "total_locations": len(master_data["locations"]),
            "total_vehicles": len(master_data["vehicles"]),
            "total_demand": sum(loc.get("demand", 0) for loc in master_data["locations"]),
            "total_capacity": sum(veh["capacity"] for veh in master_data["vehicles"]),
            "system_status": "operational"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/vehicles")
async def get_vehicles():
    """Get vehicle data"""
    if not BACKEND_AVAILABLE:
        return {"vehicles": []}
    
    try:
        return {"vehicles": uds.master_data["vehicles"]}
    except Exception as e:
        return {"vehicles": [], "error": str(e)}

@app.get("/api/locations")
async def get_locations():
    """Get location data"""
    if not BACKEND_AVAILABLE:
        return {"locations": []}
    
    try:
        return {"locations": uds.master_data["locations"]}
    except Exception as e:
        return {"locations": [], "error": str(e)}

if __name__ == "__main__":
    # Change to combined site directory
    os.chdir(Path(__file__).parent)
    
    # Start the server
    uvicorn.run(
        "combined_server:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )

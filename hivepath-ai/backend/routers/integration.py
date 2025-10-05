
"""
API Integration Layer for Frontend-Backend Communication
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import httpx
import json

router = APIRouter(prefix="/api/v1/integration", tags=["integration"])

class FrontendBackendIntegration:
    def __init__(self):
        self.frontend_url = "https://fleet-flow-7189cccb.base44.app"
        self.backend_url = "http://localhost:8000"
    
    async def get_frontend_data(self) -> Dict[str, Any]:
        """Fetch data from frontend"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.frontend_url}/api/data")
                return response.json()
        except Exception as e:
            return {"error": str(e), "data": {}}
    
    async def sync_with_frontend(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync data with frontend"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.frontend_url}/api/sync",
                    json=data
                )
                return response.json()
        except Exception as e:
            return {"error": str(e), "success": False}

integration = FrontendBackendIntegration()

@router.get("/frontend-data")
async def get_frontend_data():
    """Get data from frontend"""
    return await integration.get_frontend_data()

@router.post("/sync")
async def sync_data(data: Dict[str, Any]):
    """Sync data with frontend"""
    return await integration.sync_with_frontend(data)

@router.get("/status")
async def get_integration_status():
    """Get integration status"""
    return {
        "frontend_url": integration.frontend_url,
        "backend_url": integration.backend_url,
        "status": "connected",
        "timestamp": datetime.now().isoformat()
    }

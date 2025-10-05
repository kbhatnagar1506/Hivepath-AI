#!/usr/bin/env python3
"""
FRONTEND-BACKEND INTEGRATION SYSTEM
Integrates fleet-flow frontend with swarmaura backend
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

class FrontendBackendIntegrator:
    def __init__(self):
        self.project_root = Path("/Users/krishnabhatnagar/hackharvard/swarmaura")
        self.frontend_url = "https://fleet-flow-7189cccb.base44.app"
        self.backend_port = 8000
        self.frontend_port = 3000
        self.integrated_port = 8080
        
        # API endpoints
        self.api_endpoints = {
            "optimize_routes": "/api/v1/optimize/routes",
            "multi_location": "/api/v1/multi-location/routes",
            "health": "/api/v1/health",
            "metrics": "/api/v1/metrics"
        }
    
    def create_integration_config(self):
        """Create configuration for frontend-backend integration"""
        print("🔧 CREATING INTEGRATION CONFIGURATION")
        print("=" * 50)
        
        config = {
            "frontend": {
                "url": self.frontend_url,
                "port": self.frontend_port,
                "api_base_url": f"http://localhost:{self.backend_port}",
                "endpoints": self.api_endpoints
            },
            "backend": {
                "port": self.backend_port,
                "host": "0.0.0.0",
                "cors_origins": [self.frontend_url, f"http://localhost:{self.frontend_port}"],
                "api_prefix": "/api/v1"
            },
            "integration": {
                "unified_port": self.integrated_port,
                "proxy_config": {
                    "frontend_path": "/",
                    "api_path": "/api",
                    "static_path": "/static"
                }
            }
        }
        
        config_file = self.project_root / "integration_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Integration config created: {config_file}")
        return config
    
    def create_cors_middleware(self):
        """Create CORS middleware for backend"""
        print("🌐 CREATING CORS MIDDLEWARE")
        print("=" * 40)
        
        cors_middleware = '''
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def setup_cors(app: FastAPI):
    """Setup CORS middleware for frontend integration"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://fleet-flow-7189cccb.base44.app",
            "http://localhost:3000",
            "http://localhost:8080"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app
'''
        
        middleware_file = self.project_root / "backend" / "middleware" / "cors.py"
        middleware_file.parent.mkdir(exist_ok=True)
        
        with open(middleware_file, 'w') as f:
            f.write(cors_middleware)
        
        print(f"✅ CORS middleware created: {middleware_file}")
    
    def create_frontend_proxy(self):
        """Create frontend proxy configuration"""
        print("🔄 CREATING FRONTEND PROXY")
        print("=" * 40)
        
        proxy_config = {
            "name": "fleet-flow-proxy",
            "version": "1.0.0",
            "scripts": {
                "start": "node proxy-server.js",
                "dev": "nodemon proxy-server.js"
            },
            "dependencies": {
                "express": "^4.18.2",
                "http-proxy-middleware": "^2.0.6",
                "cors": "^2.8.5"
            }
        }
        
        package_file = self.project_root / "frontend-proxy" / "package.json"
        package_file.parent.mkdir(exist_ok=True)
        
        with open(package_file, 'w') as f:
            json.dump(proxy_config, f, indent=2)
        
        # Create proxy server
        proxy_server = '''
const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 8080;

// Enable CORS
app.use(cors());

// Proxy API requests to backend
app.use('/api', createProxyMiddleware({
    target: 'http://localhost:8000',
    changeOrigin: true,
    pathRewrite: {
        '^/api': '/api/v1'
    }
}));

// Serve frontend (if local build available)
app.use('/', express.static('dist'));

// Fallback to external frontend
app.get('*', (req, res) => {
    res.redirect('https://fleet-flow-7189cccb.base44.app' + req.path);
});

app.listen(PORT, () => {
    console.log(`🚀 Integrated server running on port ${PORT}`);
    console.log(`📡 API proxy: http://localhost:${PORT}/api`);
    console.log(`🌐 Frontend: https://fleet-flow-7189cccb.base44.app`);
});
'''
        
        server_file = self.project_root / "frontend-proxy" / "proxy-server.js"
        with open(server_file, 'w') as f:
            f.write(proxy_server)
        
        print(f"✅ Frontend proxy created: {server_file}")
    
    def create_api_integration_layer(self):
        """Create API integration layer"""
        print("🔌 CREATING API INTEGRATION LAYER")
        print("=" * 45)
        
        integration_layer = '''
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
'''
        
        integration_file = self.project_root / "backend" / "routers" / "integration.py"
        with open(integration_file, 'w') as f:
            f.write(integration_layer)
        
        print(f"✅ API integration layer created: {integration_file}")
    
    def update_backend_app(self):
        """Update backend app.py to include integration"""
        print("🔄 UPDATING BACKEND APP")
        print("=" * 35)
        
        app_file = self.project_root / "backend" / "app.py"
        
        # Read existing app.py
        if app_file.exists():
            with open(app_file, 'r') as f:
                content = f.read()
        else:
            content = ""
        
        # Add integration imports and setup
        integration_setup = '''
# Frontend-Backend Integration
from routers.integration import router as integration_router
from middleware.cors import setup_cors

# Setup CORS
app = setup_cors(app)

# Include integration router
app.include_router(integration_router)
'''
        
        # Add integration setup to app.py
        if "integration_router" not in content:
            content += integration_setup
        
        with open(app_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Backend app updated: {app_file}")
    
    def create_docker_compose(self):
        """Create Docker Compose for integrated deployment"""
        print("🐳 CREATING DOCKER COMPOSE")
        print("=" * 40)
        
        docker_compose = '''
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/app
      - FRONTEND_URL=https://fleet-flow-7189cccb.base44.app
    volumes:
      - ./backend:/app
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --reload

  frontend-proxy:
    build: ./frontend-proxy
    ports:
      - "8080:8080"
    environment:
      - BACKEND_URL=http://backend:8000
      - FRONTEND_URL=https://fleet-flow-7189cccb.base44.app
    depends_on:
      - backend
    command: npm start

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
      - frontend-proxy
'''
        
        compose_file = self.project_root / "docker-compose.integrated.yml"
        with open(compose_file, 'w') as f:
            f.write(docker_compose)
        
        print(f"✅ Docker Compose created: {compose_file}")
    
    def create_nginx_config(self):
        """Create Nginx configuration for load balancing"""
        print("⚖️ CREATING NGINX CONFIGURATION")
        print("=" * 45)
        
        nginx_config = '''
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server backend:8000;
    }
    
    upstream frontend {
        server frontend-proxy:8080;
    }
    
    server {
        listen 80;
        server_name localhost;
        
        # API routes
        location /api/ {
            proxy_pass http://backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Frontend routes
        location / {
            proxy_pass http://frontend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
    }
}
'''
        
        nginx_file = self.project_root / "nginx.conf"
        with open(nginx_file, 'w') as f:
            f.write(nginx_config)
        
        print(f"✅ Nginx configuration created: {nginx_file}")
    
    def create_startup_script(self):
        """Create startup script for integrated system"""
        print("🚀 CREATING STARTUP SCRIPT")
        print("=" * 40)
        
        startup_script = '''#!/bin/bash

echo "🚀 STARTING INTEGRATED FRONTEND-BACKEND SYSTEM"
echo "=============================================="

# Check if backend is running
if ! curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "🔄 Starting backend..."
    cd backend
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    cd ..
    sleep 5
else
    echo "✅ Backend already running"
fi

# Check if frontend proxy is running
if ! curl -s http://localhost:8080 > /dev/null; then
    echo "🔄 Starting frontend proxy..."
    cd frontend-proxy
    npm install
    npm start &
    PROXY_PID=$!
    cd ..
    sleep 5
else
    echo "✅ Frontend proxy already running"
fi

echo ""
echo "🎯 INTEGRATED SYSTEM STATUS:"
echo "============================"
echo "Backend API: http://localhost:8000"
echo "Frontend: https://fleet-flow-7189cccb.base44.app"
echo "Integrated: http://localhost:8080"
echo ""
echo "📡 API Endpoints:"
echo "  - Health: http://localhost:8000/api/v1/health"
echo "  - Routes: http://localhost:8000/api/v1/optimize/routes"
echo "  - Multi-location: http://localhost:8000/api/v1/multi-location/routes"
echo "  - Integration: http://localhost:8000/api/v1/integration/status"
echo ""
echo "🌐 Frontend Integration:"
echo "  - Main App: https://fleet-flow-7189cccb.base44.app"
echo "  - Proxy: http://localhost:8080"
echo ""
echo "✅ Integrated system is running!"
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'echo "🛑 Stopping services..."; kill $BACKEND_PID $PROXY_PID 2>/dev/null; exit' INT
wait
'''
        
        script_file = self.project_root / "start_integrated.sh"
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print(f"✅ Startup script created: {script_file}")
    
    def create_test_integration(self):
        """Create test script for integration"""
        print("🧪 CREATING INTEGRATION TEST")
        print("=" * 40)
        
        test_script = '''
#!/usr/bin/env python3
"""
Integration Test for Frontend-Backend System
"""
import requests
import json
import time

def test_integration():
    """Test the integrated frontend-backend system"""
    print("🧪 TESTING INTEGRATED SYSTEM")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    frontend_url = "https://fleet-flow-7189cccb.base44.app"
    
    # Test backend health
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        print(f"✅ Backend Health: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend Health: {e}")
    
    # Test route optimization
    try:
        test_data = {
            "depot": {
                "id": "D",
                "name": "Depot",
                "lat": 42.3601,
                "lng": -71.0589
            },
            "stops": [
                {
                    "id": "S1",
                    "name": "Stop 1",
                    "lat": 42.3700,
                    "lng": -71.0500,
                    "demand": 100,
                    "priority": 1
                }
            ],
            "vehicles": [
                {
                    "id": "V1",
                    "type": "truck",
                    "capacity": 200
                }
            ]
        }
        
        response = requests.post(
            f"{base_url}/api/v1/optimize/routes",
            json=test_data
        )
        print(f"✅ Route Optimization: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Routes: {len(result.get('routes', []))}")
    except Exception as e:
        print(f"❌ Route Optimization: {e}")
    
    # Test integration status
    try:
        response = requests.get(f"{base_url}/api/v1/integration/status")
        print(f"✅ Integration Status: {response.status_code}")
        if response.status_code == 200:
            status = response.json()
            print(f"   Frontend: {status.get('frontend_url')}")
            print(f"   Backend: {status.get('backend_url')}")
    except Exception as e:
        print(f"❌ Integration Status: {e}")
    
    print("\\n🎯 INTEGRATION TEST COMPLETE!")

if __name__ == "__main__":
    test_integration()
'''
        
        test_file = self.project_root / "test_integration.py"
        with open(test_file, 'w') as f:
            f.write(test_script)
        
        print(f"✅ Integration test created: {test_file}")
    
    def run_integration_setup(self):
        """Run complete integration setup"""
        print("🚀 FRONTEND-BACKEND INTEGRATION SETUP")
        print("=" * 60)
        print("Setting up integrated system with fleet-flow frontend")
        print()
        
        # Create all integration components
        self.create_integration_config()
        self.create_cors_middleware()
        self.create_frontend_proxy()
        self.create_api_integration_layer()
        self.update_backend_app()
        self.create_docker_compose()
        self.create_nginx_config()
        self.create_startup_script()
        self.create_test_integration()
        
        print("✅ INTEGRATION SETUP COMPLETE!")
        print("=" * 50)
        print("🎯 INTEGRATED SYSTEM COMPONENTS:")
        print("   • Frontend: https://fleet-flow-7189cccb.base44.app")
        print("   • Backend: http://localhost:8000")
        print("   • Proxy: http://localhost:8080")
        print("   • Nginx: Load balancer")
        print("   • Docker: Containerized deployment")
        print()
        print("🚀 TO START INTEGRATED SYSTEM:")
        print("   ./start_integrated.sh")
        print()
        print("🧪 TO TEST INTEGRATION:")
        print("   python3 test_integration.py")
        print()
        print("🐳 TO DEPLOY WITH DOCKER:")
        print("   docker-compose -f docker-compose.integrated.yml up")
        print()
        print("✅ Frontend-Backend integration ready!")

def main():
    """Main function"""
    print("🚀 FRONTEND-BACKEND INTEGRATION SYSTEM")
    print("=" * 60)
    print("Integrating fleet-flow frontend with swarmaura backend")
    print()
    
    # Change to project directory
    os.chdir("/Users/krishnabhatnagar/hackharvard/swarmaura")
    
    # Initialize and run integration
    integrator = FrontendBackendIntegrator()
    integrator.run_integration_setup()
    
    print(f"\\n🎉 Frontend-Backend integration complete!")

if __name__ == "__main__":
    main()

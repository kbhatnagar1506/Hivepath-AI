#!/usr/bin/env python3
"""
COMBINED SITE SYSTEM
Creates a unified web application combining frontend and backend
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

class CombinedSiteBuilder:
    def __init__(self):
        self.project_root = Path("/Users/krishnabhatnagar/hackharvard/swarmaura")
        self.combined_dir = self.project_root / "combined_site"
        self.combined_dir.mkdir(exist_ok=True)
        
        # API endpoints
        self.api_endpoints = {
            "health": "/api/health",
            "optimize": "/api/optimize",
            "multi_location": "/api/multi-location",
            "metrics": "/api/metrics"
        }
    
    def create_combined_html(self):
        """Create the main combined HTML page"""
        print("🌐 CREATING COMBINED HTML INTERFACE")
        print("=" * 50)
        
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SwarmAura - Fleet Management & Route Optimization</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-hover { transition: transform 0.3s ease, box-shadow 0.3s ease; }
        .card-hover:hover { transform: translateY(-5px); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1); }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
        .status-online { background-color: #10b981; }
        .status-offline { background-color: #ef4444; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <!-- Navigation -->
    <nav class="gradient-bg shadow-lg">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div class="flex justify-between h-16">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <h1 class="text-white text-xl font-bold">SwarmAura</h1>
                    </div>
                    <div class="ml-10 flex items-baseline space-x-4">
                        <a href="#dashboard" class="text-white hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded-md text-sm font-medium">Dashboard</a>
                        <a href="#routes" class="text-white hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded-md text-sm font-medium">Routes</a>
                        <a href="#vehicles" class="text-white hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded-md text-sm font-medium">Vehicles</a>
                        <a href="#analytics" class="text-white hover:bg-white hover:bg-opacity-20 px-3 py-2 rounded-md text-sm font-medium">Analytics</a>
                    </div>
                </div>
                <div class="flex items-center">
                    <div class="flex items-center space-x-2">
                        <span class="status-indicator status-online" id="status-indicator"></span>
                        <span class="text-white text-sm" id="status-text">System Online</span>
                    </div>
                </div>
            </div>
        </div>
    </nav>

    <!-- Main Content -->
    <div class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <!-- Dashboard Section -->
        <div id="dashboard" class="mb-8">
            <div class="bg-white overflow-hidden shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h2 class="text-lg font-medium text-gray-900 mb-4">System Dashboard</h2>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="bg-blue-50 p-4 rounded-lg">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-route text-blue-600 text-2xl"></i>
                                </div>
                                <div class="ml-4">
                                    <p class="text-sm font-medium text-gray-500">Active Routes</p>
                                    <p class="text-2xl font-semibold text-gray-900" id="active-routes">4</p>
                                </div>
                            </div>
                        </div>
                        <div class="bg-green-50 p-4 rounded-lg">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-truck text-green-600 text-2xl"></i>
                                </div>
                                <div class="ml-4">
                                    <p class="text-sm font-medium text-gray-500">Vehicles</p>
                                    <p class="text-2xl font-semibold text-gray-900" id="total-vehicles">4</p>
                                </div>
                            </div>
                        </div>
                        <div class="bg-yellow-50 p-4 rounded-lg">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-map-marker-alt text-yellow-600 text-2xl"></i>
                                </div>
                                <div class="ml-4">
                                    <p class="text-sm font-medium text-gray-500">Locations</p>
                                    <p class="text-2xl font-semibold text-gray-900" id="total-locations">6</p>
                                </div>
                            </div>
                        </div>
                        <div class="bg-purple-50 p-4 rounded-lg">
                            <div class="flex items-center">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-chart-line text-purple-600 text-2xl"></i>
                                </div>
                                <div class="ml-4">
                                    <p class="text-sm font-medium text-gray-500">Efficiency</p>
                                    <p class="text-2xl font-semibold text-gray-900" id="efficiency">95%</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Route Optimization Section -->
        <div id="routes" class="mb-8">
            <div class="bg-white shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h2 class="text-lg font-medium text-gray-900 mb-4">Route Optimization</h2>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                            <h3 class="text-md font-medium text-gray-700 mb-3">Optimize Routes</h3>
                            <form id="route-form" class="space-y-4">
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">Depot Location</label>
                                    <input type="text" id="depot-name" placeholder="Depot Name" class="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                                </div>
                                <div class="grid grid-cols-2 gap-4">
                                    <input type="number" id="depot-lat" placeholder="Latitude" step="0.000001" class="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                                    <input type="number" id="depot-lng" placeholder="Longitude" step="0.000001" class="block w-full border-gray-300 rounded-md shadow-sm focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm">
                                </div>
                                <button type="submit" class="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
                                    <i class="fas fa-route mr-2"></i>Optimize Routes
                                </button>
                            </form>
                        </div>
                        <div>
                            <h3 class="text-md font-medium text-gray-700 mb-3">Route Results</h3>
                            <div id="route-results" class="bg-gray-50 p-4 rounded-lg min-h-64">
                                <p class="text-gray-500 text-center">Click "Optimize Routes" to see results</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Vehicle Management Section -->
        <div id="vehicles" class="mb-8">
            <div class="bg-white shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h2 class="text-lg font-medium text-gray-900 mb-4">Vehicle Management</h2>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-50">
                                <tr>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Vehicle</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Type</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Capacity</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Capabilities</th>
                                    <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                                </tr>
                            </thead>
                            <tbody id="vehicle-table" class="bg-white divide-y divide-gray-200">
                                <!-- Vehicle data will be loaded here -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Analytics Section -->
        <div id="analytics" class="mb-8">
            <div class="bg-white shadow rounded-lg">
                <div class="px-4 py-5 sm:p-6">
                    <h2 class="text-lg font-medium text-gray-900 mb-4">Analytics & Performance</h2>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div>
                            <h3 class="text-md font-medium text-gray-700 mb-3">Route Performance</h3>
                            <canvas id="performance-chart" width="400" height="200"></canvas>
                        </div>
                        <div>
                            <h3 class="text-md font-medium text-gray-700 mb-3">System Metrics</h3>
                            <div id="metrics-container" class="space-y-3">
                                <!-- Metrics will be loaded here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Scripts -->
    <script>
        // Global variables
        let performanceChart = null;
        let systemStatus = 'online';

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initializeApp();
            loadVehicleData();
            loadSystemMetrics();
            setupEventListeners();
        });

        // Initialize application
        function initializeApp() {
            console.log('🚀 SwarmAura Combined Site Initialized');
            updateSystemStatus();
            setInterval(updateSystemStatus, 30000); // Update every 30 seconds
        }

        // Update system status
        async function updateSystemStatus() {
            try {
                const response = await fetch('/api/health');
                if (response.ok) {
                    systemStatus = 'online';
                    document.getElementById('status-indicator').className = 'status-indicator status-online';
                    document.getElementById('status-text').textContent = 'System Online';
                } else {
                    throw new Error('API not responding');
                }
            } catch (error) {
                systemStatus = 'offline';
                document.getElementById('status-indicator').className = 'status-indicator status-offline';
                document.getElementById('status-text').textContent = 'System Offline';
            }
        }

        // Load vehicle data
        async function loadVehicleData() {
            const vehicles = [
                { id: 'V1', type: 'Truck', capacity: 400, capabilities: ['lift_gate', 'refrigeration', 'hazmat'], status: 'Active' },
                { id: 'V2', type: 'Van', capacity: 200, capabilities: ['lift_gate'], status: 'Active' },
                { id: 'V3', type: 'Truck', capacity: 350, capabilities: ['lift_gate', 'refrigeration'], status: 'Active' },
                { id: 'V4', type: 'Van', capacity: 150, capabilities: ['standard'], status: 'Active' }
            ];

            const tbody = document.getElementById('vehicle-table');
            tbody.innerHTML = vehicles.map(vehicle => `
                <tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${vehicle.id}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${vehicle.type}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${vehicle.capacity} units</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${vehicle.capabilities.join(', ')}</td>
                    <td class="px-6 py-4 whitespace-nowrap">
                        <span class="inline-flex px-2 py-1 text-xs font-semibold rounded-full bg-green-100 text-green-800">${vehicle.status}</span>
                    </td>
                </tr>
            `).join('');
        }

        // Load system metrics
        async function loadSystemMetrics() {
            const metrics = [
                { label: 'Total Distance', value: '15.12 km', icon: 'fas fa-route' },
                { label: 'Average Route Time', value: '8.5 min', icon: 'fas fa-clock' },
                { label: 'Capacity Utilization', value: '65.9%', icon: 'fas fa-chart-pie' },
                { label: 'Vehicle Efficiency', value: '100%', icon: 'fas fa-truck' },
                { label: 'AI Predictions', value: '14.1 min avg', icon: 'fas fa-brain' },
                { label: 'Risk Assessment', value: '0.313 avg', icon: 'fas fa-exclamation-triangle' }
            ];

            const container = document.getElementById('metrics-container');
            container.innerHTML = metrics.map(metric => `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center">
                        <i class="${metric.icon} text-indigo-600 mr-3"></i>
                        <span class="text-sm font-medium text-gray-700">${metric.label}</span>
                    </div>
                    <span class="text-sm font-semibold text-gray-900">${metric.value}</span>
                </div>
            `).join('');

            // Initialize performance chart
            initializePerformanceChart();
        }

        // Initialize performance chart
        function initializePerformanceChart() {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: ['Route 1', 'Route 2', 'Route 3', 'Route 4'],
                    datasets: [{
                        label: 'Distance (km)',
                        data: [6.98, 0.00, 6.82, 1.32],
                        borderColor: 'rgb(99, 102, 241)',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.1
                    }, {
                        label: 'Load (%)',
                        data: [71.2, 0.0, 82.9, 100.0],
                        borderColor: 'rgb(16, 185, 129)',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }

        // Setup event listeners
        function setupEventListeners() {
            document.getElementById('route-form').addEventListener('submit', handleRouteOptimization);
        }

        // Handle route optimization
        async function handleRouteOptimization(event) {
            event.preventDefault();
            
            const depotName = document.getElementById('depot-name').value || 'Boston Depot';
            const depotLat = parseFloat(document.getElementById('depot-lat').value) || 42.3601;
            const depotLng = parseFloat(document.getElementById('depot-lng').value) || -71.0589;

            const routeData = {
                depot: {
                    id: 'D',
                    name: depotName,
                    lat: depotLat,
                    lng: depotLng
                },
                stops: [
                    { id: 'S1', name: 'Back Bay Station', lat: 42.3700, lng: -71.0500, demand: 150, priority: 2 },
                    { id: 'S2', name: 'North End', lat: 42.3400, lng: -71.1000, demand: 140, priority: 1 },
                    { id: 'S3', name: 'Harvard Square', lat: 42.3900, lng: -71.0200, demand: 145, priority: 2 },
                    { id: 'S4', name: 'Beacon Hill', lat: 42.3300, lng: -71.0600, demand: 150, priority: 1 },
                    { id: 'S5', name: 'South End', lat: 42.4100, lng: -71.0300, demand: 140, priority: 2 }
                ],
                vehicles: [
                    { id: 'V1', type: 'truck', capacity: 400 },
                    { id: 'V2', type: 'van', capacity: 200 },
                    { id: 'V3', type: 'truck', capacity: 350 },
                    { id: 'V4', type: 'van', capacity: 150 }
                ]
            };

            try {
                const response = await fetch('/api/optimize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(routeData)
                });

                if (response.ok) {
                    const result = await response.json();
                    displayRouteResults(result);
                } else {
                    throw new Error('Optimization failed');
                }
            } catch (error) {
                displayRouteResults({ error: 'Failed to optimize routes. Please check system status.' });
            }
        }

        // Display route results
        function displayRouteResults(result) {
            const container = document.getElementById('route-results');
            
            if (result.error) {
                container.innerHTML = `
                    <div class="text-red-600 text-center">
                        <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                        <p>${result.error}</p>
                    </div>
                `;
                return;
            }

            const routes = result.routes || [];
            const summary = result.summary || {};

            container.innerHTML = `
                <div class="space-y-4">
                    <div class="bg-blue-50 p-4 rounded-lg">
                        <h4 class="font-semibold text-blue-900 mb-2">Optimization Summary</h4>
                        <div class="grid grid-cols-2 gap-4 text-sm">
                            <div><span class="font-medium">Total Distance:</span> ${summary.total_distance_km || 'N/A'} km</div>
                            <div><span class="font-medium">Total Time:</span> ${summary.total_time_min || 'N/A'} min</div>
                            <div><span class="font-medium">Served Stops:</span> ${summary.served_stops || 'N/A'}</div>
                            <div><span class="font-medium">Served Rate:</span> ${summary.served_rate || 'N/A'}</div>
                        </div>
                    </div>
                    <div class="space-y-2">
                        <h4 class="font-semibold text-gray-900">Route Details</h4>
                        ${routes.map((route, index) => `
                            <div class="bg-gray-50 p-3 rounded-lg">
                                <div class="flex justify-between items-center mb-2">
                                    <span class="font-medium">Route ${index + 1}</span>
                                    <span class="text-sm text-gray-600">${route.distance_km || 0} km</span>
                                </div>
                                <div class="text-sm text-gray-600">
                                    Stops: ${route.stops ? route.stops.length : 0} | 
                                    Load: ${route.load || 0} units
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    </script>
</body>
</html>'''
        
        html_file = self.combined_dir / "index.html"
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Combined HTML created: {html_file}")
    
    def create_combined_server(self):
        """Create the combined server"""
        print("🚀 CREATING COMBINED SERVER")
        print("=" * 40)
        
        server_code = '''#!/usr/bin/env python3
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
'''
        
        server_file = self.combined_dir / "combined_server.py"
        with open(server_file, 'w') as f:
            f.write(server_code)
        
        # Make executable
        os.chmod(server_file, 0o755)
        
        print(f"✅ Combined server created: {server_file}")
    
    def create_requirements(self):
        """Create requirements file for combined site"""
        print("📦 CREATING REQUIREMENTS")
        print("=" * 35)
        
        requirements = '''fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
pydantic==2.5.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
ortools==9.7.2996
torch==2.0.1
torch-geometric==2.4.0
googlemaps==4.10.0
opencv-python==4.8.1.78
requests==2.31.0
'''
        
        req_file = self.combined_dir / "requirements.txt"
        with open(req_file, 'w') as f:
            f.write(requirements)
        
        print(f"✅ Requirements created: {req_file}")
    
    def create_startup_script(self):
        """Create startup script for combined site"""
        print("🚀 CREATING STARTUP SCRIPT")
        print("=" * 40)
        
        startup_script = '''#!/bin/bash

echo "🚀 STARTING COMBINED SITE"
echo "========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "🔄 Installing requirements..."
pip install -r requirements.txt

echo "🚀 Starting combined site server..."
echo "🌐 Frontend: http://localhost:8080"
echo "📡 API: http://localhost:8080/api"
echo ""

python3 combined_server.py
'''
        
        script_file = self.combined_dir / "start_combined.sh"
        with open(script_file, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_file, 0o755)
        
        print(f"✅ Startup script created: {script_file}")
    
    def create_dockerfile(self):
        """Create Dockerfile for combined site"""
        print("🐳 CREATING DOCKERFILE")
        print("=" * 35)
        
        dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Start the application
CMD ["python3", "combined_server.py"]
'''
        
        docker_file = self.combined_dir / "Dockerfile"
        with open(docker_file, 'w') as f:
            f.write(dockerfile)
        
        print(f"✅ Dockerfile created: {docker_file}")
    
    def build_combined_site(self):
        """Build the complete combined site"""
        print("🏗️ BUILDING COMBINED SITE")
        print("=" * 40)
        
        # Create all components
        self.create_combined_html()
        self.create_combined_server()
        self.create_requirements()
        self.create_startup_script()
        self.create_dockerfile()
        
        print("✅ COMBINED SITE BUILT SUCCESSFULLY!")
        print("=" * 50)
        print("🎯 COMBINED SITE FEATURES:")
        print("   • Unified Frontend & Backend")
        print("   • Modern Web Interface")
        print("   • Real-time Route Optimization")
        print("   • Vehicle Management")
        print("   • Analytics Dashboard")
        print("   • API Integration")
        print()
        print("🚀 TO START COMBINED SITE:")
        print("   cd combined_site")
        print("   ./start_combined.sh")
        print()
        print("🐳 TO BUILD DOCKER IMAGE:")
        print("   cd combined_site")
        print("   docker build -t swarmaura-combined .")
        print("   docker run -p 8080:8080 swarmaura-combined")
        print()
        print("🌐 ACCESS COMBINED SITE:")
        print("   http://localhost:8080")
        print()
        print("✅ Combined site ready for deployment!")

def main():
    """Main function"""
    print("🏗️ COMBINED SITE BUILDER")
    print("=" * 50)
    print("Building unified frontend-backend application")
    print()
    
    # Change to project directory
    os.chdir("/Users/krishnabhatnagar/hackharvard/swarmaura")
    
    # Initialize and build combined site
    builder = CombinedSiteBuilder()
    builder.build_combined_site()
    
    print(f"\\n🎉 Combined site build complete!")

if __name__ == "__main__":
    main()

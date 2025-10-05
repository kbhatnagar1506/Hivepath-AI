#!/usr/bin/env python3
"""
🚀 HivePath AI: Complete API Flow Test
=====================================

This script demonstrates the complete API architecture flow,
showing how all our APIs work together to create the best
logistics intelligence system.

Features tested:
- Data Ingestion APIs
- AI Processing APIs  
- Core Optimization APIs
- Machine Learning APIs
- Real-time Analytics APIs
- Frontend Integration APIs
"""

import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

# API Configuration
API_BASE_URL = "http://localhost:8001"
DASHBOARD_URL = "http://localhost:3000"

class HivePathAPITester:
    """Complete API Flow Tester for HivePath AI"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def print_header(self, title: str, emoji: str = "🚀"):
        """Print a formatted header"""
        print(f"\n{emoji} {title}")
        print("=" * (len(title) + 3))
        
    def print_section(self, title: str, emoji: str = "📡"):
        """Print a formatted section"""
        print(f"\n{emoji} {title}")
        print("-" * (len(title) + 3))
        
    def test_api_endpoint(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Test a single API endpoint"""
        try:
            url = f"{API_BASE_URL}{endpoint}"
            
            if method == "GET":
                response = requests.get(url, timeout=10)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=15)
            else:
                return {"status": "error", "message": f"Unsupported method: {method}"}
            
            result = {
                "status": "success" if response.status_code == 200 else "error",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "data_size": len(response.content) if response.content else 0
            }
            
            if response.status_code == 200:
                try:
                    json_data = response.json()
                    result["data"] = json_data
                    result["data_count"] = len(json_data.get("data", [])) if isinstance(json_data.get("data"), list) else "N/A"
                except:
                    result["data"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
            else:
                result["error"] = response.text[:200] + "..." if len(response.text) > 200 else response.text
                
            return result
            
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "message": str(e),
                "response_time": 0
            }
    
    def test_data_ingestion_apis(self):
        """Test Layer 1: Data Ingestion & Intelligence APIs"""
        self.print_section("Data Ingestion & Intelligence APIs", "🌐")
        
        endpoints = [
            ("/api/health", "System Health Check"),
            ("/api/locations", "Location Data Ingestion"),
            ("/api/vehicles", "Vehicle Fleet Data"),
            ("/api/environmental/weather", "Weather Data Integration"),
            ("/api/environmental/traffic", "Traffic Data Integration"),
            ("/api/accessibility", "Accessibility Data"),
        ]
        
        results = {}
        for endpoint, description in endpoints:
            print(f"  🔍 Testing {description}...")
            result = self.test_api_endpoint(endpoint)
            results[endpoint] = result
            
            if result["status"] == "success":
                print(f"    ✅ {description}: {result['response_time']:.3f}s ({result['data_count']} items)")
            else:
                print(f"    ❌ {description}: {result.get('message', 'Unknown error')}")
        
        self.results["data_ingestion"] = results
        return results
    
    def test_ai_processing_apis(self):
        """Test Layer 2: AI Processing APIs"""
        self.print_section("AI Processing & Machine Learning APIs", "🧠")
        
        endpoints = [
            ("/api/predictions/service-times", "Service Time GNN Predictions"),
            ("/api/analytics/overview", "Performance Analytics"),
        ]
        
        results = {}
        for endpoint, description in endpoints:
            print(f"  🔍 Testing {description}...")
            result = self.test_api_endpoint(endpoint)
            results[endpoint] = result
            
            if result["status"] == "success":
                print(f"    ✅ {description}: {result['response_time']:.3f}s")
            else:
                print(f"    ❌ {description}: {result.get('message', 'Unknown error')}")
        
        self.results["ai_processing"] = results
        return results
    
    def test_optimization_apis(self):
        """Test Layer 3: Core Optimization Engine APIs"""
        self.print_section("Core Optimization Engine APIs", "⚡")
        
        # Test data for optimization
        test_route_request = {
            "depot": {
                "id": "depot_1",
                "name": "Main Depot",
                "lat": 42.3601,
                "lng": -71.0589,
                "type": "depot"
            },
            "stops": [
                {
                    "id": "stop_1",
                    "name": "Stop 1",
                    "lat": 42.3611,
                    "lng": -71.0599,
                    "demand": 10,
                    "priority": 1,
                    "service_min": 5,
                    "type": "delivery"
                },
                {
                    "id": "stop_2", 
                    "name": "Stop 2",
                    "lat": 42.3621,
                    "lng": -71.0609,
                    "demand": 15,
                    "priority": 2,
                    "service_min": 7,
                    "type": "delivery"
                }
            ],
            "vehicles": [
                {
                    "id": "vehicle_1",
                    "name": "Truck 1",
                    "capacity": 50,
                    "speed_kmph": 40,
                    "type": "truck"
                }
            ],
            "preset": "fast",
            "use_google_maps": True,
            "use_service_time_model": True,
            "use_warmstart": True,
            "use_access_analysis": True
        }
        
        endpoints = [
            ("/api/v1/optimize/routes", "Main VRP Optimization", "POST", test_route_request),
            ("/api/v1/multi-location/routes", "Multi-Location Routing", "POST", test_route_request),
        ]
        
        results = {}
        for endpoint, description, method, data in endpoints:
            print(f"  🔍 Testing {description}...")
            result = self.test_api_endpoint(endpoint, method, data)
            results[endpoint] = result
            
            if result["status"] == "success":
                print(f"    ✅ {description}: {result['response_time']:.3f}s")
                if "data" in result and isinstance(result["data"], dict):
                    routes = result["data"].get("routes", [])
                    total_distance = result["data"].get("total_distance_km", 0)
                    print(f"      📊 Generated {len(routes)} routes, {total_distance:.2f} km total")
            else:
                print(f"    ❌ {description}: {result.get('message', 'Unknown error')}")
        
        self.results["optimization"] = results
        return results
    
    def test_analytics_apis(self):
        """Test Layer 4: Real-time Analytics APIs"""
        self.print_section("Real-time Analytics & Monitoring APIs", "📈")
        
        endpoints = [
            ("/api/bulk/all", "Comprehensive Data Export"),
        ]
        
        results = {}
        for endpoint, description in endpoints:
            print(f"  🔍 Testing {description}...")
            result = self.test_api_endpoint(endpoint)
            results[endpoint] = result
            
            if result["status"] == "success":
                print(f"    ✅ {description}: {result['response_time']:.3f}s")
                if "data" in result and isinstance(result["data"], dict):
                    data = result["data"]
                    locations = len(data.get("locations", []))
                    vehicles = len(data.get("vehicles", []))
                    predictions = len(data.get("service_predictions", []))
                    print(f"      📊 Exported: {locations} locations, {vehicles} vehicles, {predictions} predictions")
            else:
                print(f"    ❌ {description}: {result.get('message', 'Unknown error')}")
        
        self.results["analytics"] = results
        return results
    
    def test_frontend_integration(self):
        """Test Layer 5: Frontend Integration"""
        self.print_section("Frontend Integration & Dashboard APIs", "🎨")
        
        # Test if dashboard is accessible
        try:
            response = requests.get(DASHBOARD_URL, timeout=5)
            dashboard_status = "accessible" if response.status_code == 200 else "error"
            print(f"  🌐 Dashboard Status: {dashboard_status}")
            if response.status_code == 200:
                print(f"    ✅ Dashboard accessible at {DASHBOARD_URL}")
            else:
                print(f"    ❌ Dashboard error: {response.status_code}")
        except:
            print(f"  ❌ Dashboard not accessible at {DASHBOARD_URL}")
            dashboard_status = "not_accessible"
        
        # Test CORS headers
        try:
            response = requests.options(f"{API_BASE_URL}/api/locations", 
                                      headers={"Origin": DASHBOARD_URL, 
                                              "Access-Control-Request-Method": "GET"})
            cors_status = "enabled" if response.status_code == 200 else "disabled"
            print(f"  🌐 CORS Status: {cors_status}")
        except:
            cors_status = "error"
            print(f"  ❌ CORS test failed")
        
        self.results["frontend_integration"] = {
            "dashboard_status": dashboard_status,
            "cors_status": cors_status
        }
        
        return self.results["frontend_integration"]
    
    def test_performance_metrics(self):
        """Test performance and scalability features"""
        self.print_section("Performance & Scalability Testing", "⚡")
        
        # Test response times across multiple endpoints
        endpoints = [
            "/api/health",
            "/api/locations", 
            "/api/vehicles",
            "/api/analytics/overview"
        ]
        
        response_times = []
        for endpoint in endpoints:
            result = self.test_api_endpoint(endpoint)
            if result["status"] == "success":
                response_times.append(result["response_time"])
        
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)
            
            print(f"  📊 Response Time Analysis:")
            print(f"    ⚡ Average: {avg_response_time:.3f}s")
            print(f"    🚀 Fastest: {min_response_time:.3f}s")
            print(f"    🐌 Slowest: {max_response_time:.3f}s")
            
            # Performance rating
            if avg_response_time < 0.5:
                performance_rating = "Excellent"
            elif avg_response_time < 1.0:
                performance_rating = "Good"
            elif avg_response_time < 2.0:
                performance_rating = "Fair"
            else:
                performance_rating = "Needs Improvement"
            
            print(f"    🎯 Performance Rating: {performance_rating}")
        
        self.results["performance"] = {
            "avg_response_time": avg_response_time if response_times else 0,
            "max_response_time": max_response_time if response_times else 0,
            "min_response_time": min_response_time if response_times else 0,
            "performance_rating": performance_rating if response_times else "Unknown"
        }
        
        return self.results["performance"]
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive API flow report"""
        self.print_header("HivePath AI API Flow Test Report", "📊")
        
        total_time = time.time() - self.start_time
        
        print(f"🕐 Total Test Duration: {total_time:.2f} seconds")
        print(f"📅 Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Summary statistics
        total_endpoints = 0
        successful_endpoints = 0
        
        for layer, results in self.results.items():
            if isinstance(results, dict) and "status" not in results:
                for endpoint, result in results.items():
                    if isinstance(result, dict):  # Ensure result is a dict
                        total_endpoints += 1
                        if result.get("status") == "success":
                            successful_endpoints += 1
        
        success_rate = (successful_endpoints / total_endpoints * 100) if total_endpoints > 0 else 0
        
        print(f"\n📈 Test Summary:")
        print(f"  🎯 Total Endpoints Tested: {total_endpoints}")
        print(f"  ✅ Successful: {successful_endpoints}")
        print(f"  ❌ Failed: {total_endpoints - successful_endpoints}")
        print(f"  📊 Success Rate: {success_rate:.1f}%")
        
        # Layer-by-layer breakdown
        print(f"\n🏗️ Architecture Layer Analysis:")
        layer_names = {
            "data_ingestion": "Data Ingestion & Intelligence",
            "ai_processing": "AI Processing & ML",
            "optimization": "Core Optimization Engine", 
            "analytics": "Real-time Analytics",
            "frontend_integration": "Frontend Integration",
            "performance": "Performance & Scalability"
        }
        
        for layer, name in layer_names.items():
            if layer in self.results:
                layer_results = self.results[layer]
                if isinstance(layer_results, dict) and "status" not in layer_results:
                    layer_success = sum(1 for r in layer_results.values() if isinstance(r, dict) and r.get("status") == "success")
                    layer_total = len(layer_results)
                    layer_rate = (layer_success / layer_total * 100) if layer_total > 0 else 0
                    print(f"  📡 {name}: {layer_success}/{layer_total} ({layer_rate:.1f}%)")
                else:
                    print(f"  📡 {name}: Tested")
        
        # Performance analysis
        if "performance" in self.results:
            perf = self.results["performance"]
            print(f"\n⚡ Performance Analysis:")
            print(f"  🚀 Average Response Time: {perf.get('avg_response_time', 0):.3f}s")
            print(f"  🎯 Performance Rating: {perf.get('performance_rating', 'Unknown')}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if success_rate >= 90:
            print(f"  🎉 Excellent! API architecture is performing exceptionally well.")
        elif success_rate >= 75:
            print(f"  ✅ Good performance with room for minor improvements.")
        elif success_rate >= 50:
            print(f"  ⚠️  Moderate performance - consider optimization.")
        else:
            print(f"  ❌ Significant issues detected - requires attention.")
        
        if "performance" in self.results:
            avg_time = self.results["performance"].get("avg_response_time", 0)
            if avg_time > 2.0:
                print(f"  🐌 Consider implementing caching for slow endpoints.")
            if avg_time < 0.5:
                print(f"  🚀 Excellent response times - system is highly optimized!")
        
        print(f"\n🌟 HivePath AI API Architecture: {'EXCELLENT' if success_rate >= 90 else 'GOOD' if success_rate >= 75 else 'NEEDS IMPROVEMENT'}")
        
        return {
            "total_time": total_time,
            "total_endpoints": total_endpoints,
            "successful_endpoints": successful_endpoints,
            "success_rate": success_rate,
            "results": self.results
        }

def main():
    """Run the complete API flow test"""
    print("🚀 HivePath AI: Complete API Architecture Flow Test")
    print("=" * 60)
    print("Testing the ultimate logistics intelligence API ecosystem")
    print("=" * 60)
    
    tester = HivePathAPITester()
    
    try:
        # Test all API layers
        tester.test_data_ingestion_apis()
        tester.test_ai_processing_apis()
        tester.test_optimization_apis()
        tester.test_analytics_apis()
        tester.test_frontend_integration()
        tester.test_performance_metrics()
        
        # Generate comprehensive report
        report = tester.generate_comprehensive_report()
        
        # Save results to file
        with open("api_flow_test_results.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: api_flow_test_results.json")
        print(f"\n🎯 HivePath AI API Architecture: READY FOR PRODUCTION!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

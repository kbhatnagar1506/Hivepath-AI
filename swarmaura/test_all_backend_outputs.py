#!/usr/bin/env python3
"""
Comprehensive test script to demonstrate ALL backend outputs available in the system
This shows every possible output format, endpoint, and data structure the backend can generate.
"""

import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8000"  # Adjust if your backend runs on different port
TEST_RUN_ID = f"comprehensive_test_{int(time.time())}"

def print_section(title: str, content: str = ""):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print('='*80)
    if content:
        print(content)

def print_json_output(title: str, data: Dict[Any, Any], max_lines: int = 50):
    """Print JSON output with formatting"""
    print(f"\n📋 {title}:")
    print("-" * 60)
    
    json_str = json.dumps(data, indent=2, default=str)
    lines = json_str.split('\n')
    
    if len(lines) <= max_lines:
        print(json_str)
    else:
        print('\n'.join(lines[:max_lines]))
        print(f"... ({len(lines) - max_lines} more lines)")

def test_basic_optimization_outputs():
    """Test 1: Basic VRP Optimization - All Output Formats"""
    print_section("TEST 1: Basic VRP Optimization Outputs")
    
    # Test data
    depot = {
        "id": "main_depot",
        "lat": 42.3601,
        "lng": -71.0589
    }
    
    locations = [
        {"id": "location_1", "lat": 42.3736, "lng": -71.1097, "demand": 25, "priority": 1, "time_window": {"start": "09:00", "end": "17:00"}},
        {"id": "location_2", "lat": 42.3755, "lng": -71.0392, "demand": 30, "priority": 2, "time_window": {"start": "10:00", "end": "18:00"}},
        {"id": "location_3", "lat": 42.3334, "lng": -71.0495, "demand": 20, "priority": 1, "time_window": {"start": "11:00", "end": "19:00"}},
        {"id": "location_4", "lat": 42.3200, "lng": -70.9500, "demand": 35, "priority": 3, "time_window": {"start": "12:00", "end": "20:00"}},
        {"id": "location_5", "lat": 42.2529, "lng": -71.0023, "demand": 40, "priority": 2, "time_window": {"start": "13:00", "end": "21:00"}}
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 100, "fuel_type": "diesel"},
        {"id": "truck_2", "capacity": 100, "fuel_type": "diesel"},
        {"id": "truck_3", "capacity": 100, "fuel_type": "ev"}
    ]
    
    # Test different presets and configurations
    test_configs = [
        {
            "name": "Ultra Fast (Haversine)",
            "config": {
                "run_id": f"{TEST_RUN_ID}_ultra_fast",
                "depot": depot,
                "vehicles": vehicles,
                "stops": locations,
                "preset": "ultra_fast",
                "use_google_maps": False
            }
        },
        {
            "name": "High Quality (Google Maps)",
            "config": {
                "run_id": f"{TEST_RUN_ID}_high_quality",
                "depot": depot,
                "vehicles": vehicles,
                "stops": locations,
                "preset": "high_quality",
                "use_google_maps": True
            }
        },
        {
            "name": "Custom Configuration",
            "config": {
                "run_id": f"{TEST_RUN_ID}_custom",
                "depot": depot,
                "vehicles": vehicles,
                "stops": locations,
                "time_limit_sec": 5,
                "allow_drop": True,
                "drop_penalty_per_priority": 10000,
                "use_service_time_model": True,
                "use_warmstart": True,
                "use_access_analysis": True,
                "use_google_maps": True,
                "access_penalty_weight": 0.002,
                "drop_penalty_weight": 0.02
            }
        }
    ]
    
    for test_config in test_configs:
        print(f"\n🚀 Testing: {test_config['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/optimize/routes", json=test_config['config'])
            
            if response.status_code == 200:
                result = response.json()
                print_json_output(f"✅ {test_config['name']} Result", result, max_lines=30)
                
                # Show specific output components
                if result.get("ok"):
                    print(f"\n📊 Summary Metrics:")
                    summary = result.get("summary", {})
                    print(f"   • Total Distance: {summary.get('total_distance_km', 0):.2f} km")
                    print(f"   • Total Drive Time: {summary.get('total_drive_min', 0)} minutes")
                    print(f"   • Total Served Demand: {summary.get('total_served_demand', 0)} units")
                    print(f"   • Routes Generated: {len(result.get('routes', []))}")
                    
                    # Show route details
                    routes = result.get("routes", [])
                    for i, route in enumerate(routes):
                        print(f"\n🚛 Route {i+1} ({route.get('vehicle_id', 'unknown')}):")
                        print(f"   • Distance: {route.get('distance_km', 0):.2f} km")
                        print(f"   • Drive Time: {route.get('drive_min', 0)} minutes")
                        print(f"   • Load: {route.get('load', 0)} units")
                        print(f"   • CO2: {route.get('co2_kg', 0):.2f} kg")
                        print(f"   • Stops: {len(route.get('stops', []))}")
                        
                        # Show stop details
                        stops = route.get("stops", [])
                        for j, stop in enumerate(stops):
                            if stop.get("node", 0) > 0:  # Not depot
                                print(f"     {j+1}. Stop {stop.get('node')} at {stop.get('t_min', 0)} min")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

def test_multi_location_outputs():
    """Test 2: Multi-Location Routing - Complex Scenarios"""
    print_section("TEST 2: Multi-Location Routing Outputs")
    
    # Complex multi-location scenario
    locations = [
        # Depots
        {"id": "depot_main", "lat": 42.3601, "lng": -71.0589, "location_type": "depot", "demand": 0, "priority": 1},
        {"id": "depot_secondary", "lat": 42.3736, "lng": -71.1097, "location_type": "depot", "demand": 0, "priority": 1},
        
        # Pickups
        {"id": "pickup_1", "lat": 42.3755, "lng": -71.0392, "location_type": "pickup", "demand": 25, "priority": 2, "service_min": 10},
        {"id": "pickup_2", "lat": 42.3334, "lng": -71.0495, "location_type": "pickup", "demand": 30, "priority": 2, "service_min": 15},
        
        # Deliveries
        {"id": "delivery_1", "lat": 42.3200, "lng": -70.9500, "location_type": "delivery", "demand": -25, "priority": 3, "service_min": 8},
        {"id": "delivery_2", "lat": 42.2529, "lng": -71.0023, "location_type": "delivery", "demand": -30, "priority": 3, "service_min": 12},
        
        # Service locations
        {"id": "service_1", "lat": 42.4084, "lng": -71.0119, "location_type": "service", "demand": 20, "priority": 1, "service_min": 20},
        {"id": "service_2", "lat": 42.3750, "lng": -70.9833, "location_type": "service", "demand": 15, "priority": 2, "service_min": 18},
        
        # Waypoints
        {"id": "waypoint_1", "lat": 42.3767, "lng": -71.0611, "location_type": "waypoint", "demand": 0, "priority": 1, "service_min": 5}
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 100, "fuel_type": "diesel", "allowed_location_types": ["pickup", "delivery", "service"]},
        {"id": "truck_2", "capacity": 80, "fuel_type": "ev", "allowed_location_types": ["service", "waypoint"]},
        {"id": "van_1", "capacity": 50, "fuel_type": "gas", "allowed_location_types": ["pickup", "delivery"]}
    ]
    
    # Test different multi-location scenarios
    test_scenarios = [
        {
            "name": "Pickup-Delivery Pairs",
            "config": {
                "run_id": f"{TEST_RUN_ID}_pickup_delivery",
                "locations": locations,
                "vehicles": vehicles,
                "preset": "pickup_delivery",
                "pickup_delivery_pairs": [
                    {"pickup": "pickup_1", "delivery": "delivery_1"},
                    {"pickup": "pickup_2", "delivery": "delivery_2"}
                ]
            }
        },
        {
            "name": "Multi-Depot Routing",
            "config": {
                "run_id": f"{TEST_RUN_ID}_multi_depot",
                "locations": locations,
                "vehicles": vehicles,
                "preset": "multi_depot"
            }
        },
        {
            "name": "Service Routes with Dependencies",
            "config": {
                "run_id": f"{TEST_RUN_ID}_service_routes",
                "locations": locations,
                "vehicles": vehicles,
                "preset": "service_routes",
                "sequences": [
                    {
                        "vehicle_id": "truck_1",
                        "location_sequence": ["depot_main", "pickup_1", "service_1", "delivery_1", "depot_main"],
                        "priority": 1
                    }
                ]
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n🚀 Testing: {scenario['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/multi-location/multi-location-routes", json=scenario['config'])
            
            if response.status_code == 200:
                result = response.json()
                print_json_output(f"✅ {scenario['name']} Result", result, max_lines=25)
                
                # Show multi-location specific outputs
                if result.get("ok"):
                    location_info = result.get("location_info", {})
                    print(f"\n📍 Location Information:")
                    print(f"   • Total Locations: {location_info.get('total_locations', 0)}")
                    print(f"   • Location Types: {location_info.get('location_types', {})}")
                    print(f"   • Pickup-Delivery Pairs: {len(location_info.get('pickup_delivery_pairs', []))}")
                    print(f"   • Enforced Sequences: {len(location_info.get('enforced_sequences', []))}")
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

def test_ml_service_time_outputs():
    """Test 3: ML Service Time Prediction Outputs"""
    print_section("TEST 3: ML Service Time Prediction Outputs")
    
    # Test service time prediction
    test_stops = [
        {
            "id": "stop_A",
            "demand": 150,
            "access_score": 0.72,
            "hour": 14,
            "weekday": 2,
            "node_idx": 1
        },
        {
            "id": "stop_B", 
            "demand": 140,
            "access_score": 0.61,
            "hour": 16,
            "weekday": 2,
            "node_idx": 2
        },
        {
            "id": "stop_C",
            "demand": 145,
            "access_score": 0.55,
            "hour": 10,
            "weekday": 2,
            "node_idx": 3
        }
    ]
    
    # Test different ML endpoints
    ml_tests = [
        {
            "name": "Service Time Prediction",
            "endpoint": "/api/v1/learn/service-time/predict",
            "data": {"stops": test_stops}
        },
        {
            "name": "Model Information",
            "endpoint": "/api/v1/learn/service-time/model-info",
            "data": {}
        },
        {
            "name": "Service Time Enrichment",
            "endpoint": "/api/v1/learn/service-time/enrich",
            "data": test_stops
        },
        {
            "name": "Service Time Validation",
            "endpoint": "/api/v1/learn/service-time/validate",
            "data": test_stops
        },
        {
            "name": "ML Health Check",
            "endpoint": "/api/v1/learn/service-time/health",
            "data": {}
        },
        {
            "name": "ML Test Endpoint",
            "endpoint": "/api/v1/learn/service-time/test",
            "data": {}
        }
    ]
    
    for test in ml_tests:
        print(f"\n🧠 Testing: {test['name']}")
        print("-" * 50)
        
        try:
            if test['data']:
                response = requests.post(f"{BASE_URL}{test['endpoint']}", json=test['data'])
            else:
                response = requests.get(f"{BASE_URL}{test['endpoint']}")
            
            if response.status_code == 200:
                result = response.json()
                print_json_output(f"✅ {test['name']} Result", result, max_lines=20)
                
                # Show specific ML outputs
                if "predictions" in result:
                    print(f"\n🔮 Predictions: {result['predictions']}")
                if "model_info" in result:
                    model_info = result['model_info']
                    print(f"\n🤖 Model Info:")
                    print(f"   • Mode: {model_info.get('mode', 'unknown')}")
                    print(f"   • Model Type: {model_info.get('model_type', 'unknown')}")
                    print(f"   • Accuracy: {model_info.get('accuracy', 'N/A')}")
                if "stats" in result:
                    stats = result['stats']
                    print(f"\n📊 Statistics:")
                    print(f"   • Count: {stats.get('count', 0)}")
                    print(f"   • Mean: {stats.get('mean', 0):.2f}")
                    print(f"   • Min: {stats.get('min', 0):.2f}")
                    print(f"   • Max: {stats.get('max', 0):.2f}")
                    
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

def test_metrics_and_plan_outputs():
    """Test 4: Metrics and Plan Retrieval Outputs"""
    print_section("TEST 4: Metrics and Plan Retrieval Outputs")
    
    # First create a plan to retrieve
    depot = {"id": "depot", "lat": 42.3601, "lng": -71.0589}
    locations = [
        {"id": "loc1", "lat": 42.3736, "lng": -71.1097, "demand": 25, "priority": 1},
        {"id": "loc2", "lat": 42.3755, "lng": -71.0392, "demand": 30, "priority": 2}
    ]
    vehicles = [{"id": "truck1", "capacity": 100}]
    
    # Create a plan
    plan_request = {
        "run_id": f"{TEST_RUN_ID}_metrics_test",
        "depot": depot,
        "vehicles": vehicles,
        "stops": locations,
        "preset": "fast"
    }
    
    try:
        # Create plan
        response = requests.post(f"{BASE_URL}/api/v1/optimize/routes", json=plan_request)
        if response.status_code != 200:
            print(f"❌ Failed to create plan: {response.text}")
            return
            
        # Test plan retrieval
        print(f"\n📋 Testing Plan Retrieval")
        print("-" * 30)
        
        response = requests.get(f"{BASE_URL}/api/v1/plan/{TEST_RUN_ID}_metrics_test")
        if response.status_code == 200:
            result = response.json()
            print_json_output("✅ Plan Retrieved", result, max_lines=25)
        else:
            print(f"❌ Plan retrieval failed: {response.text}")
        
        # Test metrics
        print(f"\n📊 Testing Metrics")
        print("-" * 30)
        
        response = requests.get(f"{BASE_URL}/api/v1/metrics/plan?run_id={TEST_RUN_ID}_metrics_test")
        if response.status_code == 200:
            result = response.json()
            print_json_output("✅ Metrics Retrieved", result, max_lines=15)
            
            if result.get("ok"):
                print(f"\n📈 Key Metrics:")
                print(f"   • Routes: {result.get('routes', 0)}")
                print(f"   • Total Distance: {result.get('total_distance_km', 0):.2f} km")
                print(f"   • Total Drive Time: {result.get('total_drive_min', 0)} minutes")
                print(f"   • Total CO2: {result.get('total_co2_kg', 0):.2f} kg")
        else:
            print(f"❌ Metrics retrieval failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {str(e)}")

def test_google_maps_integration():
    """Test 5: Google Maps Integration Outputs"""
    print_section("TEST 5: Google Maps Integration Outputs")
    
    # Test with Google Maps enabled
    depot = {"id": "depot", "lat": 42.3601, "lng": -71.0589}
    locations = [
        {"id": "cambridge", "lat": 42.3736, "lng": -71.1097, "demand": 25, "priority": 1},
        {"id": "east_boston", "lat": 42.3755, "lng": -71.0392, "demand": 30, "priority": 2},
        {"id": "south_boston", "lat": 42.3334, "lng": -71.0495, "demand": 20, "priority": 1}
    ]
    vehicles = [{"id": "truck1", "capacity": 100}]
    
    # Test Google Maps vs Haversine comparison
    test_configs = [
        {
            "name": "Google Maps (Real Roads)",
            "config": {
                "run_id": f"{TEST_RUN_ID}_google_maps",
                "depot": depot,
                "vehicles": vehicles,
                "stops": locations,
                "use_google_maps": True,
                "preset": "balanced"
            }
        },
        {
            "name": "Haversine (Straight Line)",
            "config": {
                "run_id": f"{TEST_RUN_ID}_haversine",
                "depot": depot,
                "vehicles": vehicles,
                "stops": locations,
                "use_google_maps": False,
                "preset": "balanced"
            }
        }
    ]
    
    results = {}
    
    for test_config in test_configs:
        print(f"\n🗺️ Testing: {test_config['name']}")
        print("-" * 50)
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/optimize/routes", json=test_config['config'])
            
            if response.status_code == 200:
                result = response.json()
                results[test_config['name']] = result
                print_json_output(f"✅ {test_config['name']} Result", result, max_lines=20)
                
                if result.get("ok"):
                    summary = result.get("summary", {})
                    print(f"\n📊 {test_config['name']} Summary:")
                    print(f"   • Distance: {summary.get('total_distance_km', 0):.2f} km")
                    print(f"   • Drive Time: {summary.get('total_drive_min', 0)} minutes")
                    print(f"   • Served Demand: {summary.get('total_served_demand', 0)} units")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")
    
    # Compare results
    if len(results) == 2:
        print(f"\n📊 Google Maps vs Haversine Comparison:")
        print("-" * 50)
        
        google_result = results.get("Google Maps (Real Roads)", {})
        haversine_result = results.get("Haversine (Straight Line)", {})
        
        if google_result.get("ok") and haversine_result.get("ok"):
            google_summary = google_result.get("summary", {})
            haversine_summary = haversine_result.get("summary", {})
            
            google_dist = google_summary.get("total_distance_km", 0)
            haversine_dist = haversine_summary.get("total_distance_km", 0)
            
            print(f"   • Google Maps Distance: {google_dist:.2f} km")
            print(f"   • Haversine Distance: {haversine_dist:.2f} km")
            print(f"   • Difference: {abs(google_dist - haversine_dist):.2f} km")
            print(f"   • Ratio: {google_dist/haversine_dist:.2f}x" if haversine_dist > 0 else "   • Ratio: N/A")

def test_health_and_status():
    """Test 6: Health and Status Endpoints"""
    print_section("TEST 6: Health and Status Endpoints")
    
    health_endpoints = [
        {"name": "Main Health Check", "endpoint": "/"},
        {"name": "Health Router", "endpoint": "/health"},
        {"name": "ML Health Check", "endpoint": "/api/v1/learn/service-time/health"}
    ]
    
    for endpoint_info in health_endpoints:
        print(f"\n🏥 Testing: {endpoint_info['name']}")
        print("-" * 40)
        
        try:
            response = requests.get(f"{BASE_URL}{endpoint_info['endpoint']}")
            
            if response.status_code == 200:
                result = response.json()
                print_json_output(f"✅ {endpoint_info['name']}", result, max_lines=10)
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {str(e)}")

def main():
    """Run all comprehensive backend output tests"""
    print_section("🚀 COMPREHENSIVE BACKEND OUTPUT TEST SUITE", 
                 "This test demonstrates ALL possible outputs from the RouteLoom Optimizer API")
    
    print(f"\n⏰ Test Run ID: {TEST_RUN_ID}")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all test suites
    test_basic_optimization_outputs()
    test_multi_location_outputs()
    test_ml_service_time_outputs()
    test_metrics_and_plan_outputs()
    test_google_maps_integration()
    test_health_and_status()
    
    print_section("✅ TEST SUITE COMPLETE", 
                 "All backend output formats have been demonstrated!")
    
    print(f"\n📋 Summary of Available Outputs:")
    print("   • Basic VRP optimization results (routes, summary, telemetry)")
    print("   • Multi-location routing (pickup/delivery, multi-depot, service routes)")
    print("   • ML service time predictions and model information")
    print("   • Plan retrieval and metrics endpoints")
    print("   • Google Maps vs Haversine distance comparisons")
    print("   • Health check and status endpoints")
    print("   • CO2 emissions calculations")
    print("   • Access score analysis")
    print("   • Time window constraints")
    print("   • Vehicle capacity and fuel type considerations")

if __name__ == "__main__":
    main()

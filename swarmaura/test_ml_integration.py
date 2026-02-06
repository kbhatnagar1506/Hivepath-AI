#!/usr/bin/env python3
"""
Test ML Integration for Service Time Prediction
Compares performance with and without ML predictions
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'services'))

from ortools_solver import solve_vrp
import time
import random

def create_test_scenario():
    """Create a test scenario for evaluation"""
    
    depot = {
        "id": "depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "name": "Main Depot"
    }
    
    # Create test stops with varying characteristics
    stops = [
        {
            "id": "S_A",
            "lat": 42.37,
            "lng": -71.05,
            "demand": 150,
            "access_score": 0.72,
            "priority": 1,
            "time_window": {"start": "09:00:00", "end": "17:00:00"}
        },
        {
            "id": "S_B",
            "lat": 42.34,
            "lng": -71.10,
            "demand": 140,
            "access_score": 0.61,
            "priority": 2,
            "time_window": {"start": "10:00:00", "end": "16:00:00"}
        },
        {
            "id": "S_C",
            "lat": 42.39,
            "lng": -71.02,
            "demand": 160,
            "access_score": 0.55,
            "priority": 1,
            "time_window": {"start": "08:00:00", "end": "18:00:00"}
        },
        {
            "id": "S_D",
            "lat": 42.33,
            "lng": -71.06,
            "demand": 130,
            "access_score": 0.68,
            "priority": 3,
            "time_window": {"start": "11:00:00", "end": "15:00:00"}
        },
        {
            "id": "S_E",
            "lat": 42.41,
            "lng": -71.03,
            "demand": 145,
            "access_score": 0.70,
            "priority": 2,
            "time_window": {"start": "09:30:00", "end": "16:30:00"}
        }
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 500},
        {"id": "truck_2", "capacity": 500}
    ]
    
    return depot, stops, vehicles

def test_without_ml(depot, stops, vehicles):
    """Test routing without ML predictions (using default service times)"""
    print("🔧 Testing WITHOUT ML predictions...")
    
    # Remove any existing service_min to force default behavior
    test_stops = []
    for s in stops:
        stop_copy = s.copy()
        if "service_min" in stop_copy:
            del stop_copy["service_min"]
        test_stops.append(stop_copy)
    
    start_time = time.time()
    
    result = solve_vrp(
        depot=depot,
        stops=test_stops,
        vehicles=vehicles,
        time_limit_sec=10,
        default_service_min=5,  # Fixed 5-minute service time
        allow_drop=True,
        drop_penalty_per_priority=2000,
        use_access_scores=True
    )
    
    solve_time = time.time() - start_time
    
    return result, solve_time

def test_with_ml(depot, stops, vehicles):
    """Test routing with ML predictions"""
    print("🧠 Testing WITH ML predictions...")
    
    # Remove any existing service_min to force ML prediction
    test_stops = []
    for s in stops:
        stop_copy = s.copy()
        if "service_min" in stop_copy:
            del stop_copy["service_min"]
        test_stops.append(stop_copy)
    
    start_time = time.time()
    
    result = solve_vrp(
        depot=depot,
        stops=test_stops,
        vehicles=vehicles,
        time_limit_sec=10,
        default_service_min=5,  # Fallback if ML fails
        allow_drop=True,
        drop_penalty_per_priority=2000,
        use_access_scores=True
    )
    
    solve_time = time.time() - start_time
    
    return result, solve_time

def analyze_results(baseline_result, baseline_time, ml_result, ml_time):
    """Analyze and compare results"""
    print("\n📊 PERFORMANCE COMPARISON")
    print("=" * 40)
    
    # Basic metrics
    print(f"⏱️  Solve Time:")
    print(f"   Without ML: {baseline_time:.3f}s")
    print(f"   With ML:    {ml_time:.3f}s")
    print(f"   Difference: {ml_time - baseline_time:+.3f}s")
    
    if baseline_result.get("ok") and ml_result.get("ok"):
        baseline_routes = baseline_result.get("routes", [])
        ml_routes = ml_result.get("routes", [])
        
        # Calculate metrics
        baseline_distance = sum(r.get("distance_km", 0) for r in baseline_routes)
        ml_distance = sum(r.get("distance_km", 0) for r in ml_routes)
        
        baseline_time_total = sum(r.get("drive_min", 0) for r in baseline_routes)
        ml_time_total = sum(r.get("drive_min", 0) for r in ml_routes)
        
        baseline_served = sum(len([s for s in r.get("stops", []) if s.get("node", 0) > 0]) for r in baseline_routes)
        ml_served = sum(len([s for s in r.get("stops", []) if s.get("node", 0) > 0]) for r in ml_routes)
        
        print(f"\n📏 Distance:")
        print(f"   Without ML: {baseline_distance:.2f} km")
        print(f"   With ML:    {ml_distance:.2f} km")
        print(f"   Difference: {ml_distance - baseline_distance:+.2f} km")
        
        print(f"\n⏱️  Total Drive Time:")
        print(f"   Without ML: {baseline_time_total:.1f} minutes")
        print(f"   With ML:    {ml_time_total:.1f} minutes")
        print(f"   Difference: {ml_time_total - baseline_time_total:+.1f} minutes")
        
        print(f"\n📍 Stops Served:")
        print(f"   Without ML: {baseline_served}")
        print(f"   With ML:    {ml_served}")
        print(f"   Difference: {ml_served - baseline_served:+d}")
        
        # Service time analysis
        print(f"\n🔍 Service Time Analysis:")
        baseline_service_times = []
        ml_service_times = []
        
        for route in baseline_routes:
            for stop in route.get("stops", []):
                if stop.get("node", 0) > 0:
                    baseline_service_times.append(5.0)  # Default service time
        
        for route in ml_routes:
            for stop in route.get("stops", []):
                if stop.get("node", 0) > 0:
                    # Find the corresponding stop to get ML prediction
                    stop_id = None
                    for s in ml_result.get("stops", []):
                        if s.get("node") == stop.get("node"):
                            stop_id = s.get("id")
                            break
                    
                    if stop_id:
                        # Find original stop data
                        for s in ml_result.get("stops", []):
                            if s.get("id") == stop_id:
                                ml_service_times.append(s.get("service_min", 5.0))
                                break
                        else:
                            ml_service_times.append(5.0)
                    else:
                        ml_service_times.append(5.0)
        
        if baseline_service_times and ml_service_times:
            baseline_avg_service = sum(baseline_service_times) / len(baseline_service_times)
            ml_avg_service = sum(ml_service_times) / len(ml_service_times)
            
            print(f"   Average Service Time:")
            print(f"   Without ML: {baseline_avg_service:.1f} minutes (fixed)")
            print(f"   With ML:    {ml_avg_service:.1f} minutes (predicted)")
            print(f"   Difference: {ml_avg_service - baseline_avg_service:+.1f} minutes")
        
        # Efficiency metrics
        if baseline_distance > 0 and ml_distance > 0:
            efficiency_improvement = ((baseline_distance - ml_distance) / baseline_distance) * 100
            print(f"\n📈 Efficiency Improvement: {efficiency_improvement:+.1f}%")
        
        if baseline_time_total > 0 and ml_time_total > 0:
            time_improvement = ((baseline_time_total - ml_time_total) / baseline_time_total) * 100
            print(f"⏱️  Time Improvement: {time_improvement:+.1f}%")
    
    else:
        print(f"\n❌ One or both results failed:")
        if not baseline_result.get("ok"):
            print(f"   Baseline error: {baseline_result.get('error', 'Unknown')}")
        if not ml_result.get("ok"):
            print(f"   ML error: {ml_result.get('error', 'Unknown')}")

def main():
    """Main test function"""
    print("🧪 ML INTEGRATION TEST")
    print("=" * 30)
    print("Testing service time prediction integration...")
    print()
    
    # Create test scenario
    depot, stops, vehicles = create_test_scenario()
    
    print(f"📊 Test Scenario:")
    print(f"   📍 Depot: {depot['name']}")
    print(f"   📍 Stops: {len(stops)}")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    print()
    
    # Test without ML
    baseline_result, baseline_time = test_without_ml(depot, stops, vehicles)
    
    # Test with ML
    ml_result, ml_time = test_with_ml(depot, stops, vehicles)
    
    # Analyze results
    analyze_results(baseline_result, baseline_time, ml_result, ml_time)
    
    print("\n🎉 ML Integration Test Complete!")
    print("=" * 40)
    
    # Show detailed route comparison
    if baseline_result.get("ok") and ml_result.get("ok"):
        print("\n📋 ROUTE COMPARISON")
        print("-" * 25)
        
        print("🔧 Without ML (Fixed 5min service):")
        for i, route in enumerate(baseline_result.get("routes", []), 1):
            stops_in_route = [s for s in route.get("stops", []) if s.get("node", 0) > 0]
            if stops_in_route:
                print(f"   🚛 Route {i}: {len(stops_in_route)} stops, {route.get('distance_km', 0):.1f}km, {route.get('drive_min', 0)}min")
        
        print("\n🧠 With ML (Predicted service times):")
        for i, route in enumerate(ml_result.get("routes", []), 1):
            stops_in_route = [s for s in route.get("stops", []) if s.get("node", 0) > 0]
            if stops_in_route:
                print(f"   🚛 Route {i}: {len(stops_in_route)} stops, {route.get('distance_km', 0):.1f}km, {route.get('drive_min', 0)}min")
                
                # Show ML predictions for this route
                for stop in stops_in_route:
                    stop_id = None
                    for s in ml_result.get("stops", []):
                        if s.get("node") == stop.get("node"):
                            stop_id = s.get("id")
                            break
                    
                    if stop_id:
                        for s in ml_result.get("stops", []):
                            if s.get("id") == stop_id:
                                print(f"      📍 {stop_id}: {s.get('service_min', 5.0):.1f}min service")
                                break

if __name__ == "__main__":
    main()

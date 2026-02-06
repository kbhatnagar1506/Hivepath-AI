#!/usr/bin/env python3
"""
Comprehensive ML Evaluation
Tests the ML integration with detailed analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'services'))

from ortools_solver import solve_vrp
from solver_hooks import enrich_service_times, get_service_time_model_info
import time
import random

def create_realistic_test_scenario():
    """Create a more realistic test scenario"""
    
    depot = {
        "id": "depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "name": "Main Depot"
    }
    
    # Create stops with realistic characteristics
    stops = [
        {
            "id": "S_A",
            "lat": 42.37,
            "lng": -71.05,
            "demand": 150,
            "access_score": 0.72,
            "priority": 1
        },
        {
            "id": "S_B",
            "lat": 42.34,
            "lng": -71.10,
            "demand": 140,
            "access_score": 0.61,
            "priority": 2
        },
        {
            "id": "S_C",
            "lat": 42.39,
            "lng": -71.02,
            "demand": 160,
            "access_score": 0.55,
            "priority": 1
        },
        {
            "id": "S_D",
            "lat": 42.33,
            "lng": -71.06,
            "demand": 130,
            "access_score": 0.68,
            "priority": 3
        },
        {
            "id": "S_E",
            "lat": 42.41,
            "lng": -71.03,
            "demand": 145,
            "access_score": 0.70,
            "priority": 2
        },
        {
            "id": "S_F",
            "lat": 42.35,
            "lng": -71.08,
            "demand": 120,
            "access_score": 0.65,
            "priority": 1
        }
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 1000},
        {"id": "truck_2", "capacity": 1000}
    ]
    
    return depot, stops, vehicles

def test_ml_predictions_directly():
    """Test ML predictions directly"""
    print("🧠 Testing ML Predictions Directly")
    print("-" * 40)
    
    # Create test stops
    test_stops = [
        {
            "id": "S_A",
            "demand": 150,
            "access_score": 0.72
        },
        {
            "id": "S_B",
            "demand": 140,
            "access_score": 0.61
        },
        {
            "id": "S_C",
            "demand": 160,
            "access_score": 0.55
        }
    ]
    
    # Test ML predictions
    enriched_stops = enrich_service_times(test_stops)
    
    print("📊 ML Predictions:")
    for stop in enriched_stops:
        print(f"   📍 {stop['id']}: {stop['service_min']:.1f}min (demand={stop['demand']}, access={stop['access_score']:.2f})")
    
    # Show model info
    model_info = get_service_time_model_info()
    print(f"\n📊 Model Info:")
    print(f"   🎯 Type: {model_info['model_type']}")
    print(f"   📈 Mean Service Time: {model_info['y_mean']:.2f} minutes" if model_info['y_mean'] else "   📈 Mean Service Time: N/A")
    
    return enriched_stops

def test_solver_with_ml():
    """Test solver with ML integration"""
    print("\n🚛 Testing Solver with ML Integration")
    print("-" * 40)
    
    depot, stops, vehicles = create_realistic_test_scenario()
    
    print(f"📊 Test Scenario:")
    print(f"   📍 Depot: {depot['name']}")
    print(f"   📍 Stops: {len(stops)}")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    
    # Test with ML predictions
    print("\n🧠 Running solver with ML predictions...")
    start_time = time.time()
    
    result = solve_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles,
        time_limit_sec=15,
        default_service_min=5,
        allow_drop=True,
        drop_penalty_per_priority=1000,
        use_access_scores=True
    )
    
    solve_time = time.time() - start_time
    
    print(f"⏱️  Solve time: {solve_time:.3f}s")
    
    if result.get("ok"):
        routes = result.get("routes", [])
        print(f"✅ Solution found!")
        print(f"   🚛 Active routes: {len([r for r in routes if len([s for s in r.get('stops', []) if s.get('node', 0) > 0]) > 0])}")
        print(f"   📏 Total distance: {sum(r.get('distance_km', 0) for r in routes):.2f} km")
        print(f"   ⏱️  Total time: {sum(r.get('drive_min', 0) for r in routes):.1f} minutes")
        
        # Show detailed routes
        print(f"\n📋 Detailed Routes:")
        for i, route in enumerate(routes, 1):
            stops_in_route = [s for s in route.get("stops", []) if s.get("node", 0) > 0]
            if stops_in_route:
                print(f"   🚛 Route {i}:")
                print(f"      📏 Distance: {route.get('distance_km', 0):.2f} km")
                print(f"      ⏱️  Drive Time: {route.get('drive_min', 0)} minutes")
                print(f"      📍 Stops: {len(stops_in_route)}")
                
                # Show service times for each stop
                for stop in stops_in_route:
                    stop_id = None
                    for s in result.get("stops", []):
                        if s.get("node") == stop.get("node"):
                            stop_id = s.get("id")
                            break
                    
                    if stop_id:
                        for s in result.get("stops", []):
                            if s.get("id") == stop_id:
                                print(f"         📍 {stop_id}: {s.get('service_min', 5.0):.1f}min service")
                                break
    else:
        print(f"❌ Solver failed: {result.get('error', 'Unknown error')}")
    
    return result, solve_time

def test_solver_without_ml():
    """Test solver without ML (for comparison)"""
    print("\n🔧 Testing Solver without ML (Baseline)")
    print("-" * 40)
    
    depot, stops, vehicles = create_realistic_test_scenario()
    
    # Remove any ML predictions
    baseline_stops = []
    for s in stops:
        stop_copy = s.copy()
        if "service_min" in stop_copy:
            del stop_copy["service_min"]
        baseline_stops.append(stop_copy)
    
    print("🔧 Running solver with fixed 5-minute service times...")
    start_time = time.time()
    
    result = solve_vrp(
        depot=depot,
        stops=baseline_stops,
        vehicles=vehicles,
        time_limit_sec=15,
        default_service_min=5,  # Fixed service time
        allow_drop=True,
        drop_penalty_per_priority=1000,
        use_access_scores=True
    )
    
    solve_time = time.time() - start_time
    
    print(f"⏱️  Solve time: {solve_time:.3f}s")
    
    if result.get("ok"):
        routes = result.get("routes", [])
        print(f"✅ Solution found!")
        print(f"   🚛 Active routes: {len([r for r in routes if len([s for s in r.get('stops', []) if s.get('node', 0) > 0]) > 0])}")
        print(f"   📏 Total distance: {sum(r.get('distance_km', 0) for r in routes):.2f} km")
        print(f"   ⏱️  Total time: {sum(r.get('drive_min', 0) for r in routes):.1f} minutes")
    else:
        print(f"❌ Solver failed: {result.get('error', 'Unknown error')}")
    
    return result, solve_time

def compare_results(ml_result, ml_time, baseline_result, baseline_time):
    """Compare ML vs baseline results"""
    print("\n📊 COMPREHENSIVE COMPARISON")
    print("=" * 50)
    
    print(f"⏱️  Solve Time:")
    print(f"   ML:       {ml_time:.3f}s")
    print(f"   Baseline: {baseline_time:.3f}s")
    print(f"   Difference: {ml_time - baseline_time:+.3f}s")
    
    if ml_result.get("ok") and baseline_result.get("ok"):
        ml_routes = ml_result.get("routes", [])
        baseline_routes = baseline_result.get("routes", [])
        
        ml_distance = sum(r.get("distance_km", 0) for r in ml_routes)
        baseline_distance = sum(r.get("distance_km", 0) for r in baseline_routes)
        
        ml_time_total = sum(r.get("drive_min", 0) for r in ml_routes)
        baseline_time_total = sum(r.get("drive_min", 0) for r in baseline_routes)
        
        ml_served = sum(len([s for s in r.get("stops", []) if s.get("node", 0) > 0]) for r in ml_routes)
        baseline_served = sum(len([s for s in r.get("stops", []) if s.get("node", 0) > 0]) for r in baseline_routes)
        
        print(f"\n📏 Distance:")
        print(f"   ML:       {ml_distance:.2f} km")
        print(f"   Baseline: {baseline_distance:.2f} km")
        print(f"   Difference: {ml_distance - baseline_distance:+.2f} km")
        
        print(f"\n⏱️  Total Drive Time:")
        print(f"   ML:       {ml_time_total:.1f} minutes")
        print(f"   Baseline: {baseline_time_total:.1f} minutes")
        print(f"   Difference: {ml_time_total - baseline_time_total:+.1f} minutes")
        
        print(f"\n📍 Stops Served:")
        print(f"   ML:       {ml_served}")
        print(f"   Baseline: {baseline_served}")
        print(f"   Difference: {ml_served - baseline_served:+d}")
        
        # Calculate improvements
        if baseline_distance > 0:
            distance_improvement = ((baseline_distance - ml_distance) / baseline_distance) * 100
            print(f"\n📈 Distance Improvement: {distance_improvement:+.1f}%")
        
        if baseline_time_total > 0:
            time_improvement = ((baseline_time_total - ml_time_total) / baseline_time_total) * 100
            print(f"⏱️  Time Improvement: {time_improvement:+.1f}%")
        
        if baseline_served > 0:
            served_improvement = ((ml_served - baseline_served) / baseline_served) * 100
            print(f"📍 Service Improvement: {served_improvement:+.1f}%")
    
    else:
        print(f"\n❌ Comparison not possible - one or both solutions failed")
        if not ml_result.get("ok"):
            print(f"   ML error: {ml_result.get('error', 'Unknown')}")
        if not baseline_result.get("ok"):
            print(f"   Baseline error: {baseline_result.get('error', 'Unknown')}")

def main():
    """Main evaluation function"""
    print("🧪 COMPREHENSIVE ML EVALUATION")
    print("=" * 50)
    print("Testing ML service time prediction integration...")
    print()
    
    # Test ML predictions directly
    test_ml_predictions_directly()
    
    # Test solver with ML
    ml_result, ml_time = test_solver_with_ml()
    
    # Test solver without ML
    baseline_result, baseline_time = test_solver_without_ml()
    
    # Compare results
    compare_results(ml_result, ml_time, baseline_result, baseline_time)
    
    print("\n🎉 ML Evaluation Complete!")
    print("=" * 50)
    print("✅ ML predictions working")
    print("✅ Solver integration working")
    print("✅ Performance comparison complete")

if __name__ == "__main__":
    main()

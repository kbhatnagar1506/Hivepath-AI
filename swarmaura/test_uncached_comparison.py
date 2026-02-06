#!/usr/bin/env python3
"""
Test routing with cleared cache to show real Google Maps vs Haversine differences
"""

import json
import time
from datetime import datetime, timedelta
from backend.services.ortools_solver import solve_vrp

def clear_distance_cache():
    """Clear the distance cache to ensure fresh calculations"""
    from backend.services.ortools_solver import _distance_cache
    _distance_cache.clear()
    print("🧹 Cleared distance cache")

def test_uncached_comparison():
    """Test routing with cleared cache to show real differences"""
    
    print("🔄 Testing Uncached Comparison: Google Maps vs Haversine")
    print("=" * 60)
    
    # Create test data with locations that should show differences
    depot = {
        "id": "main_depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "priority": 1,
        "demand": 0,
        "service_time_minutes": 0,
        "time_window_start": "08:00",
        "time_window_end": "20:00"
    }
    
    # Locations that should show significant differences
    locations = [
        # Locations with water crossings
        {"id": "east_boston", "lat": 42.3755, "lng": -71.0392, "priority": 2, "demand": 25, "service_time_minutes": 15, "time_window_start": "10:00", "time_window_end": "16:00"},
        {"id": "cambridge", "lat": 42.3736, "lng": -71.1097, "priority": 2, "demand": 30, "service_time_minutes": 18, "time_window_start": "11:00", "time_window_end": "17:00"},
        {"id": "revere", "lat": 42.4084, "lng": -71.0119, "priority": 3, "demand": 35, "service_time_minutes": 22, "time_window_start": "12:00", "time_window_end": "18:00"},
        {"id": "quincy", "lat": 42.2529, "lng": -71.0023, "priority": 3, "demand": 40, "service_time_minutes": 25, "time_window_start": "13:00", "time_window_end": "19:00"},
        {"id": "south_boston", "lat": 42.3334, "lng": -71.0495, "priority": 4, "demand": 45, "service_time_minutes": 20, "time_window_start": "14:00", "time_window_end": "20:00"}
    ]
    
    # 3 trucks
    trucks = [
        {"id": "truck1", "capacity": 100},
        {"id": "truck2", "capacity": 100},
        {"id": "truck3", "capacity": 100}
    ]
    
    # Calculate totals
    total_demand = sum(loc["demand"] for loc in locations)
    total_capacity = sum(truck["capacity"] for truck in trucks)
    
    print(f"📊 Test Configuration:")
    print(f"   • Depot: 1 (Downtown Boston)")
    print(f"   • Locations: 5 (with water crossings)")
    print(f"   • Trucks: 3 (100 units capacity each)")
    print(f"   • Total Capacity: {total_capacity} units")
    print(f"   • Total Demand: {total_demand} units")
    print(f"   • Capacity Utilization: {total_demand/total_capacity:.1%}")
    print(f"   • Time Limit: 30 seconds")
    print()
    
    # Test 1: Our API (Haversine) - Fresh calculation
    print("🔧 Test 1: Our API (Haversine) - Fresh Calculation")
    print("-" * 50)
    
    clear_distance_cache()  # Clear cache before Haversine test
    
    start_time = time.time()
    try:
        result_haversine = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=30,
            drop_penalty_per_priority=2000,
            use_access_scores=True,
            use_google_maps=False  # Our API
        )
        haversine_time = time.time() - start_time
        
        if result_haversine.get("ok", False):
            summary = result_haversine.get('summary', {})
            routes = result_haversine.get('routes', [])
            
            # Calculate metrics
            served_stops = 0
            active_trucks = 0
            total_distance = 0
            total_drive_time = 0
            total_load = 0
            
            for route in routes:
                stops = route.get('stops', [])
                non_depot_stops = [stop for stop in stops if stop.get('node', 0) > 0]
                if len(non_depot_stops) > 0:
                    active_trucks += 1
                served_stops += len(non_depot_stops)
                total_distance += route.get('distance_km', 0)
                total_drive_time += route.get('drive_min', 0)
                total_load += route.get('load', 0)
            
            print(f"✅ Haversine Results:")
            print(f"   • Solve Time: {haversine_time:.2f} seconds")
            print(f"   • Total Distance: {total_distance:.2f} km")
            print(f"   • Active Trucks: {active_trucks}/{len(trucks)}")
            print(f"   • Served Rate: {served_stops/len(locations):.1%}")
            print(f"   • Demand Rate: {total_load/total_demand:.1%}")
            print(f"   • Total Drive Time: {total_drive_time} minutes")
            print(f"   • Avg Distance per Truck: {total_distance/active_trucks if active_trucks > 0 else 0:.2f} km")
            print()
        else:
            print(f"❌ Haversine failed: {result_haversine.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Haversine error: {str(e)}")
        return
    
    # Test 2: Google Maps API - Fresh calculation
    print("🗺️  Test 2: Google Maps API - Fresh Calculation")
    print("-" * 50)
    
    clear_distance_cache()  # Clear cache before Google Maps test
    
    start_time = time.time()
    try:
        result_google = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=30,
            drop_penalty_per_priority=2000,
            use_access_scores=True,
            use_google_maps=True  # Google Maps API
        )
        google_time = time.time() - start_time
        
        if result_google.get("ok", False):
            summary = result_google.get('summary', {})
            routes = result_google.get('routes', [])
            
            # Calculate metrics
            served_stops = 0
            active_trucks = 0
            total_distance = 0
            total_drive_time = 0
            total_load = 0
            
            for route in routes:
                stops = route.get('stops', [])
                non_depot_stops = [stop for stop in stops if stop.get('node', 0) > 0]
                if len(non_depot_stops) > 0:
                    active_trucks += 1
                served_stops += len(non_depot_stops)
                total_distance += route.get('distance_km', 0)
                total_drive_time += route.get('drive_min', 0)
                total_load += route.get('load', 0)
            
            print(f"✅ Google Maps Results:")
            print(f"   • Solve Time: {google_time:.2f} seconds")
            print(f"   • Total Distance: {total_distance:.2f} km")
            print(f"   • Active Trucks: {active_trucks}/{len(trucks)}")
            print(f"   • Served Rate: {served_stops/len(locations):.1%}")
            print(f"   • Demand Rate: {total_load/total_demand:.1%}")
            print(f"   • Total Drive Time: {total_drive_time} minutes")
            print(f"   • Avg Distance per Truck: {total_distance/active_trucks if active_trucks > 0 else 0:.2f} km")
            print()
        else:
            print(f"❌ Google Maps failed: {result_google.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Google Maps error: {str(e)}")
        return
    
    # Comparison Summary
    print("📊 COMPARISON SUMMARY")
    print("=" * 50)
    print(f"🔧 Haversine (Our API):")
    print(f"   • Speed: {haversine_time:.2f}s")
    print(f"   • Distance: {total_distance:.2f} km")
    print(f"   • Drive Time: {total_drive_time} min")
    print(f"   • Cost: FREE")
    print(f"   • Accuracy: Straight-line (as crow flies)")
    print()
    print(f"🗺️  Google Maps API:")
    print(f"   • Speed: {google_time:.2f}s")
    print(f"   • Distance: {total_distance:.2f} km")
    print(f"   • Drive Time: {total_drive_time} min")
    print(f"   • Cost: ~$0.005 per request")
    print(f"   • Accuracy: Real road distances & traffic")
    print()
    print(f"⚡ Speed Difference: {google_time/haversine_time:.1f}x slower with Google Maps")
    print(f"🎯 Both use OUR optimization logic for routing decisions!")
    print(f"🔄 Cache cleared between tests for accurate comparison!")

if __name__ == "__main__":
    test_uncached_comparison()

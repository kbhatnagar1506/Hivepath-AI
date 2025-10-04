#!/usr/bin/env python3
"""
Test hybrid routing: Google Maps for data + Our optimization for routing decisions
"""

import json
import time
from datetime import datetime, timedelta
from backend.services.ortools_solver import solve_vrp

def test_hybrid_routing():
    """Test hybrid routing with Google Maps data + our optimization"""
    
    print("🚛 Testing Hybrid Routing: Google Maps Data + Our Optimization")
    print("=" * 70)
    
    # Create test data with 1 depot and 15 locations across Boston area
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
    
    # 15 locations across Greater Boston area
    locations = [
        # Downtown/Central locations
        {"id": "back_bay", "lat": 42.3503, "lng": -71.0740, "priority": 2, "demand": 20, "service_time_minutes": 12, "time_window_start": "09:00", "time_window_end": "15:00"},
        {"id": "north_end", "lat": 42.3647, "lng": -71.0542, "priority": 2, "demand": 25, "service_time_minutes": 15, "time_window_start": "10:00", "time_window_end": "16:00"},
        {"id": "south_end", "lat": 42.3431, "lng": -71.0711, "priority": 2, "demand": 18, "service_time_minutes": 10, "time_window_start": "11:00", "time_window_end": "17:00"},
        {"id": "beacon_hill", "lat": 42.3584, "lng": -71.0677, "priority": 3, "demand": 22, "service_time_minutes": 14, "time_window_start": "12:00", "time_window_end": "18:00"},
        
        # North locations
        {"id": "charlestown", "lat": 42.3767, "lng": -71.0611, "priority": 3, "demand": 30, "service_time_minutes": 18, "time_window_start": "13:00", "time_window_end": "19:00"},
        {"id": "cambridge", "lat": 42.3736, "lng": -71.1097, "priority": 3, "demand": 35, "service_time_minutes": 20, "time_window_start": "14:00", "time_window_end": "20:00"},
        {"id": "somerville", "lat": 42.3876, "lng": -71.0995, "priority": 4, "demand": 28, "service_time_minutes": 16, "time_window_start": "15:00", "time_window_end": "21:00"},
        
        # East locations
        {"id": "east_boston", "lat": 42.3755, "lng": -71.0392, "priority": 4, "demand": 32, "service_time_minutes": 22, "time_window_start": "16:00", "time_window_end": "22:00"},
        {"id": "revere", "lat": 42.4084, "lng": -71.0119, "priority": 4, "demand": 40, "service_time_minutes": 25, "time_window_start": "17:00", "time_window_end": "23:00"},
        
        # South locations
        {"id": "dorchester", "lat": 42.3158, "lng": -71.0922, "priority": 5, "demand": 45, "service_time_minutes": 28, "time_window_start": "18:00", "time_window_end": "24:00"},
        {"id": "south_boston", "lat": 42.3334, "lng": -71.0495, "priority": 5, "demand": 38, "service_time_minutes": 24, "time_window_start": "19:00", "time_window_end": "01:00"},
        {"id": "quincy", "lat": 42.2529, "lng": -71.0023, "priority": 5, "demand": 50, "service_time_minutes": 30, "time_window_start": "20:00", "time_window_end": "02:00"},
        
        # West locations
        {"id": "west_roxbury", "lat": 42.2834, "lng": -71.1614, "priority": 6, "demand": 42, "service_time_minutes": 26, "time_window_start": "21:00", "time_window_end": "03:00"},
        {"id": "brookline", "lat": 42.3317, "lng": -71.1212, "priority": 6, "demand": 36, "service_time_minutes": 20, "time_window_start": "22:00", "time_window_end": "04:00"},
        {"id": "newton", "lat": 42.3370, "lng": -71.2092, "priority": 6, "demand": 48, "service_time_minutes": 32, "time_window_start": "23:00", "time_window_end": "05:00"}
    ]
    
    # 10 trucks with varying capacities
    trucks = [
        {"id": "truck1", "capacity": 100},
        {"id": "truck2", "capacity": 100},
        {"id": "truck3", "capacity": 100},
        {"id": "truck4", "capacity": 100},
        {"id": "truck5", "capacity": 100},
        {"id": "truck6", "capacity": 100},
        {"id": "truck7", "capacity": 100},
        {"id": "truck8", "capacity": 100},
        {"id": "truck9", "capacity": 100},
        {"id": "truck10", "capacity": 100}
    ]
    
    # Calculate totals
    total_demand = sum(loc["demand"] for loc in locations)
    total_capacity = sum(truck["capacity"] for truck in trucks)
    
    print(f"📊 Test Configuration:")
    print(f"   • Depot: 1 (Downtown Boston)")
    print(f"   • Locations: 15 (across Greater Boston)")
    print(f"   • Trucks: 10 (100 units capacity each)")
    print(f"   • Total Capacity: {total_capacity} units")
    print(f"   • Total Demand: {total_demand} units")
    print(f"   • Capacity Utilization: {total_demand/total_capacity:.1%}")
    print(f"   • Time Limit: 60 seconds")
    print()
    
    # Test 1: Our API (Haversine) - Fast baseline
    print("🔧 Test 1: Our API (Haversine) - Fast Baseline")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result_haversine = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=60,
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
            print()
        else:
            print(f"❌ Haversine failed: {result_haversine.get('error', 'Unknown error')}")
            return
            
    except Exception as e:
        print(f"❌ Haversine error: {str(e)}")
        return
    
    # Test 2: Google Maps API - Real-world data
    print("🗺️  Test 2: Google Maps API - Real-World Data")
    print("-" * 50)
    
    start_time = time.time()
    try:
        result_google = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=60,
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
    print()
    print(f"🗺️  Google Maps API:")
    print(f"   • Speed: {google_time:.2f}s")
    print(f"   • Distance: {total_distance:.2f} km")
    print(f"   • Drive Time: {total_drive_time} min")
    print(f"   • Cost: ~$0.005 per request")
    print()
    print(f"⚡ Speed Difference: {google_time/haversine_time:.1f}x slower with Google Maps")
    print(f"🎯 Both use OUR optimization logic for routing decisions!")

if __name__ == "__main__":
    test_hybrid_routing()

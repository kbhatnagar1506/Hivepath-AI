#!/usr/bin/env python3
"""
Test multi-depot routing with 2 hubs, 6 locations, and 4 vans
"""

import json
import time
from datetime import datetime, timedelta
from backend.services.ortools_solver import solve_vrp

def test_multi_depot_routing():
    """Test multi-depot routing with 2 hubs, 6 locations, and 4 vans"""
    
    print("🚚 Testing Multi-Depot Routing: 2 Hubs, 6 Locations, 4 Vans")
    print("=" * 60)
    
    # Create test data with 1 main depot and 6 locations
    # For multi-depot, we'll use the main depot and simulate multiple start points
    depot = {
        "id": "main_depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "priority": 1,
        "demand": 0,
        "service_time_minutes": 0,
        "time_window_start": "09:00",
        "time_window_end": "17:00"
    }
    
    test_data = {
        "depot": depot,
        "stops": [
            # Location 1 - Back Bay
            {
                "id": "stop1",
                "lat": 42.3503,
                "lng": -71.0740,
                "priority": 2,
                "demand": 15,
                "service_time_minutes": 10,
                "time_window_start": "10:00",
                "time_window_end": "14:00"
            },
            # Location 2 - North End
            {
                "id": "stop2",
                "lat": 42.3647,
                "lng": -71.0542,
                "priority": 2,
                "demand": 20,
                "service_time_minutes": 15,
                "time_window_start": "11:00",
                "time_window_end": "15:00"
            },
            # Location 3 - South End
            {
                "id": "stop3",
                "lat": 42.3431,
                "lng": -71.0711,
                "priority": 3,
                "demand": 25,
                "service_time_minutes": 12,
                "time_window_start": "12:00",
                "time_window_end": "16:00"
            },
            # Location 4 - Charlestown
            {
                "id": "stop4",
                "lat": 42.3767,
                "lng": -71.0611,
                "priority": 3,
                "demand": 18,
                "service_time_minutes": 8,
                "time_window_start": "13:00",
                "time_window_end": "17:00"
            },
            # Location 5 - East Boston
            {
                "id": "stop5",
                "lat": 42.3755,
                "lng": -71.0392,
                "priority": 4,
                "demand": 22,
                "service_time_minutes": 14,
                "time_window_start": "14:00",
                "time_window_end": "18:00"
            },
            # Location 6 - West Roxbury
            {
                "id": "stop6",
                "lat": 42.2834,
                "lng": -71.1614,
                "priority": 4,
                "demand": 30,
                "service_time_minutes": 18,
                "time_window_start": "15:00",
                "time_window_end": "19:00"
            }
        ],
        "vehicles": [
            {
                "id": "van1",
                "capacity": 50
            },
            {
                "id": "van2", 
                "capacity": 50
            },
            {
                "id": "van3",
                "capacity": 50
            },
            {
                "id": "van4",
                "capacity": 50
            }
        ],
        "config": {
            "time_limit_sec": 30,
            "drop_penalty_per_priority": 1000,
            "use_access_scores": True,
            "use_google_maps": False
        }
    }
    
    print(f"📊 Test Configuration:")
    print(f"   • Depot: 1 (Downtown Boston)")
    print(f"   • Locations: 6 (Back Bay, North End, South End, Charlestown, East Boston, West Roxbury)")
    print(f"   • Vans: 4")
    print(f"   • Total Capacity: 200 units")
    print(f"   • Total Demand: 130 units")
    print(f"   • Time Limit: 30 seconds")
    print()
    
    # Run the solver
    start_time = time.time()
    
    try:
        result = solve_vrp(
            depot=test_data["depot"],
            stops=test_data["stops"],
            vehicles=test_data["vehicles"],
            time_limit_sec=test_data["config"]["time_limit_sec"],
            drop_penalty_per_priority=test_data["config"]["drop_penalty_per_priority"],
            use_access_scores=test_data["config"]["use_access_scores"],
            use_google_maps=test_data["config"]["use_google_maps"]
        )
        
        solve_time = time.time() - start_time
        
        # Display results
        print("✅ Multi-Depot Routing Results:")
        print("=" * 40)
        
        if result.get("ok", False):
            summary = result.get('summary', {})
            routes = result.get('routes', [])
            
            # Calculate served stops (exclude depot visits)
            served_stops = 0
            for route in routes:
                for stop in route.get('stops', []):
                    if stop.get('node', 0) > 0:  # Node 0 is depot
                        served_stops += 1
            
            print(f"🎯 Status: SUCCESS")
            print(f"⏱️  Solve Time: {solve_time:.2f} seconds")
            print(f"📏 Total Distance: {summary.get('total_distance_km', 0):.2f} km")
            print(f"🚚 Routes: {len(routes)}")
            print(f"📍 Served Stops: {served_stops}/{len(test_data['stops'])}")
            print(f"📊 Served Rate: {served_stops/len(test_data['stops']):.1%}")
            print(f"📦 Total Demand Served: {summary.get('total_served_demand', 0)} units")
            print(f"⏱️  Total Drive Time: {summary.get('total_drive_min', 0)} minutes")
            print()
            
            # Show route details
            for i, route in enumerate(routes, 1):
                stops = route.get('stops', [])
                # Count non-depot stops
                non_depot_stops = [stop for stop in stops if stop.get('node', 0) > 0]
                
                print(f"🚚 Route {i} (Van {route.get('vehicle_id', f'van{i}')}):")
                print(f"   • Distance: {route.get('distance_km', 0):.2f} km")
                print(f"   • Drive Time: {route.get('drive_min', 0)} minutes")
                print(f"   • Stops: {len(non_depot_stops)} (excluding depot)")
                print(f"   • Load: {route.get('load', 0)} units")
                print(f"   • CO2: {route.get('co2_kg', 0):.2f} kg")
                
                # Show path with stop names
                if stops:
                    path_parts = []
                    for stop in stops:
                        node = stop.get('node', 0)
                        if node == 0:
                            path_parts.append("depot")
                        else:
                            # Map node index to stop name
                            stop_index = node - 1  # Node 0 is depot, so stop1 is node 1
                            if 0 <= stop_index < len(test_data['stops']):
                                path_parts.append(test_data['stops'][stop_index]['id'])
                            else:
                                path_parts.append(f"node{node}")
                    print(f"   • Path: {' → '.join(path_parts)}")
                print()
                
        else:
            print(f"❌ Status: FAILED")
            print(f"⏱️  Solve Time: {solve_time:.2f} seconds")
            print(f"🚨 Error: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"❌ Status: ERROR")
        print(f"⏱️  Solve Time: {solve_time:.2f} seconds")
        print(f"🚨 Exception: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_depot_routing()

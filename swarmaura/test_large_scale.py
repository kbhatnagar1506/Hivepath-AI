#!/usr/bin/env python3
"""
Test large-scale routing with 10 trucks and 15 locations
"""

import json
import time
from datetime import datetime, timedelta
from backend.services.ortools_solver import solve_vrp

def test_large_scale_routing():
    """Test large-scale routing with 10 trucks and 15 locations"""
    
    print("🚛 Testing Large-Scale Routing: 10 Trucks, 15 Locations")
    print("=" * 60)
    
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
    
    test_data = {
        "depot": depot,
        "stops": locations,
        "vehicles": trucks,
        "config": {
            "time_limit_sec": 60,  # Longer time limit for complex problem
            "drop_penalty_per_priority": 2000,  # Higher penalty for larger problem
            "use_access_scores": True,
            "use_google_maps": False
        }
    }
    
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
        print("✅ Large-Scale Routing Results:")
        print("=" * 40)
        
        if result.get("ok", False):
            summary = result.get('summary', {})
            routes = result.get('routes', [])
            
            # Calculate served stops (exclude depot visits)
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
            
            print(f"🎯 Status: SUCCESS")
            print(f"⏱️  Solve Time: {solve_time:.2f} seconds")
            print(f"📏 Total Distance: {total_distance:.2f} km")
            print(f"🚛 Active Trucks: {active_trucks}/{len(trucks)}")
            print(f"📍 Served Stops: {served_stops}/{len(locations)}")
            print(f"📊 Served Rate: {served_stops/len(locations):.1%}")
            print(f"📦 Total Demand Served: {total_load} units")
            print(f"📊 Demand Rate: {total_load/total_demand:.1%}")
            print(f"⏱️  Total Drive Time: {total_drive_time} minutes")
            print(f"🚛 Avg Distance per Truck: {total_distance/active_trucks if active_trucks > 0 else 0:.2f} km")
            print()
            
            # Show route details
            for i, route in enumerate(routes, 1):
                stops = route.get('stops', [])
                non_depot_stops = [stop for stop in stops if stop.get('node', 0) > 0]
                
                if len(non_depot_stops) > 0:  # Only show active routes
                    print(f"🚛 Truck {i} ({route.get('vehicle_id', f'truck{i}')}):")
                    print(f"   • Distance: {route.get('distance_km', 0):.2f} km")
                    print(f"   • Drive Time: {route.get('drive_min', 0)} minutes")
                    print(f"   • Stops: {len(non_depot_stops)}")
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
                                if 0 <= stop_index < len(locations):
                                    path_parts.append(locations[stop_index]['id'])
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
    test_large_scale_routing()



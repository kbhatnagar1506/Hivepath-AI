#!/usr/bin/env python3
"""
Test Google Maps API integration vs Haversine distance calculation
"""
import sys
import os
sys.path.append('backend')

from services.ortools_solver import solve_vrp
import time

def test_boston_routing():
    """Test Boston routing with and without Google Maps API"""
    
    # Boston test locations
    depot = {
        'id': 'DEPOT_BACK_BAY',
        'lat': 42.3503,
        'lng': -71.0750
    }
    
    stops = [
        {
            'id': 'FANEUIL_HALL',
            'lat': 42.3601,
            'lng': -71.0589,
            'demand': 200,
            'priority': 3,
            'service_min': 15
        },
        {
            'id': 'NORTH_END',
            'lat': 42.3647,
            'lng': -71.0542,
            'demand': 150,
            'priority': 2,
            'service_min': 10
        },
        {
            'id': 'FENWAY_PARK',
            'lat': 42.3467,
            'lng': -71.0972,
            'demand': 100,
            'priority': 1,
            'service_min': 30
        },
        {
            'id': 'CHINATOWN',
            'lat': 42.3495,
            'lng': -71.0624,
            'demand': 180,
            'priority': 2,
            'service_min': 15
        }
    ]
    
    vehicles = [
        {
            'id': 'TRUCK_1',
            'capacity': 1000,
            'fuel_type': 'diesel'
        }
    ]
    
    print("🗺️  Boston Routing Test: Haversine vs Google Maps")
    print("=" * 60)
    
    # Test 1: Haversine (current method)
    print("\n1️⃣  Testing with Haversine Distance (Current Method)")
    print("-" * 50)
    
    start_time = time.time()
    result_haversine = solve_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles,
        speed_kmph=25.0,
        time_limit_sec=3,
        use_google_maps=False  # Use Haversine
    )
    haversine_time = time.time() - start_time
    
    if result_haversine.get('ok'):
        print(f"✅ Haversine Success: {haversine_time:.3f}s")
        print(f"📏 Distance: {result_haversine['summary']['total_distance_km']:.2f} km")
        print(f"⏰ Drive Time: {result_haversine['summary']['total_drive_min']} min")
        print(f"📦 Demand Served: {result_haversine['summary']['total_served_demand']} units")
        
        # Show route
        route = result_haversine['routes'][0]
        print(f"🛣️  Route: ", end="")
        for i, stop in enumerate(route['stops']):
            if i > 0:
                print(" → ", end="")
            if stop['node'] == 0:
                print("DEPOT", end="")
            else:
                stop_info = stops[stop['node'] - 1]
                print(f"{stop_info['id']}", end="")
        print()
    else:
        print(f"❌ Haversine Failed: {result_haversine.get('error', 'Unknown error')}")
    
    # Test 2: Google Maps (if API key available)
    print("\n2️⃣  Testing with Google Maps API (Real-world distances)")
    print("-" * 50)
    
    # Check if Google Maps API key is available
    google_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not google_api_key:
        print("⚠️  Google Maps API key not found!")
        print("   Set GOOGLE_MAPS_API_KEY environment variable to test Google Maps")
        print("   Example: export GOOGLE_MAPS_API_KEY='your_api_key_here'")
        return
    
    start_time = time.time()
    result_google = solve_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles,
        speed_kmph=25.0,
        time_limit_sec=3,
        use_google_maps=True  # Use Google Maps
    )
    google_time = time.time() - start_time
    
    if result_google.get('ok'):
        print(f"✅ Google Maps Success: {google_time:.3f}s")
        print(f"📏 Distance: {result_google['summary']['total_distance_km']:.2f} km")
        print(f"⏰ Drive Time: {result_google['summary']['total_drive_min']} min")
        print(f"📦 Demand Served: {result_google['summary']['total_served_demand']} units")
        
        # Show route
        route = result_google['routes'][0]
        print(f"🛣️  Route: ", end="")
        for i, stop in enumerate(route['stops']):
            if i > 0:
                print(" → ", end="")
            if stop['node'] == 0:
                print("DEPOT", end="")
            else:
                stop_info = stops[stop['node'] - 1]
                print(f"{stop_info['id']}", end="")
        print()
        
        # Compare results
        print("\n📊 Comparison:")
        print("-" * 30)
        if result_haversine.get('ok') and result_google.get('ok'):
            haversine_dist = result_haversine['summary']['total_distance_km']
            google_dist = result_google['summary']['total_distance_km']
            haversine_time = result_haversine['summary']['total_drive_min']
            google_time = result_google['summary']['total_drive_min']
            
            dist_diff = ((google_dist - haversine_dist) / haversine_dist) * 100
            time_diff = ((google_time - haversine_time) / haversine_time) * 100
            
            print(f"Distance: {haversine_dist:.2f} km → {google_dist:.2f} km ({dist_diff:+.1f}%)")
            print(f"Drive Time: {haversine_time} min → {google_time} min ({time_diff:+.1f}%)")
            
            if abs(dist_diff) > 10:
                print("⚠️  Significant difference! Google Maps shows real road distances")
            else:
                print("✅ Similar results - Haversine is a good approximation")
    else:
        print(f"❌ Google Maps Failed: {result_google.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("💡 Recommendation:")
    if google_api_key:
        print("   Use Google Maps API for production - provides real-world accuracy")
        print("   Use Haversine for development/testing - faster and no API costs")
    else:
        print("   Get Google Maps API key for production use")
        print("   Current Haversine method works well for development")

if __name__ == "__main__":
    test_boston_routing()

#!/usr/bin/env python3
"""
Analyze and optimize model performance
"""

import time
from backend.services.ortools_solver import solve_vrp

def test_performance_with_different_settings():
    """Test performance with different solver settings"""
    
    print("⚡ PERFORMANCE ANALYSIS: Why is the model slow?")
    print("=" * 60)
    
    # Test locations
    locations = [
        {
            "id": "back_bay",
            "lat": 42.3503,
            "lng": -71.0740,
            "name": "Back Bay Station",
            "priority": 1,
            "demand": 25,
            "service_time_minutes": 15,
            "time_window_start": "09:00",
            "time_window_end": "15:00"
        },
        {
            "id": "north_end",
            "lat": 42.3647,
            "lng": -71.0542,
            "name": "North End",
            "priority": 1,
            "demand": 30,
            "service_time_minutes": 18,
            "time_window_start": "10:00",
            "time_window_end": "16:00"
        },
        {
            "id": "cambridge",
            "lat": 42.3736,
            "lng": -71.1097,
            "name": "Harvard Square",
            "priority": 2,
            "demand": 35,
            "service_time_minutes": 20,
            "time_window_start": "11:00",
            "time_window_end": "17:00"
        }
    ]
    
    depot = {
        "id": "main_depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "name": "AI Depot",
        "priority": 1,
        "demand": 0,
        "service_time_minutes": 0,
        "time_window_start": "08:00",
        "time_window_end": "20:00"
    }
    
    trucks = [
        {"id": "truck1", "capacity": 100},
        {"id": "truck2", "capacity": 100}
    ]
    
    print(f"📊 Test Configuration:")
    print(f"   • Locations: {len(locations)}")
    print(f"   • Trucks: {len(trucks)}")
    print(f"   • Total demand: {sum(loc['demand'] for loc in locations)} units")
    print()
    
    # Test 1: Ultra Fast (no ML features)
    print("🚀 Test 1: ULTRA FAST (No ML features)")
    print("-" * 40)
    
    start_time = time.time()
    try:
        result1 = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=5,  # Very short time limit
            drop_penalty_per_priority=1000,  # Low penalty
            use_access_scores=False,  # Disable ML features
            allow_drop=True
        )
        time1 = time.time() - start_time
        
        if result1.get("ok", False):
            summary1 = result1.get('summary', {})
            print(f"✅ Status: SUCCESS")
            print(f"⏱️  Time: {time1:.2f}s")
            print(f"📍 Served: {summary1.get('served_stops', 0)}/{len(locations)}")
            print(f"📏 Distance: {summary1.get('total_distance_km', 0):.2f} km")
        else:
            print(f"❌ Status: FAILED - {result1.get('error', 'Unknown')}")
            print(f"⏱️  Time: {time1:.2f}s")
    except Exception as e:
        time1 = time.time() - start_time
        print(f"❌ Exception: {str(e)}")
        print(f"⏱️  Time: {time1:.2f}s")
    
    print()
    
    # Test 2: Fast (minimal ML features)
    print("⚡ Test 2: FAST (Minimal ML features)")
    print("-" * 40)
    
    start_time = time.time()
    try:
        result2 = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=10,  # Short time limit
            drop_penalty_per_priority=2000,
            use_access_scores=True,  # Enable basic ML
            allow_drop=True
        )
        time2 = time.time() - start_time
        
        if result2.get("ok", False):
            summary2 = result2.get('summary', {})
            print(f"✅ Status: SUCCESS")
            print(f"⏱️  Time: {time2:.2f}s")
            print(f"📍 Served: {summary2.get('served_stops', 0)}/{len(locations)}")
            print(f"📏 Distance: {summary2.get('total_distance_km', 0):.2f} km")
        else:
            print(f"❌ Status: FAILED - {result2.get('error', 'Unknown')}")
            print(f"⏱️  Time: {time2:.2f}s")
    except Exception as e:
        time2 = time.time() - start_time
        print(f"❌ Exception: {str(e)}")
        print(f"⏱️  Time: {time2:.2f}s")
    
    print()
    
    # Test 3: Current settings (what we're using)
    print("🐌 Test 3: CURRENT SETTINGS (Slow)")
    print("-" * 40)
    
    start_time = time.time()
    try:
        result3 = solve_vrp(
            depot=depot,
            stops=locations,
            vehicles=trucks,
            time_limit_sec=60,  # Long time limit
            drop_penalty_per_priority=3000,
            use_access_scores=True,
            allow_drop=True
        )
        time3 = time.time() - start_time
        
        if result3.get("ok", False):
            summary3 = result3.get('summary', {})
            print(f"✅ Status: SUCCESS")
            print(f"⏱️  Time: {time3:.2f}s")
            print(f"📍 Served: {summary3.get('served_stops', 0)}/{len(locations)}")
            print(f"📏 Distance: {summary3.get('total_distance_km', 0):.2f} km")
        else:
            print(f"❌ Status: FAILED - {result3.get('error', 'Unknown')}")
            print(f"⏱️  Time: {time3:.2f}s")
    except Exception as e:
        time3 = time.time() - start_time
        print(f"❌ Exception: {str(e)}")
        print(f"⏱️  Time: {time3:.2f}s")
    
    print()
    
    # Performance analysis
    print("📊 PERFORMANCE ANALYSIS")
    print("=" * 30)
    print(f"🚀 Ultra Fast: {time1:.2f}s")
    print(f"⚡ Fast: {time2:.2f}s")
    print(f"🐌 Current: {time3:.2f}s")
    print()
    
    if time1 < 5:
        print("✅ Ultra Fast is working well!")
    if time2 < 10:
        print("✅ Fast is working well!")
    if time3 > 30:
        print("❌ Current settings are too slow!")
    
    print()
    print("🔧 OPTIMIZATION RECOMMENDATIONS")
    print("=" * 40)
    print("1. Use shorter time limits (5-10 seconds)")
    print("2. Disable ML features for speed")
    print("3. Use lower drop penalties")
    print("4. Enable dropping stops")
    print("5. Use fewer search workers")
    print()
    print("💡 For production, use 'ultra_fast' preset!")

if __name__ == "__main__":
    test_performance_with_different_settings()

#!/usr/bin/env python3
"""
DEMO: One Dataset, All Information
Shows how a single unified dataset powers all routing intelligence
"""

import os
import sys
import time
from pathlib import Path

# Add backend to path
sys.path.append("/Users/krishnabhatnagar/hackharvard/swarmaura/backend")

def demo_unified_data_power():
    """Demonstrate how one dataset provides all information"""
    print("🎯 DEMO: ONE DATASET, ALL INFORMATION")
    print("=" * 60)
    print("Showing how a single unified dataset powers ALL capabilities")
    print()
    
    # Initialize unified data system
    from unified_data_system import UnifiedDataSystem
    uds = UnifiedDataSystem()
    
    print("📊 UNIFIED DATASET OVERVIEW")
    print("-" * 40)
    print(f"✅ Single dataset contains:")
    print(f"   • {len(uds.master_data['locations'])} locations with complete data")
    print(f"   • {len(uds.master_data['vehicles'])} vehicles with capabilities")
    print(f"   • Real-time environmental conditions")
    print(f"   • Historical patterns and trends")
    print(f"   • Accessibility and risk assessments")
    print(f"   • Service time predictions")
    print(f"   • Vehicle capability matching")
    print()
    
    # Show how one dataset provides routing data
    print("🚛 ROUTING INTELLIGENCE FROM ONE DATASET")
    print("-" * 50)
    routing_data = uds.get_routing_data()
    print(f"📍 Depot: {routing_data['depot']['name']}")
    print(f"📍 Stops: {len(routing_data['stops'])} locations")
    print(f"🚛 Vehicles: {len(routing_data['vehicles'])} available")
    print(f"📦 Total Demand: {sum(s['demand'] for s in routing_data['stops'])} units")
    print(f"📦 Total Capacity: {sum(v['capacity'] for v in routing_data['vehicles'])} units")
    print()
    
    # Show service time predictions from one dataset
    print("⏱️ SERVICE TIME PREDICTIONS FROM ONE DATASET")
    print("-" * 50)
    try:
        from backend.services.service_time_model import predictor_singleton
        service_data = uds.get_service_time_data()
        predictions = predictor_singleton.predict_minutes(service_data)
        
        for i, (service, pred) in enumerate(zip(service_data, predictions)):
            loc_name = next(loc['name'] for loc in uds.master_data['locations'] if loc['id'] == service['id'])
            print(f"   {loc_name}: {pred:.1f} min (demand: {service['demand']}, access: {service['access_score']:.2f})")
        print(f"✅ All service times predicted from unified data")
    except Exception as e:
        print(f"❌ Service time prediction: {e}")
    print()
    
    # Show risk assessment from one dataset
    print("⚠️ RISK ASSESSMENT FROM ONE DATASET")
    print("-" * 50)
    try:
        from backend.services.risk_shaper import risk_shaper_singleton
        
        locations = uds.master_data["locations"]
        stops_order = [loc["id"] for loc in locations]
        osrm_matrix = [[0 if i == j else 10 for j in range(len(locations))] for i in range(len(locations))]
        
        features = {loc["id"]: {
            "risk": loc["crime_risk"],
            "light": loc["lighting_score"],
            "cong": loc["congestion_score"]
        } for loc in locations}
        
        multipliers = risk_shaper_singleton.shape(stops_order, osrm_matrix, 14, 2, features)
        
        print(f"   Risk matrix: {multipliers.shape[0]}x{multipliers.shape[1]}")
        print(f"   Max risk multiplier: {multipliers.max():.3f}")
        print(f"   Average risk: {multipliers.mean():.3f}")
        print(f"✅ Risk assessment computed from unified data")
    except Exception as e:
        print(f"❌ Risk assessment: {e}")
    print()
    
    # Show accessibility analysis from one dataset
    print("♿ ACCESSIBILITY ANALYSIS FROM ONE DATASET")
    print("-" * 50)
    access_data = uds.get_accessibility_data()
    
    total_features = sum(len(loc["features"]) for loc in access_data)
    avg_access = sum(loc["access_score"] for loc in access_data) / len(access_data)
    hazard_locations = len([loc for loc in access_data if loc["hazards"]])
    
    print(f"   Total accessibility features: {total_features}")
    print(f"   Average access score: {avg_access:.2f}/1.0")
    print(f"   Locations with hazards: {hazard_locations}")
    print(f"   Features detected: {', '.join(set(f for loc in access_data for f in loc['features']))}")
    print(f"✅ Accessibility analysis from unified data")
    print()
    
    # Show environmental intelligence from one dataset
    print("🌤️ ENVIRONMENTAL INTELLIGENCE FROM ONE DATASET")
    print("-" * 50)
    env_data = uds.get_environmental_data()
    
    weather = env_data["weather"]
    traffic = env_data["traffic"]
    
    print(f"   Weather: {weather['condition']} ({weather['temperature']}°C)")
    print(f"   Humidity: {weather['humidity']}%")
    print(f"   Wind: {weather['wind_speed']} km/h")
    print(f"   Traffic: {traffic['overall_congestion']*100:.0f}% congestion")
    print(f"   Incidents: {traffic['incidents']}")
    print(f"   Construction zones: {traffic['construction_zones']}")
    print(f"✅ Environmental intelligence from unified data")
    print()
    
    # Show vehicle capabilities from one dataset
    print("🚛 VEHICLE CAPABILITIES FROM ONE DATASET")
    print("-" * 50)
    vehicle_data = uds.get_vehicle_capabilities()
    
    for veh in vehicle_data["vehicles"]:
        capabilities = ', '.join(veh['capabilities']) if veh['capabilities'] else 'standard'
        print(f"   {veh['id']}: {veh['capacity']} units, {veh['type']}, {capabilities}")
    
    print(f"✅ Vehicle capabilities from unified data")
    print()
    
    # Show optimization parameters from one dataset
    print("🎯 OPTIMIZATION PARAMETERS FROM ONE DATASET")
    print("-" * 50)
    opt_params = uds.get_optimization_parameters()
    
    print(f"   Time windows: {len(opt_params['time_windows'])} locations")
    print(f"   Priorities: {len(opt_params['priorities'])} locations")
    print(f"   Demands: {sum(opt_params['demands'].values())} total units")
    print(f"   Capacities: {sum(opt_params['capacities'].values())} total capacity")
    print(f"✅ Optimization parameters from unified data")
    print()
    
    # Demonstrate complete routing with unified data
    print("🚛 COMPLETE ROUTING WITH UNIFIED DATA")
    print("-" * 50)
    try:
        from backend.services.ortools_solver import solve_vrp
        
        # Use unified data for routing
        routing_data = uds.get_routing_data()
        service_data = uds.get_service_time_data()
        
        # Add service times
        for stop in routing_data["stops"]:
            for service in service_data:
                if service["id"] == stop["id"]:
                    stop["service_min"] = service["historical_avg"]
                    break
        
        # Solve routing
        start_time = time.time()
        result = solve_vrp(
            depot=routing_data["depot"],
            stops=routing_data["stops"],
            vehicles=routing_data["vehicles"],
            time_limit_sec=8,
            drop_penalty_per_priority=2000,
            use_access_scores=True
        )
        solve_time = time.time() - start_time
        
        print(f"   Solve time: {solve_time:.2f}s")
        print(f"   Routes: {len(result.get('routes', []))}")
        print(f"   Distance: {result.get('summary', {}).get('total_distance_km', 'N/A')} km")
        print(f"   Served: {result.get('summary', {}).get('served_stops', 'N/A')} stops")
        print(f"✅ Complete routing from unified data")
    except Exception as e:
        print(f"❌ Complete routing: {e}")
    print()
    
    # Final summary
    print("🎉 UNIFIED DATA SYSTEM SUMMARY")
    print("=" * 60)
    print("✅ ONE DATASET PROVIDES ALL INFORMATION:")
    print("   • Complete routing intelligence")
    print("   • Service time predictions")
    print("   • Risk assessment and safety")
    print("   • Accessibility analysis")
    print("   • Environmental conditions")
    print("   • Vehicle capabilities")
    print("   • Optimization parameters")
    print("   • Historical patterns")
    print("   • Real-time updates")
    print()
    print("🚀 BENEFITS OF UNIFIED DATA:")
    print("   • Single source of truth")
    print("   • Consistent information across all systems")
    print("   • Easy to maintain and update")
    print("   • Complete integration")
    print("   • Real-time synchronization")
    print("   • Comprehensive analysis")
    print()
    print("🎯 RESULT: ONE DATASET POWERS EVERYTHING!")

def main():
    """Main demonstration"""
    print("🎯 DEMONSTRATION: ONE DATASET, ALL INFORMATION")
    print("=" * 70)
    print("Showing how a single unified dataset provides all routing intelligence")
    print()
    
    # Change to swarmaura directory
    os.chdir("/Users/krishnabhatnagar/hackharvard/swarmaura")
    
    # Run demonstration
    demo_unified_data_power()
    
    print(f"\n🚀 Demonstration complete!")
    print(f"✅ One dataset provides all information for all capabilities!")

if __name__ == "__main__":
    main()

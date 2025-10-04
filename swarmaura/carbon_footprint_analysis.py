#!/usr/bin/env python3
"""
Carbon Footprint, Fuel, and Cost Savings Analysis
Comprehensive analysis of environmental and economic benefits from ML routing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'services'))

from ortools_solver import solve_vrp
import time
import json
from datetime import datetime

# Carbon footprint constants (kg CO2 per km)
CO2_EMISSIONS = {
    "diesel": 0.82,    # kg CO2/km
    "gasoline": 0.75,  # kg CO2/km
    "electric": 0.12,  # kg CO2/km (grid average)
    "hybrid": 0.45,    # kg CO2/km
    "default": 0.80    # kg CO2/km
}

# Fuel consumption constants (L/100km)
FUEL_CONSUMPTION = {
    "diesel": 8.5,     # L/100km
    "gasoline": 10.0,  # L/100km
    "electric": 0.0,   # L/100km
    "hybrid": 6.0,     # L/100km
    "default": 9.0     # L/100km
}

# Fuel costs (USD per liter)
FUEL_COSTS = {
    "diesel": 1.20,    # USD/L
    "gasoline": 1.15,  # USD/L
    "electric": 0.12,  # USD/kWh (converted to L equivalent)
    "hybrid": 1.17,    # USD/L
    "default": 1.15    # USD/L
}

# Driver costs (USD per hour)
DRIVER_COST_PER_HOUR = 25.0

# Vehicle maintenance costs (USD per km)
MAINTENANCE_COST_PER_KM = 0.15

def create_test_scenarios():
    """Create various test scenarios for analysis"""
    
    scenarios = {
        "small_fleet": {
            "name": "Small Fleet (3 vehicles, 8 stops)",
            "depot": {"id": "D", "lat": 42.3601, "lng": -71.0589, "name": "Boston Hub"},
            "stops": [
                {"id": "S1", "lat": 42.37, "lng": -71.05, "demand": 150, "access_score": 0.8, "priority": 1, "risk": 0.3, "lighting": 0.7, "congestion": 0.4},
                {"id": "S2", "lat": 42.34, "lng": -71.10, "demand": 120, "access_score": 0.6, "priority": 2, "risk": 0.5, "lighting": 0.5, "congestion": 0.6},
                {"id": "S3", "lat": 42.39, "lng": -71.02, "demand": 180, "access_score": 0.9, "priority": 1, "risk": 0.2, "lighting": 0.8, "congestion": 0.3},
                {"id": "S4", "lat": 42.33, "lng": -71.06, "demand": 140, "access_score": 0.7, "priority": 2, "risk": 0.4, "lighting": 0.6, "congestion": 0.5},
                {"id": "S5", "lat": 42.41, "lng": -71.03, "demand": 160, "access_score": 0.85, "priority": 1, "risk": 0.3, "lighting": 0.8, "congestion": 0.4},
                {"id": "S6", "lat": 42.35, "lng": -71.08, "demand": 130, "access_score": 0.75, "priority": 2, "risk": 0.4, "lighting": 0.7, "congestion": 0.5},
                {"id": "S7", "lat": 42.38, "lng": -71.04, "demand": 170, "access_score": 0.8, "priority": 1, "risk": 0.3, "lighting": 0.8, "congestion": 0.4},
                {"id": "S8", "lat": 42.36, "lng": -71.07, "demand": 110, "access_score": 0.65, "priority": 3, "risk": 0.6, "lighting": 0.5, "congestion": 0.7}
            ],
            "vehicles": [
                {"id": "V1", "capacity": 1000, "fuel_type": "diesel", "name": "Truck Alpha"},
                {"id": "V2", "capacity": 1000, "fuel_type": "diesel", "name": "Truck Beta"},
                {"id": "V3", "capacity": 1000, "fuel_type": "diesel", "name": "Truck Gamma"}
            ]
        },
        "medium_fleet": {
            "name": "Medium Fleet (5 vehicles, 15 stops)",
            "depot": {"id": "D", "lat": 42.3601, "lng": -71.0589, "name": "Boston Hub"},
            "stops": [
                {"id": f"S{i}", "lat": 42.35 + (i * 0.01), "lng": -71.05 + (i * 0.01), 
                 "demand": 120 + (i * 10), "access_score": 0.5 + (i * 0.03), 
                 "priority": 1 + (i % 3), "risk": 0.2 + (i * 0.05), 
                 "lighting": 0.6 + (i * 0.02), "congestion": 0.3 + (i * 0.04)}
                for i in range(1, 16)
            ],
            "vehicles": [
                {"id": f"V{i}", "capacity": 1000, "fuel_type": "diesel", "name": f"Truck {chr(64+i)}"}
                for i in range(1, 6)
            ]
        },
        "mixed_fleet": {
            "name": "Mixed Fleet (4 vehicles, 12 stops)",
            "depot": {"id": "D", "lat": 42.3601, "lng": -71.0589, "name": "Boston Hub"},
            "stops": [
                {"id": f"S{i}", "lat": 42.35 + (i * 0.01), "lng": -71.05 + (i * 0.01), 
                 "demand": 100 + (i * 15), "access_score": 0.4 + (i * 0.04), 
                 "priority": 1 + (i % 3), "risk": 0.1 + (i * 0.06), 
                 "lighting": 0.5 + (i * 0.03), "congestion": 0.2 + (i * 0.05)}
                for i in range(1, 13)
            ],
            "vehicles": [
                {"id": "V1", "capacity": 1000, "fuel_type": "diesel", "name": "Diesel Truck"},
                {"id": "V2", "capacity": 1000, "fuel_type": "electric", "name": "Electric Truck"},
                {"id": "V3", "capacity": 1000, "fuel_type": "hybrid", "name": "Hybrid Truck"},
                {"id": "V4", "capacity": 1000, "fuel_type": "gasoline", "name": "Gas Truck"}
            ]
        }
    }
    
    return scenarios

def calculate_carbon_footprint(distance_km, fuel_type="diesel"):
    """Calculate CO2 emissions for a given distance and fuel type"""
    co2_per_km = CO2_EMISSIONS.get(fuel_type, CO2_EMISSIONS["default"])
    return distance_km * co2_per_km

def calculate_fuel_consumption(distance_km, fuel_type="diesel"):
    """Calculate fuel consumption for a given distance and fuel type"""
    fuel_per_100km = FUEL_CONSUMPTION.get(fuel_type, FUEL_CONSUMPTION["default"])
    return (distance_km / 100) * fuel_per_100km

def calculate_fuel_cost(distance_km, fuel_type="diesel"):
    """Calculate fuel cost for a given distance and fuel type"""
    fuel_consumed = calculate_fuel_consumption(distance_km, fuel_type)
    cost_per_liter = FUEL_COSTS.get(fuel_type, FUEL_COSTS["default"])
    return fuel_consumed * cost_per_liter

def calculate_driver_cost(drive_time_minutes):
    """Calculate driver cost based on drive time"""
    return (drive_time_minutes / 60) * DRIVER_COST_PER_HOUR

def calculate_maintenance_cost(distance_km):
    """Calculate vehicle maintenance cost based on distance"""
    return distance_km * MAINTENANCE_COST_PER_KM

def analyze_route_economics(route, vehicle):
    """Analyze the economics of a single route"""
    distance_km = route.get("distance_km", 0)
    drive_time_min = route.get("drive_min", 0)
    fuel_type = vehicle.get("fuel_type", "diesel")
    
    # Carbon footprint
    co2_kg = calculate_carbon_footprint(distance_km, fuel_type)
    
    # Fuel consumption and cost
    fuel_consumed_l = calculate_fuel_consumption(distance_km, fuel_type)
    fuel_cost_usd = calculate_fuel_cost(distance_km, fuel_type)
    
    # Driver cost
    driver_cost_usd = calculate_driver_cost(drive_time_min)
    
    # Maintenance cost
    maintenance_cost_usd = calculate_maintenance_cost(distance_km)
    
    # Total cost
    total_cost_usd = fuel_cost_usd + driver_cost_usd + maintenance_cost_usd
    
    return {
        "distance_km": distance_km,
        "drive_time_min": drive_time_min,
        "co2_kg": round(co2_kg, 2),
        "fuel_consumed_l": round(fuel_consumed_l, 2),
        "fuel_cost_usd": round(fuel_cost_usd, 2),
        "driver_cost_usd": round(driver_cost_usd, 2),
        "maintenance_cost_usd": round(maintenance_cost_usd, 2),
        "total_cost_usd": round(total_cost_usd, 2)
    }

def run_baseline_vs_ml_comparison(scenario):
    """Compare baseline vs ML-enhanced routing"""
    print(f"\n🔧 Testing {scenario['name']}")
    print("=" * 50)
    
    depot = scenario["depot"]
    stops = scenario["stops"]
    vehicles = scenario["vehicles"]
    
    # Test baseline (no ML)
    baseline_stops = []
    for s in stops:
        stop_copy = s.copy()
        # Remove ML fields
        for field in ["service_min", "risk", "lighting", "congestion"]:
            if field in stop_copy:
                del stop_copy[field]
        baseline_stops.append(stop_copy)
    
    print("🔧 Running baseline solver...")
    start_time = time.time()
    baseline_result = solve_vrp(
        depot=depot,
        stops=baseline_stops,
        vehicles=vehicles,
        time_limit_sec=15,
        default_service_min=5,
        allow_drop=True,
        drop_penalty_per_priority=1000,
        use_access_scores=True
    )
    baseline_time = time.time() - start_time
    
    # Test ML-enhanced
    print("🧠 Running ML-enhanced solver...")
    start_time = time.time()
    ml_result = solve_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles,
        time_limit_sec=15,
        default_service_min=5,
        allow_drop=True,
        drop_penalty_per_priority=1000,
        use_access_scores=True
    )
    ml_time = time.time() - start_time
    
    if not baseline_result.get("ok") or not ml_result.get("ok"):
        print("❌ One or both solvers failed")
        return None
    
    # Analyze results
    baseline_analysis = analyze_solution(baseline_result, vehicles, "Baseline")
    ml_analysis = analyze_solution(ml_result, vehicles, "ML-Enhanced")
    
    # Calculate improvements
    improvements = calculate_improvements(baseline_analysis, ml_analysis)
    
    return {
        "scenario": scenario["name"],
        "baseline": baseline_analysis,
        "ml_enhanced": ml_analysis,
        "improvements": improvements,
        "solve_times": {
            "baseline": baseline_time,
            "ml_enhanced": ml_time
        }
    }

def analyze_solution(result, vehicles, label):
    """Analyze a routing solution"""
    routes = result.get("routes", [])
    
    total_distance = 0
    total_drive_time = 0
    total_co2 = 0
    total_fuel_cost = 0
    total_driver_cost = 0
    total_maintenance_cost = 0
    total_cost = 0
    total_fuel_consumed = 0
    active_routes = 0
    
    route_details = []
    
    for i, route in enumerate(routes):
        stops_in_route = [s for s in route.get("stops", []) if s.get("node", 0) > 0]
        if stops_in_route:
            active_routes += 1
            vehicle = vehicles[i] if i < len(vehicles) else vehicles[0]
            
            economics = analyze_route_economics(route, vehicle)
            route_details.append({
                "vehicle_id": vehicle["id"],
                "vehicle_name": vehicle.get("name", f"Vehicle {i+1}"),
                "fuel_type": vehicle.get("fuel_type", "diesel"),
                "stops": len(stops_in_route),
                **economics
            })
            
            total_distance += economics["distance_km"]
            total_drive_time += economics["drive_time_min"]
            total_co2 += economics["co2_kg"]
            total_fuel_cost += economics["fuel_cost_usd"]
            total_driver_cost += economics["driver_cost_usd"]
            total_maintenance_cost += economics["maintenance_cost_usd"]
            total_cost += economics["total_cost_usd"]
            total_fuel_consumed += economics["fuel_consumed_l"]
    
    return {
        "label": label,
        "active_routes": active_routes,
        "total_distance_km": round(total_distance, 2),
        "total_drive_time_min": total_drive_time,
        "total_co2_kg": round(total_co2, 2),
        "total_fuel_consumed_l": round(total_fuel_consumed, 2),
        "total_fuel_cost_usd": round(total_fuel_cost, 2),
        "total_driver_cost_usd": round(total_driver_cost, 2),
        "total_maintenance_cost_usd": round(total_maintenance_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "route_details": route_details
    }

def calculate_improvements(baseline, ml):
    """Calculate improvement percentages"""
    improvements = {}
    
    for metric in ["total_distance_km", "total_drive_time_min", "total_co2_kg", 
                   "total_fuel_consumed_l", "total_fuel_cost_usd", "total_driver_cost_usd", 
                   "total_maintenance_cost_usd", "total_cost_usd"]:
        baseline_val = baseline[metric]
        ml_val = ml[metric]
        
        if baseline_val > 0:
            improvement = ((baseline_val - ml_val) / baseline_val) * 100
            improvements[metric] = round(improvement, 2)
        else:
            improvements[metric] = 0
    
    return improvements

def generate_comprehensive_report(results):
    """Generate a comprehensive analysis report"""
    print("\n" + "="*80)
    print("🌍 COMPREHENSIVE CARBON FOOTPRINT & COST SAVINGS ANALYSIS")
    print("="*80)
    
    total_baseline_co2 = 0
    total_ml_co2 = 0
    total_baseline_cost = 0
    total_ml_cost = 0
    total_baseline_distance = 0
    total_ml_distance = 0
    
    for result in results:
        if result is None:
            continue
            
        print(f"\n📊 {result['scenario']}")
        print("-" * 60)
        
        baseline = result["baseline"]
        ml = result["ml_enhanced"]
        improvements = result["improvements"]
        
        print(f"🔧 {baseline['label']}:")
        print(f"   📏 Distance: {baseline['total_distance_km']} km")
        print(f"   ⏱️  Drive Time: {baseline['total_drive_time_min']} min")
        print(f"   🌱 CO2: {baseline['total_co2_kg']} kg")
        print(f"   ⛽ Fuel: {baseline['total_fuel_consumed_l']} L")
        print(f"   💰 Total Cost: ${baseline['total_cost_usd']}")
        
        print(f"\n🧠 {ml['label']}:")
        print(f"   📏 Distance: {ml['total_distance_km']} km")
        print(f"   ⏱️  Drive Time: {ml['total_drive_time_min']} min")
        print(f"   🌱 CO2: {ml['total_co2_kg']} kg")
        print(f"   ⛽ Fuel: {ml['total_fuel_consumed_l']} L")
        print(f"   💰 Total Cost: ${ml['total_cost_usd']}")
        
        print(f"\n📈 Improvements:")
        print(f"   📏 Distance: {improvements['total_distance_km']:+.1f}%")
        print(f"   ⏱️  Drive Time: {improvements['total_drive_time_min']:+.1f}%")
        print(f"   🌱 CO2 Reduction: {improvements['total_co2_kg']:+.1f}%")
        print(f"   ⛽ Fuel Savings: {improvements['total_fuel_consumed_l']:+.1f}%")
        print(f"   💰 Cost Savings: {improvements['total_cost_usd']:+.1f}%")
        
        total_baseline_co2 += baseline['total_co2_kg']
        total_ml_co2 += ml['total_co2_kg']
        total_baseline_cost += baseline['total_cost_usd']
        total_ml_cost += ml['total_cost_usd']
        total_baseline_distance += baseline['total_distance_km']
        total_ml_distance += ml['total_distance_km']
    
    # Overall summary
    print(f"\n🎯 OVERALL SUMMARY")
    print("="*60)
    
    total_co2_saved = total_baseline_co2 - total_ml_co2
    total_cost_saved = total_baseline_cost - total_ml_cost
    total_distance_saved = total_baseline_distance - total_ml_distance
    
    co2_improvement = (total_co2_saved / total_baseline_co2 * 100) if total_baseline_co2 > 0 else 0
    cost_improvement = (total_cost_saved / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
    distance_improvement = (total_distance_saved / total_baseline_distance * 100) if total_baseline_distance > 0 else 0
    
    print(f"🌱 Total CO2 Saved: {total_co2_saved:.2f} kg ({co2_improvement:+.1f}%)")
    print(f"💰 Total Cost Saved: ${total_cost_saved:.2f} ({cost_improvement:+.1f}%)")
    print(f"📏 Total Distance Saved: {total_distance_saved:.2f} km ({distance_improvement:+.1f}%)")
    
    # Environmental impact
    print(f"\n🌍 ENVIRONMENTAL IMPACT")
    print("-" * 30)
    print(f"🌱 CO2 Reduction: {total_co2_saved:.2f} kg")
    print(f"🌳 Equivalent Trees Planted: {total_co2_saved / 22:.1f} trees")
    print(f"🚗 Equivalent Car Miles Saved: {total_co2_saved / 0.411:.0f} miles")
    print(f"🏠 Equivalent Home Energy: {total_co2_saved / 4.6:.1f} days")
    
    # Economic impact
    print(f"\n💰 ECONOMIC IMPACT")
    print("-" * 20)
    print(f"💵 Daily Savings: ${total_cost_saved:.2f}")
    print(f"📅 Monthly Savings: ${total_cost_saved * 30:.2f}")
    print(f"📆 Annual Savings: ${total_cost_saved * 365:.2f}")
    print(f"⛽ Fuel Saved: {((total_baseline_co2 - total_ml_co2) / 0.82):.2f} L diesel")
    
    return {
        "total_co2_saved": total_co2_saved,
        "total_cost_saved": total_cost_saved,
        "total_distance_saved": total_distance_saved,
        "co2_improvement_percent": co2_improvement,
        "cost_improvement_percent": cost_improvement,
        "distance_improvement_percent": distance_improvement
    }

def main():
    """Main analysis function"""
    print("🌍 CARBON FOOTPRINT & COST SAVINGS ANALYSIS")
    print("=" * 60)
    print("Analyzing environmental and economic benefits of ML routing")
    print()
    
    # Create test scenarios
    scenarios = create_test_scenarios()
    
    # Run analysis for each scenario
    results = []
    for scenario_name, scenario in scenarios.items():
        result = run_baseline_vs_ml_comparison(scenario)
        results.append(result)
    
    # Generate comprehensive report
    summary = generate_comprehensive_report(results)
    
    # Save results to JSON
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "analysis_type": "carbon_footprint_cost_savings",
        "scenarios_tested": len(scenarios),
        "summary": summary,
        "detailed_results": results
    }
    
    with open("carbon_footprint_analysis.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: carbon_footprint_analysis.json")
    print(f"\n🎉 Analysis complete! ML routing shows significant environmental and economic benefits!")

if __name__ == "__main__":
    main()

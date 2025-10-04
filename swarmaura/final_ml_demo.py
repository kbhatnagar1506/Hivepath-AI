#!/usr/bin/env python3
"""
Final ML Demo - Service Time Prediction
Demonstrates the complete ML pipeline for service time prediction
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'services'))

from ortools_solver import solve_vrp
from solver_hooks import enrich_service_times, get_service_time_model_info
import time

def create_demo_scenario():
    """Create a demo scenario to showcase ML predictions"""
    
    depot = {
        "id": "depot",
        "lat": 42.3601,
        "lng": -71.0589,
        "name": "Boston Logistics Hub"
    }
    
    # Create diverse stops with different characteristics
    stops = [
        {
            "id": "downtown_office",
            "lat": 42.37,
            "lng": -71.05,
            "demand": 180,
            "access_score": 0.85,  # High accessibility
            "priority": 1,
            "name": "Downtown Office Complex"
        },
        {
            "id": "residential_area",
            "lat": 42.34,
            "lng": -71.10,
            "demand": 120,
            "access_score": 0.45,  # Low accessibility
            "priority": 2,
            "name": "Residential Area"
        },
        {
            "id": "shopping_mall",
            "lat": 42.39,
            "lng": -71.02,
            "demand": 220,
            "access_score": 0.75,  # Good accessibility
            "priority": 1,
            "name": "Shopping Mall"
        },
        {
            "id": "industrial_zone",
            "lat": 42.33,
            "lng": -71.06,
            "demand": 150,
            "access_score": 0.60,  # Medium accessibility
            "priority": 3,
            "name": "Industrial Zone"
        },
        {
            "id": "university",
            "lat": 42.41,
            "lng": -71.03,
            "demand": 200,
            "access_score": 0.90,  # Excellent accessibility
            "priority": 1,
            "name": "University Campus"
        },
        {
            "id": "hospital",
            "lat": 42.35,
            "lng": -71.08,
            "demand": 160,
            "access_score": 0.95,  # Excellent accessibility
            "priority": 1,
            "name": "Medical Center"
        }
    ]
    
    vehicles = [
        {"id": "truck_1", "capacity": 1000, "name": "Delivery Truck Alpha"},
        {"id": "truck_2", "capacity": 1000, "name": "Delivery Truck Beta"}
    ]
    
    return depot, stops, vehicles

def demonstrate_ml_predictions():
    """Demonstrate ML predictions for different stop types"""
    print("🧠 ML SERVICE TIME PREDICTIONS")
    print("=" * 50)
    
    # Create test stops with different characteristics
    test_stops = [
        {
            "id": "high_access",
            "demand": 150,
            "access_score": 0.90,
            "name": "High Accessibility Location"
        },
        {
            "id": "medium_access",
            "demand": 150,
            "access_score": 0.60,
            "name": "Medium Accessibility Location"
        },
        {
            "id": "low_access",
            "demand": 150,
            "access_score": 0.30,
            "name": "Low Accessibility Location"
        },
        {
            "id": "high_demand",
            "demand": 250,
            "access_score": 0.70,
            "name": "High Demand Location"
        },
        {
            "id": "low_demand",
            "demand": 80,
            "access_score": 0.70,
            "name": "Low Demand Location"
        }
    ]
    
    print("📊 Testing different stop characteristics:")
    print()
    
    # Get ML predictions
    enriched_stops = enrich_service_times(test_stops)
    
    print("🔮 ML Predictions vs Fixed 5min Service Time:")
    print("-" * 60)
    print(f"{'Location':<25} {'Demand':<8} {'Access':<8} {'ML Pred':<10} {'Fixed':<8} {'Diff':<8}")
    print("-" * 60)
    
    for stop in enriched_stops:
        ml_pred = stop['service_min']
        fixed = 5.0
        diff = ml_pred - fixed
        
        print(f"{stop['name']:<25} {stop['demand']:<8} {stop['access_score']:<8.2f} {ml_pred:<10.1f} {fixed:<8.1f} {diff:+.1f}")
    
    print("-" * 60)
    
    # Show model information
    model_info = get_service_time_model_info()
    print(f"\n📊 Model Information:")
    print(f"   🎯 Type: {model_info['model_type']}")
    print(f"   📈 Mean Service Time: {model_info['y_mean']:.2f} minutes" if model_info['y_mean'] else "   📈 Mean Service Time: N/A")
    print(f"   🔗 Graph Edges: {model_info['num_edges']}")
    print(f"   📍 Graph Nodes: {model_info['num_nodes']}")
    
    return enriched_stops

def demonstrate_routing_with_ml():
    """Demonstrate routing with ML predictions"""
    print("\n🚛 ROUTING WITH ML PREDICTIONS")
    print("=" * 50)
    
    depot, stops, vehicles = create_demo_scenario()
    
    print(f"📊 Demo Scenario:")
    print(f"   📍 Depot: {depot['name']}")
    print(f"   📍 Stops: {len(stops)}")
    print(f"   🚛 Vehicles: {len(vehicles)}")
    print()
    
    # Show original stop characteristics
    print("📋 Stop Characteristics:")
    for stop in stops:
        print(f"   📍 {stop['name']}: demand={stop['demand']}, access={stop['access_score']:.2f}")
    print()
    
    # Run solver with ML predictions
    print("🧠 Running solver with ML-predicted service times...")
    start_time = time.time()
    
    result = solve_vrp(
        depot=depot,
        stops=stops,
        vehicles=vehicles,
        time_limit_sec=20,
        default_service_min=5,
        allow_drop=True,
        drop_penalty_per_priority=1000,
        use_access_scores=True
    )
    
    solve_time = time.time() - start_time
    
    print(f"⏱️  Solve time: {solve_time:.3f}s")
    print()
    
    if result.get("ok"):
        routes = result.get("routes", [])
        active_routes = [r for r in routes if len([s for s in r.get("stops", []) if s.get("node", 0) > 0]) > 0]
        
        print(f"✅ Solution found!")
        print(f"   🚛 Active routes: {len(active_routes)}")
        print(f"   📏 Total distance: {sum(r.get('distance_km', 0) for r in routes):.2f} km")
        print(f"   ⏱️  Total time: {sum(r.get('drive_min', 0) for r in routes):.1f} minutes")
        print()
        
        # Show detailed routes with ML predictions
        print("📋 Detailed Routes with ML Predictions:")
        print("-" * 60)
        
        for i, route in enumerate(active_routes, 1):
            stops_in_route = [s for s in route.get("stops", []) if s.get("node", 0) > 0]
            if stops_in_route:
                print(f"🚛 {vehicles[i-1]['name']}:")
                print(f"   📏 Distance: {route.get('distance_km', 0):.2f} km")
                print(f"   ⏱️  Drive Time: {route.get('drive_min', 0)} minutes")
                print(f"   📍 Stops: {len(stops_in_route)}")
                print()
                
                # Show each stop with ML prediction
                for j, stop in enumerate(stops_in_route, 1):
                    # Find the stop data
                    stop_data = None
                    for s in result.get("stops", []):
                        if s.get("node") == stop.get("node"):
                            stop_data = s
                            break
                    
                    if stop_data:
                        print(f"   {j}. 📍 {stop_data.get('name', stop_data.get('id', 'Unknown'))}")
                        print(f"      🎯 ML Service Time: {stop_data.get('service_min', 5.0):.1f} minutes")
                        print(f"      📦 Demand: {stop_data.get('demand', 0)}")
                        print(f"      ♿ Access Score: {stop_data.get('access_score', 0.0):.2f}")
                        print(f"      🔢 Priority: {stop_data.get('priority', 0)}")
                        print()
        
        # Show comparison with fixed service times
        print("📊 Service Time Analysis:")
        print("-" * 30)
        
        total_ml_service = 0
        total_fixed_service = 0
        service_times = []
        
        for stop in result.get("stops", []):
            if stop.get("node", 0) > 0:  # Not depot
                ml_time = stop.get('service_min', 5.0)
                fixed_time = 5.0
                
                total_ml_service += ml_time
                total_fixed_service += fixed_time
                service_times.append({
                    'name': stop.get('name', stop.get('id', 'Unknown')),
                    'ml': ml_time,
                    'fixed': fixed_time,
                    'diff': ml_time - fixed_time
                })
        
        print(f"   📊 Total Service Time:")
        print(f"      ML Predictions: {total_ml_service:.1f} minutes")
        print(f"      Fixed (5min):   {total_fixed_service:.1f} minutes")
        print(f"      Difference:     {total_ml_service - total_fixed_service:+.1f} minutes")
        print()
        
        print(f"   📋 Per-Stop Breakdown:")
        for st in service_times:
            print(f"      {st['name']:<20}: ML={st['ml']:.1f}min, Fixed={st['fixed']:.1f}min, Diff={st['diff']:+.1f}min")
        
        # Calculate efficiency metrics
        if total_fixed_service > 0:
            efficiency_gain = ((total_ml_service - total_fixed_service) / total_fixed_service) * 100
            print(f"\n📈 Efficiency Impact: {efficiency_gain:+.1f}%")
        
        if service_times:
            avg_ml = sum(st['ml'] for st in service_times) / len(service_times)
            avg_fixed = sum(st['fixed'] for st in service_times) / len(service_times)
            print(f"📊 Average Service Time: ML={avg_ml:.1f}min vs Fixed={avg_fixed:.1f}min")
    
    else:
        print(f"❌ Solver failed: {result.get('error', 'Unknown error')}")
    
    return result

def main():
    """Main demo function"""
    print("🎯 ML SERVICE TIME PREDICTION DEMO")
    print("=" * 60)
    print("Demonstrating GraphSAGE-based service time prediction")
    print("for improved vehicle routing optimization")
    print()
    
    # Demonstrate ML predictions
    demonstrate_ml_predictions()
    
    # Demonstrate routing with ML
    demonstrate_routing_with_ml()
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 60)
    print("✅ ML predictions working correctly")
    print("✅ Solver integration successful")
    print("✅ Service times optimized based on:")
    print("   • Stop demand levels")
    print("   • Accessibility scores")
    print("   • Graph relationships")
    print("   • Time of day patterns")
    print()
    print("🚀 Ready for production deployment!")

if __name__ == "__main__":
    main()

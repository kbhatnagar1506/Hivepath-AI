#!/usr/bin/env python3
"""
Synthetic History Generator for Service Time Training
Generates realistic training data for GNN service time prediction
"""

import pandas as pd
import numpy as np
import json
import os

def generate_synthetic_data():
    """Generate synthetic training data for service time prediction"""
    
    print("🎯 Generating synthetic training data...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)
    
    # Define a realistic stop set (based on Boston area)
    stops = [
        {"id": "S_A", "lat": 42.37, "lng": -71.05, "demand": 150, "access_score": 0.72},
        {"id": "S_B", "lat": 42.34, "lng": -71.10, "demand": 140, "access_score": 0.61},
        {"id": "S_C", "lat": 42.39, "lng": -71.02, "demand": 145, "access_score": 0.55},
        {"id": "S_D", "lat": 42.33, "lng": -71.06, "demand": 150, "access_score": 0.65},
        {"id": "S_E", "lat": 42.41, "lng": -71.03, "demand": 140, "access_score": 0.70},
        {"id": "S_F", "lat": 42.35, "lng": -71.08, "demand": 160, "access_score": 0.68},
        {"id": "S_G", "lat": 42.38, "lng": -71.04, "demand": 135, "access_score": 0.58},
        {"id": "S_H", "lat": 42.36, "lng": -71.07, "demand": 155, "access_score": 0.73},
        {"id": "S_I", "lat": 42.40, "lng": -71.01, "demand": 145, "access_score": 0.62},
        {"id": "S_J", "lat": 42.32, "lng": -71.09, "demand": 130, "access_score": 0.59},
    ]
    
    print(f"   📍 Created {len(stops)} stops")
    
    # Create Knowledge Graph nodes
    nodes = [
        {
            "id": "D",
            "type": "Depot", 
            "lat": 42.3601,
            "lng": -71.0589,
            "features_json": json.dumps({"city": "Boston", "capacity": 1000})
        }
    ]
    
    for s in stops:
        nodes.append({
            "id": s["id"],
            "type": "Stop",
            "lat": s["lat"],
            "lng": s["lng"],
            "features_json": json.dumps({
                "demand": int(s["demand"]), 
                "access_score": float(s["access_score"]),
                "priority": int(np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2]))
            })
        })
    
    # Create Knowledge Graph edges
    edges = []
    
    # Depot to all stops
    for s in stops:
        edges.append({
            "src": "D", 
            "dst": s["id"], 
            "rel": "ROUTES_NEAR", 
            "weight": 1.0
        })
    
    # Co-visit edges (stops that are often visited together)
    for i in range(len(stops)):
        for j in range(i + 1, len(stops)):
            # Higher probability for nearby stops
            dist = np.sqrt((stops[i]["lat"] - stops[j]["lat"])**2 + 
                          (stops[i]["lng"] - stops[j]["lng"])**2)
            if dist < 0.05:  # Close stops
                weight = np.random.uniform(0.4, 0.8)
            elif dist < 0.1:  # Medium distance
                weight = np.random.uniform(0.2, 0.5)
            else:  # Far stops
                weight = np.random.uniform(0.1, 0.3)
            
            if weight > 0.2:  # Only add significant co-visit relationships
                edges.append({
                    "src": stops[i]["id"],
                    "dst": stops[j]["id"],
                    "rel": "CO_VISITED",
                    "weight": weight
                })
    
    # Add some spatial neighborhood edges
    for i, s in enumerate(stops):
        for j, other in enumerate(stops):
            if i != j:
                dist = np.sqrt((s["lat"] - other["lat"])**2 + 
                              (s["lng"] - other["lng"])**2)
                if dist < 0.03:  # Very close stops
                    edges.append({
                        "src": s["id"],
                        "dst": other["id"],
                        "rel": "NEAR",
                        "weight": 0.8
                    })
    
    print(f"   🔗 Created {len(edges)} edges")
    
    # Generate synthetic visit history
    print("   📊 Generating visit history...")
    rows = []
    
    for day in range(1, 31):  # 30 historical days
        for s in stops:
            # Generate 1-3 visits per stop per day
            num_visits = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            
            for visit in range(num_visits):
                # Time of day distribution (more visits during business hours)
                hour = np.random.choice(
                    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], 
                    p=[0.05, 0.1, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0]
                )
                
                # Add some demand variation
                demand = s["demand"] + int(np.random.normal(0, 15))
                demand = max(50, min(300, demand))  # Clamp to reasonable range
                
                # Add some access score variation
                access = s["access_score"] + np.random.normal(0, 0.05)
                access = max(0.1, min(1.0, access))  # Clamp to [0.1, 1.0]
                
                # Calculate realistic service time
                base_time = 4.0 + 0.06 * demand  # Demand-driven base time
                access_penalty = 5.0 * (1.0 - access)  # Worse access = longer service
                
                # Time of day effects
                if hour in [8, 16, 17, 18]:  # Rush hours
                    tod_penalty = 2.0
                elif hour in [12, 13]:  # Lunch time
                    tod_penalty = 1.5
                else:
                    tod_penalty = 0.0
                
                # Weather effects (simulated)
                weather_penalty = np.random.choice([0, 1, 2, 3], p=[0.6, 0.2, 0.15, 0.05])
                
                # Random noise
                noise = np.random.normal(0, 1.5)
                
                # Final service time
                service_min = max(3.0, base_time + access_penalty + tod_penalty + weather_penalty + noise)
                
                rows.append({
                    "stop_id": s["id"],
                    "weekday": day % 7,  # 0=Monday, 6=Sunday
                    "hour": hour,
                    "demand": demand,
                    "access_score": access,
                    "service_min_actual": round(service_min, 1)
                })
    
    print(f"   📈 Generated {len(rows)} visit records")
    
    # Save to CSV files
    pd.DataFrame(nodes).to_csv("data/kg_nodes.csv", index=False)
    pd.DataFrame(edges).to_csv("data/kg_edges.csv", index=False)
    pd.DataFrame(rows).to_csv("data/visits.csv", index=False)
    
    print("✅ Synthetic data generation complete!")
    print(f"   📁 Files created:")
    print(f"      • data/kg_nodes.csv ({len(nodes)} nodes)")
    print(f"      • data/kg_edges.csv ({len(edges)} edges)")
    print(f"      • data/visits.csv ({len(rows)} visits)")
    
    # Show some statistics
    visits_df = pd.DataFrame(rows)
    print(f"\n📊 Data Statistics:")
    print(f"   🕐 Service time range: {visits_df['service_min_actual'].min():.1f} - {visits_df['service_min_actual'].max():.1f} minutes")
    print(f"   📈 Average service time: {visits_df['service_min_actual'].mean():.1f} minutes")
    print(f"   📦 Demand range: {visits_df['demand'].min()} - {visits_df['demand'].max()}")
    print(f"   ♿ Access score range: {visits_df['access_score'].min():.2f} - {visits_df['access_score'].max():.2f}")
    print(f"   🕐 Hour distribution: {visits_df['hour'].value_counts().sort_index().to_dict()}")

if __name__ == "__main__":
    generate_synthetic_data()

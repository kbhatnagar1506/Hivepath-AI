#!/usr/bin/env python3
"""
Generate Edge Observation Data for Risk Shaper Training
Creates realistic historical edge data with risk factors
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def generate_edge_observations():
    """Generate realistic edge observation data"""
    
    print("🔗 Generating Edge Observation Data")
    print("=" * 40)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    os.makedirs("data", exist_ok=True)
    
    # Load existing stops from KG
    try:
        nodes_df = pd.read_csv("data/kg_nodes.csv")
        stops = [row["id"] for _, row in nodes_df.iterrows() if row["type"] == "Stop"]
        depot = [row["id"] for _, row in nodes_df.iterrows() if row["type"] == "Depot"][0]
    except FileNotFoundError:
        # Fallback stops if KG doesn't exist
        stops = ["S_A", "S_B", "S_C", "S_D", "S_E", "S_F", "S_G", "S_H", "S_I", "S_J"]
        depot = "D"
    
    print(f"   📍 Using {len(stops)} stops + depot")
    
    # Generate edge observations
    observations = []
    
    # Create all possible edge pairs
    all_locations = [depot] + stops
    edge_pairs = []
    
    for i, src in enumerate(all_locations):
        for j, dst in enumerate(all_locations):
            if i != j:  # No self-loops
                edge_pairs.append((src, dst))
    
    print(f"   🔗 Generated {len(edge_pairs)} edge pairs")
    
    # Generate observations for each edge pair over time
    for run_id in range(1, 51):  # 50 historical runs
        for src, dst in edge_pairs:
            # Random weekday and hour
            weekday = np.random.choice([0, 1, 2, 3, 4, 5, 6], p=[0.1, 0.2, 0.2, 0.2, 0.2, 0.05, 0.05])
            hour = np.random.choice(range(6, 22))
            
            # Base OSRM time (simulated)
            base_distance = np.random.uniform(1.0, 15.0)  # km
            base_speed = np.random.uniform(25, 50)  # km/h
            osrm_min = (base_distance / base_speed) * 60  # minutes
            
            # Risk factors
            risk_i = np.random.beta(2, 3)  # 0-1, skewed toward lower risk
            risk_j = np.random.beta(2, 3)
            
            # Lighting factors (worse at night)
            if 6 <= hour <= 18:
                light_i = np.random.beta(3, 2)  # Better lighting during day
                light_j = np.random.beta(3, 2)
            else:
                light_i = np.random.beta(2, 3)  # Worse lighting at night
                light_j = np.random.beta(2, 3)
            
            # Congestion factors (worse during rush hours)
            if (7 <= hour <= 9) or (17 <= hour <= 19):
                cong_i = np.random.beta(2, 2)  # Higher congestion
                cong_j = np.random.beta(2, 2)
            else:
                cong_i = np.random.beta(3, 2)  # Lower congestion
                cong_j = np.random.beta(3, 2)
            
            # Incident probability (rare but impactful)
            incident_ij = 1 if np.random.random() < 0.05 else 0
            
            # Calculate observed time with risk factors
            risk_multiplier = 1.0
            risk_multiplier += (risk_i + risk_j) * 0.3  # Risk penalty
            risk_multiplier += (1.0 - light_i + 1.0 - light_j) * 0.2  # Lighting penalty
            risk_multiplier += (cong_i + cong_j) * 0.4  # Congestion penalty
            risk_multiplier += incident_ij * 0.8  # Incident penalty
            
            # Time-of-day effects
            if weekday < 5:  # Weekday
                if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
                    risk_multiplier += 0.3
                elif 12 <= hour <= 13:  # Lunch time
                    risk_multiplier += 0.1
            else:  # Weekend
                if 10 <= hour <= 18:  # Weekend day
                    risk_multiplier += 0.1
            
            # Weather effects (simulated)
            weather_factor = np.random.choice([1.0, 1.2, 1.5, 2.0], p=[0.6, 0.2, 0.15, 0.05])
            risk_multiplier *= weather_factor
            
            # Add some noise
            noise = np.random.normal(1.0, 0.1)
            risk_multiplier *= max(0.5, noise)
            
            observed_min = osrm_min * risk_multiplier
            
            observations.append({
                "src_id": src,
                "dst_id": dst,
                "weekday": weekday,
                "hour": hour,
                "osrm_min": round(osrm_min, 2),
                "observed_min": round(observed_min, 2),
                "risk_i": round(risk_i, 3),
                "risk_j": round(risk_j, 3),
                "light_i": round(light_i, 3),
                "light_j": round(light_j, 3),
                "cong_i": round(cong_i, 3),
                "cong_j": round(cong_j, 3),
                "incident_ij": incident_ij
            })
    
    # Create DataFrame and save
    df = pd.DataFrame(observations)
    df.to_csv("data/edges_obs.csv", index=False)
    
    print(f"✅ Generated {len(observations)} edge observations")
    print(f"   📁 Saved to data/edges_obs.csv")
    
    # Show statistics
    print(f"\n📊 Data Statistics:")
    print(f"   🔗 Edge pairs: {len(edge_pairs)}")
    print(f"   📈 Observations: {len(observations)}")
    print(f"   ⏱️  OSRM time range: {df['osrm_min'].min():.1f} - {df['osrm_min'].max():.1f} minutes")
    print(f"   ⏱️  Observed time range: {df['observed_min'].min():.1f} - {df['observed_min'].max():.1f} minutes")
    print(f"   📊 Risk range: {df['risk_i'].min():.3f} - {df['risk_i'].max():.3f}")
    print(f"   💡 Lighting range: {df['light_i'].min():.3f} - {df['light_i'].max():.3f}")
    print(f"   🚦 Congestion range: {df['cong_i'].min():.3f} - {df['cong_i'].max():.3f}")
    print(f"   🚨 Incidents: {df['incident_ij'].sum()} ({df['incident_ij'].mean()*100:.1f}%)")
    
    # Calculate target variable statistics
    df['y'] = np.clip(df['observed_min'] / np.maximum(1e-6, df['osrm_min']) - 1.0, 0, 5.0)
    print(f"   🎯 Target (y) range: {df['y'].min():.3f} - {df['y'].max():.3f}")
    print(f"   🎯 Target mean: {df['y'].mean():.3f}")

def generate_assignment_history():
    """Generate assignment history data for warm-start clustering"""
    
    print("\n🚛 Generating Assignment History Data")
    print("=" * 40)
    
    # Load existing stops
    try:
        nodes_df = pd.read_csv("data/kg_nodes.csv")
        stops_data = []
        for _, row in nodes_df.iterrows():
            if row["type"] == "Stop":
                features = json.loads(row["features_json"]) if isinstance(row["features_json"], str) else {}
                stops_data.append({
                    "id": row["id"],
                    "lat": row["lat"],
                    "lng": row["lng"],
                    "demand": features.get("demand", 150),
                    "priority": features.get("priority", 1)
                })
    except FileNotFoundError:
        # Fallback stops
        stops_data = [
            {"id": "S_A", "lat": 42.37, "lng": -71.05, "demand": 150, "priority": 1},
            {"id": "S_B", "lat": 42.34, "lng": -71.10, "demand": 140, "priority": 2},
            {"id": "S_C", "lat": 42.39, "lng": -71.02, "demand": 160, "priority": 1},
            {"id": "S_D", "lat": 42.33, "lng": -71.06, "demand": 130, "priority": 3},
            {"id": "S_E", "lat": 42.41, "lng": -71.03, "demand": 145, "priority": 2},
            {"id": "S_F", "lat": 42.35, "lng": -71.08, "demand": 120, "priority": 1},
            {"id": "S_G", "lat": 42.38, "lng": -71.04, "demand": 135, "priority": 2},
            {"id": "S_H", "lat": 42.36, "lng": -71.07, "demand": 155, "priority": 1},
            {"id": "S_I", "lat": 42.40, "lng": -71.01, "demand": 125, "priority": 3},
            {"id": "S_J", "lat": 42.32, "lng": -71.09, "demand": 110, "priority": 2}
        ]
    
    print(f"   📍 Using {len(stops_data)} stops")
    
    # Generate assignment history
    assignments = []
    
    # Simulate different routing scenarios
    scenarios = [
        {"vehicles": 2, "runs": 20},
        {"vehicles": 3, "runs": 15},
        {"vehicles": 4, "runs": 10},
        {"vehicles": 5, "runs": 5}
    ]
    
    for scenario in scenarios:
        num_vehicles = scenario["vehicles"]
        num_runs = scenario["runs"]
        
        for run_id in range(1, num_runs + 1):
            run_name = f"r{run_id:03d}"
            
            # Randomly assign stops to vehicles
            vehicle_ids = [f"V{i+1}" for i in range(num_vehicles)]
            
            # Use demand and priority to influence assignment
            for stop in stops_data:
                # Higher priority stops more likely to get assigned
                priority_weight = stop["priority"]
                demand_weight = stop["demand"] / 200.0  # Normalize demand
                
                # Weighted random assignment
                weights = [1.0] * num_vehicles
                if priority_weight == 1:  # High priority
                    weights[0] = 2.0  # More likely to go to first vehicle
                elif priority_weight == 2:  # Medium priority
                    weights[1] = 1.5
                else:  # Low priority
                    weights[-1] = 1.2
                
                # Add some randomness
                weights = [w * np.random.uniform(0.8, 1.2) for w in weights]
                
                # Normalize and sample
                weights = np.array(weights)
                weights = weights / weights.sum()
                vehicle_id = np.random.choice(vehicle_ids, p=weights)
                
                # Random time within business hours
                hour = np.random.choice(range(8, 18))
                weekday = np.random.choice([0, 1, 2, 3, 4], p=[0.2, 0.2, 0.2, 0.2, 0.2])
                
                assignments.append({
                    "run_id": run_name,
                    "stop_id": stop["id"],
                    "vehicle_id": vehicle_id,
                    "lat": stop["lat"],
                    "lng": stop["lng"],
                    "demand": stop["demand"],
                    "priority": stop["priority"],
                    "hour": hour,
                    "weekday": weekday
                })
    
    # Create DataFrame and save
    df = pd.DataFrame(assignments)
    df.to_csv("data/assign_history.csv", index=False)
    
    print(f"✅ Generated {len(assignments)} assignment records")
    print(f"   📁 Saved to data/assign_history.csv")
    
    # Show statistics
    print(f"\n📊 Assignment Statistics:")
    print(f"   🚛 Total runs: {df['run_id'].nunique()}")
    print(f"   🚛 Vehicles per run: {df.groupby('run_id')['vehicle_id'].nunique().mean():.1f}")
    print(f"   📍 Stops per run: {df.groupby('run_id')['stop_id'].nunique().mean():.1f}")
    print(f"   📦 Demand range: {df['demand'].min()} - {df['demand'].max()}")
    print(f"   🔢 Priority distribution: {df['priority'].value_counts().to_dict()}")
    print(f"   🕐 Hour distribution: {df['hour'].value_counts().sort_index().to_dict()}")

def main():
    """Generate both datasets"""
    print("📊 GENERATING ML TRAINING DATA")
    print("=" * 50)
    print("Creating datasets for risk shaper and warm-start clustering...")
    print()
    
    # Generate edge observations
    generate_edge_observations()
    
    # Generate assignment history
    generate_assignment_history()
    
    print("\n🎉 Data Generation Complete!")
    print("=" * 50)
    print("✅ Edge observations ready for risk shaper training")
    print("✅ Assignment history ready for warm-start clustering")
    print("🚀 Ready to train both ML models!")

if __name__ == "__main__":
    main()

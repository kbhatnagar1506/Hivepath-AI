#!/usr/bin/env python3
"""
Synthesize assignment history for warm-start clusterer training
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# Generate synthetic assignment history
np.random.seed(42)

# Define sample stops and vehicles
stops = [
    {"id":"S_A","lat":42.37,"lng":-71.05,"demand":150,"priority":2},
    {"id":"S_B","lat":42.34,"lng":-71.10,"demand":140,"priority":1},
    {"id":"S_C","lat":42.39,"lng":-71.02,"demand":145,"priority":2},
    {"id":"S_D","lat":42.33,"lng":-71.06,"demand":150,"priority":1},
    {"id":"S_E","lat":42.41,"lng":-71.03,"demand":140,"priority":2},
    {"id":"S_F","lat":42.35,"lng":-71.08,"demand":130,"priority":1},
    {"id":"S_G","lat":42.38,"lng":-71.04,"demand":155,"priority":2},
    {"id":"S_H","lat":42.32,"lng":-71.09,"demand":125,"priority":1},
]

vehicles = ["V1", "V2", "V3", "V4"]

# Generate 50 historical runs
rows = []
for run_id in range(1, 51):
    # Randomly assign stops to vehicles using K-means-like clustering
    n_vehicles = np.random.choice([3, 4])
    selected_vehicles = np.random.choice(vehicles, n_vehicles, replace=False)
    
    # Simple clustering based on lat/lng
    coords = np.array([[s["lat"], s["lng"]] for s in stops])
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_vehicles, random_state=run_id)
    labels = kmeans.fit_predict(coords)
    
    # Assign each cluster to a vehicle
    for i, stop in enumerate(stops):
        vehicle_idx = labels[i] % len(selected_vehicles)
        vehicle_id = selected_vehicles[vehicle_idx]
        
        # Add some randomness to hour/weekday
        hour = np.random.choice([8, 10, 12, 14, 16, 18])
        weekday = np.random.choice([0, 1, 2, 3, 4, 5, 6])
        
        rows.append({
            "run_id": f"r{run_id:03d}",
            "stop_id": stop["id"],
            "vehicle_id": vehicle_id,
            "lat": stop["lat"],
            "lng": stop["lng"],
            "demand": stop["demand"],
            "priority": stop["priority"],
            "hour": hour,
            "weekday": weekday
        })

df = pd.DataFrame(rows)
df.to_csv("data/assign_history.csv", index=False)
print(f"✅ Generated data/assign_history.csv with {len(df)} assignment records")
print(f"   • Runs: {df['run_id'].nunique()}")
print(f"   • Stops: {df['stop_id'].nunique()}")
print(f"   • Vehicles: {df['vehicle_id'].nunique()}")

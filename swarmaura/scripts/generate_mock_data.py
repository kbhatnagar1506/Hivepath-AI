#!/usr/bin/env python3
"""
Generate mock training data for ML models.
"""
import pandas as pd, numpy as np, random, math
from datetime import datetime, timezone
import pathlib

def generate_service_actuals(n_rows=500):
    """Generate mock service actuals data."""
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for i in range(n_rows):
        # Basic stop info
        lat = 42.36 + np.random.normal(0, 0.05)
        lng = -71.06 + np.random.normal(0, 0.05)
        demand = np.random.randint(50, 200)
        priority = random.choice([1, 1, 2, 2, 3])  # skew lower
        
        # Time window (some stops have windows)
        has_window = random.random() < 0.6
        if has_window:
            start_hour = np.random.randint(8, 18)
            start_min = np.random.randint(0, 60)
            window_span = np.random.randint(120, 300)  # 2-5 hours
            end_hour = min(22, start_hour + window_span // 60)
            end_min = (start_min + window_span % 60) % 60
            if end_min >= 60:
                end_hour += 1
                end_min -= 60
            window_start = f"{start_hour:02d}:{start_min:02d}:00"
            window_end = f"{end_hour:02d}:{end_min:02d}:00"
        else:
            window_start = window_end = None
            
        # Arrival/departure times
        arrived_min = np.random.randint(0, 480)  # 0-8 hours from start
        service_minutes = max(1, int(np.random.normal(6, 2)))  # service time
        departed_min = arrived_min + service_minutes
        
        # Additional features
        walk_m = np.random.exponential(10)  # walking distance
        blocked_flag = random.random() < 0.1  # 10% blocked
        dow = random.randint(0, 6)  # day of week
        hod = random.randint(8, 18)  # hour of day
        
        data.append({
            "run_id": f"run_{i//10:03d}",
            "stop_id": f"S{i%100:03d}",
            "vehicle_id": f"T{(i%3)+1}",
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "demand": demand,
            "priority": priority,
            "window_start": window_start,
            "window_end": window_end,
            "arrived_min": arrived_min,
            "departed_min": departed_min,
            "walk_m": round(walk_m, 1),
            "blocked_flag": blocked_flag,
            "dow": dow,
            "hod": hod
        })
    
    return pd.DataFrame(data)

def generate_plan_edges(n_rows=1000):
    """Generate mock plan edges data."""
    np.random.seed(42)
    random.seed(42)
    
    data = []
    for i in range(n_rows):
        # Current and next stop info
        curr_lat = 42.36 + np.random.normal(0, 0.05)
        curr_lng = -71.06 + np.random.normal(0, 0.05)
        next_lat = 42.36 + np.random.normal(0, 0.05)
        next_lng = -71.06 + np.random.normal(0, 0.05)
        
        # Distance
        dist_km = math.sqrt((curr_lat - next_lat)**2 + (curr_lng - next_lng)**2) * 111  # rough km
        
        # Demands and priority
        curr_demand = np.random.randint(50, 200)
        next_demand = np.random.randint(50, 200)
        next_priority = random.choice([1, 1, 2, 2, 3])
        
        # Window slack
        window_slack_min = np.random.randint(60, 480)
        
        # Label (1 if edge appears in plan, 0 for negative)
        label = random.random() < 0.3  # 30% positive examples
        
        data.append({
            "run_id": f"run_{i//20:03d}",
            "vehicle_id": f"T{(i%3)+1}",
            "curr_stop_id": f"S{i%50:03d}",
            "next_stop_id": f"S{(i+1)%50:03d}",
            "curr_lat": round(curr_lat, 6),
            "curr_lng": round(curr_lng, 6),
            "next_lat": round(next_lat, 6),
            "next_lng": round(next_lng, 6),
            "curr_demand": curr_demand,
            "next_demand": next_demand,
            "next_priority": next_priority,
            "dist_km": round(dist_km, 2),
            "window_slack_min": window_slack_min,
            "label": int(label)
        })
    
    return pd.DataFrame(data)

def main():
    print("Generating mock training data...")
    
    # Generate service actuals
    service_df = generate_service_actuals(500)
    service_df.to_csv("data/service_actuals.csv", index=False)
    print(f"Generated {len(service_df)} service actuals rows")
    
    # Generate plan edges
    edges_df = generate_plan_edges(1000)
    edges_df.to_csv("data/plan_edges.csv", index=False)
    print(f"Generated {len(edges_df)} plan edges rows")
    
    print("Mock data saved to data/ directory")

if __name__ == "__main__":
    main()




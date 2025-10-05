#!/usr/bin/env python3
"""
Generate synthetic training data for Knowledge Graph + GNN system
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)
os.makedirs("data", exist_ok=True)

# Define a small stop set (you can pull from your real stops later)
stops = [
    {"id":"S_A","lat":42.37,"lng":-71.05,"demand":150,"access_score":0.72},
    {"id":"S_B","lat":42.34,"lng":-71.10,"demand":140,"access_score":0.61},
    {"id":"S_C","lat":42.39,"lng":-71.02,"demand":145,"access_score":0.55},
    {"id":"S_D","lat":42.33,"lng":-71.06,"demand":150,"access_score":0.65},
    {"id":"S_E","lat":42.41,"lng":-71.03,"demand":140,"access_score":0.70},
]

# Create node/edge CSVs for KG
nodes = [{"id":"D","type":"Depot","lat":42.3601,"lng":-71.0589,"features_json":json.dumps({"city":"Boston"})}]
for s in stops:
    nodes.append({"id":s["id"],"type":"Stop","lat":s["lat"],"lng":s["lng"],
                  "features_json":json.dumps({"demand":s["demand"],"access_score":s["access_score"]})})
pd.DataFrame(nodes).to_csv("data/kg_nodes.csv", index=False)

edges = [{"src":"D","dst":s["id"],"rel":"ROUTES_NEAR","weight":1.0} for s in stops]
# simple co-visit edges
for i in range(len(stops)-1):
    edges.append({"src":stops[i]["id"], "dst":stops[i+1]["id"], "rel":"CO_VISITED", "weight":0.3})
pd.DataFrame(edges).to_csv("data/kg_edges.csv", index=False)

# Synthetic visits: (weekday,hour,demand,access_score) -> service_min_actual
rows = []
for day in range(1,21):      # 20 historical days
  for s in stops:
    hour = np.random.choice([8,10,12,14,16,18])
    demand = s["demand"] + int(np.random.normal(0,15))
    access = s["access_score"] + np.random.normal(0,0.05)
    base = 4.0 + 0.06*(demand)              # demand-driven
    curb = 5.0*(1.0-access)                 # worse access -> longer
    tod  = 2.0 if hour in [8,16,18] else 0  # busy hours
    noise= np.random.normal(0,1.5)
    service = max(3.0, base + curb + tod + noise)  # minutes
    rows.append({"stop_id":s["id"], "weekday":(day%7), "hour":hour,
                 "demand":demand, "access_score":access, "service_min_actual":service})
pd.DataFrame(rows).to_csv("data/visits.csv", index=False)

print("Wrote data/kg_nodes.csv, data/kg_edges.csv, data/visits.csv")

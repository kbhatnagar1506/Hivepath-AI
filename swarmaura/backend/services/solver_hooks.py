"""
Solver hooks for Knowledge Graph + GNN integration
"""

from .service_time_model import predictor_singleton
from datetime import datetime

def enrich_service_times(stops):
    """Add learned service times to stops using GNN predictions"""
    # Add hour/weekday + access_score feature (if you don't have real one, default ~0.6)
    now = datetime.now()
    feats = []
    id2idx = {s["id"]: i+1 for i,s in enumerate(stops)}  # if you mirror training node order
    for s in stops:
        feats.append({
            "id": s["id"],
            "node_idx": id2idx.get(s["id"], 0),
            "demand": s.get("demand", 120),
            "access_score": s.get("access_score", 0.6),
            "hour": now.hour,
            "weekday": now.weekday()
        })
    pred = predictor_singleton.predict_minutes(feats)
    for s, m in zip(stops, pred):
        s["service_min"] = float(round(m, 1))
    return stops

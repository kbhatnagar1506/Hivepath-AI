import os, joblib, math
from typing import List, Dict, Any, Tuple

MODEL_PATH = os.getenv("WARMSTART_MODEL", "models/warmstart_edge_clf.joblib")

def _load():
    if not os.path.exists(MODEL_PATH): return None
    obj = joblib.load(MODEL_PATH); return obj["model"], obj["features"]

def _haversine(a:Tuple[float,float], b:Tuple[float,float]):
    R=6371.0
    import math
    la1,lo1 = map(math.radians, a); la2,lo2=map(math.radians,b)
    dlat=la2-la1; dlon=lo2-lo1
    x=(math.sin(dlat/2)**2+math.cos(la1)*math.cos(la2)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(x))

def build_initial_routes(depot: Dict[str,Any], stops: List[Dict[str,Any]], vehicles: List[Dict[str,Any]], speed_kmph: float) -> List[List[int]]:
    """Return node-index routes [[0, s_i, ..., 0], ...] as warm start."""
    loaded = _load()
    if not loaded:
        # trivial warm start: split by angle (sweep)
        return _sweep_warm_start(depot, stops, len(vehicles))
    clf, feats = loaded
    # Feasible greedy per-vehicle using classifier scores
    remaining = {s["id"]: s for s in stops}
    id_to_node = {s["id"]: i+1 for i,s in enumerate(stops)}  # node index
    routes = []
    for v in vehicles:
        cap = int(v.get("capacity", 1000))
        load = 0
        curr_xy = (depot["lat"], depot["lng"])
        path = [0]
        while remaining:
            # candidates still feasible by capacity
            cands = [s for s in remaining.values() if (load + int(s.get("demand",0)) <= cap)]
            if not cands: break
            rows, objs = [], []
            for s in cands:
                dist = _haversine(curr_xy, (s["lat"], s["lng"]))
                # naive slack (if window known)
                tw = s.get("time_window") or {}
                slack = _window_slack_minutes(tw)
                rows.append([dist, load, s.get("demand",0), s.get("priority",1), slack])
                objs.append(s)
            import numpy as np
            probs = clf.predict_proba(np.array(rows))[:,1]
            # pick best
            best = objs[int(probs.argmax())]
            path.append(id_to_node[best["id"]])
            curr_xy = (best["lat"], best["lng"])
            load += int(best.get("demand",0))
            remaining.pop(best["id"], None)
            # small cap to avoid absurdly long route; next vehicle picks up rest
            if len(path) > max(2, math.ceil(len(stops)/len(vehicles))+1): break
        path.append(0)
        routes.append(path)
    # If any remain, dump them into last vehicle path
    if remaining:
        last = routes[-1]
        for s in list(remaining.values()):
            last.insert(-1, id_to_node[s["id"]])
            remaining.pop(s["id"], None)
    return routes

def _window_slack_minutes(tw: Dict[str,str]) -> int:
    try:
        start = tw.get("start") or "00:00:00"
        end   = tw.get("end")   or "23:59:59"
        def m(s): h,m,_ = map(int, s.split(":")); return h*60+m
        return max(0, m(end) - m(start))
    except:
        return 24*60

def _sweep_warm_start(depot, stops, k) -> List[List[int]]:
    def ang(s): return math.atan2(s["lat"]-depot["lat"], s["lng"]-depot["lng"])
    sorted_stops = sorted(stops, key=ang)
    chunks = [sorted_stops[i::k] for i in range(k)]
    out=[]
    id_to_node = {s["id"]: i+1 for i,s in enumerate(stops)}
    for chunk in chunks:
        path=[0]+[id_to_node[s["id"]] for s in chunk]+[0]
        out.append(path)
    return out

#!/usr/bin/env python3
"""
RouteLoom evaluation harness.

Generates random VRP instances, runs:
  - a "quick" OR-Tools solve (your default config)
  - a "long" OR-Tools solve (reference/best-known proxy)
  - a naive greedy baseline

Computes per-instance metrics:
  - served_stop_rate (% of stops visited)
  - on_time_rate (% of visited stops inside time window)
  - total_distance_km, total_drive_min, total_co2_kg
  - runtime_sec
  - optimality_gap_% = (quick_distance - long_distance) / long_distance * 100
  - improvement_vs_greedy_% = (greedy_distance - quick_distance) / greedy_distance * 100

Writes a CSV and prints summary stats.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, statistics, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Tuple, Optional
import pathlib

# Allow "from services..." imports when running from /scripts
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from services.ortools_solver import solve_vrp  # type: ignore


# ----------------------- helpers -----------------------

def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = (math.sin(dlat/2)**2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(x))


@dataclass
class Depot:
    id: str
    lat: float
    lng: float


@dataclass
class Vehicle:
    id: str
    capacity: int
    fuel_type: str = "diesel"


@dataclass
class Stop:
    id: str
    lat: float
    lng: float
    demand: int
    priority: int = 1
    service_min: int = 5
    time_window: Optional[Dict[str, str]] = None  # {"start":"HH:MM:SS","end":"HH:MM:SS"}


@dataclass
class Instance:
    depot: Depot
    vehicles: List[Vehicle]
    stops: List[Stop]
    speed_kmph: float


# ---------------- instance generation ------------------

def random_instance(
    n_stops: int = 12,
    n_vehicles: int = 2,
    seed: int = 42,
    center_lat: float = 42.3601,  # Boston-ish
    center_lng: float = -71.0589,
    radius_km: float = 15.0,
    demand_lo: int = 80,
    demand_hi: int = 220,
    veh_capacity: int = 900,
    window_fraction: float = 0.5,
    window_span_min: Tuple[int, int] = (120, 240),
    speed_kmph: float = 35.0,
) -> Instance:
    rnd = random.Random(seed)
    # approx degrees per km near Boston
    deg_lat_per_km = 1.0 / 111.0
    deg_lng_per_km = 1.0 / (111.0 * math.cos(math.radians(center_lat)))

    def jitter():
        r = rnd.uniform(0, radius_km)
        theta = rnd.uniform(0, 2 * math.pi)
        return (center_lat + r * deg_lat_per_km * math.cos(theta),
                center_lng + r * deg_lng_per_km * math.sin(theta))

    depot_lat, depot_lng = jitter()
    depot = Depot(id="DEPOT", lat=depot_lat, lng=depot_lng)

    vehicles = []
    fuels = ["diesel", "ev"]
    for i in range(n_vehicles):
        vehicles.append(Vehicle(id=f"T{i+1}", capacity=veh_capacity, fuel_type=fuels[i % len(fuels)]))

    stops = []
    now = datetime.now(timezone.utc)
    for i in range(n_stops):
        lat, lng = jitter()
        demand = rnd.randint(demand_lo, demand_hi)
        prio = rnd.choice([1, 1, 2, 2, 3])  # skew lower, with some high priority
        svc = rnd.choice([4, 5, 6, 8])
        tw = None
        if rnd.random() < window_fraction:
            span = rnd.randint(*window_span_min)
            # Create a same-day window in HH:MM:SS (solver interprets as today)
            start_minutes = rnd.randint(8*60, 18*60)  # between 08:00 and 18:00
            end_minutes = min(start_minutes + span, 22*60)  # end by 22:00
            def mm(m): return f"{m//60:02d}:{m%60:02d}:00"
            tw = {"start": mm(start_minutes), "end": mm(end_minutes)}
        stops.append(Stop(id=f"S{i+1}", lat=lat, lng=lng, demand=demand, priority=prio, service_min=svc, time_window=tw))

    return Instance(depot=depot, vehicles=vehicles, stops=stops, speed_kmph=speed_kmph)


# --------------- metrics & evaluation ------------------

def build_node_map(stops: List[Stop]) -> Dict[int, Stop]:
    # node index 0 = depot, stop j => node j+1
    return {j+1: s for j, s in enumerate(stops)}

def window_ok(arrival_min: int, window: Optional[Dict[str, str]], base_start: datetime) -> bool:
    if not window:
        return True
    # Interpret solver semantics: arrival minutes since base_start; compare to same-day window
    def to_minutes(s: str) -> int:
        h, m, sec = map(int, s.split(":"))
        return h*60 + m
    start = to_minutes(window["start"])
    end = to_minutes(window["end"])
    return (arrival_min >= start) and (arrival_min <= end)

def extract_metrics(plan: Dict[str, Any], inst: Instance) -> Dict[str, Any]:
    # Sum distance/co2 from plan; compute served stops & on-time %
    total_distance = float(plan["summary"]["total_distance_km"])
    total_drive_min = int(plan["summary"]["total_drive_min"])
    total_co2 = sum(float(r["co2_kg"]) for r in plan["routes"])

    node_to_stop = build_node_map(inst.stops)
    visited: Dict[str, int] = {}
    on_time_hits = 0
    visited_count = 0

    # We don't have base_start in plan; approximate "same-day minutes" using arrival t_min
    base_start = datetime.now(timezone.utc)

    for rte in plan["routes"]:
        for st in rte["stops"]:
            node = st["node"]
            if node == 0:  # depot
                continue
            s = node_to_stop[node]
            visited[s.id] = st["t_min"]
            visited_count += 1
            if window_ok(st["t_min"], s.time_window, base_start):
                on_time_hits += 1

    served_stop_rate = visited_count / max(1, len(inst.stops))
    on_time_rate = on_time_hits / max(1, visited_count)
    dropped = len(inst.stops) - visited_count

    return {
        "served_stop_rate": round(served_stop_rate, 4),
        "on_time_rate": round(on_time_rate, 4),
        "dropped_count": dropped,
        "total_distance_km": round(total_distance, 2),
        "total_drive_min": total_drive_min,
        "total_co2_kg": round(total_co2, 2),
    }


def greedy_baseline(inst: Instance) -> Dict[str, Any]:
    """
    Very naive: assign each vehicle a route by nearest-neighbor from depot,
    respecting capacity only (ignores time windows); returns distance & served stops.
    """
    remaining = set(s.id for s in inst.stops)
    coord = {s.id: (s.lat, s.lng) for s in inst.stops}
    demand = {s.id: s.demand for s in inst.stops}
    depot_xy = (inst.depot.lat, inst.depot.lng)

    routes: List[List[str]] = [[] for _ in inst.vehicles]
    loads = [0 for _ in inst.vehicles]
    total_distance = 0.0

    # Pre-allocate current pos for each vehicle at depot
    positions = [depot_xy for _ in inst.vehicles]

    while remaining:
        progress = False
        for v_idx, veh in enumerate(inst.vehicles):
            # pick nearest feasible stop
            best_id, best_dist = None, float("inf")
            for sid in list(remaining):
                if loads[v_idx] + demand[sid] > veh.capacity:
                    continue
                d = haversine_km(positions[v_idx], coord[sid])
                if d < best_dist:
                    best_dist, best_id = d, sid
            if best_id is None:
                continue
            # move vehicle
            total_distance += best_dist
            positions[v_idx] = coord[best_id]
            loads[v_idx] += demand[best_id]
            routes[v_idx].append(best_id)
            remaining.remove(best_id)
            progress = True
        if not progress:
            # nobody can take remaining stops -> drop them (baseline limitation)
            break

    # return to depot
    for v_idx in range(len(inst.vehicles)):
        total_distance += haversine_km(positions[v_idx], depot_xy) if routes[v_idx] else 0.0

    co2 = 0.0
    for v_idx, veh in enumerate(inst.vehicles):
        route_dist = 0.0
        pos = depot_xy
        for sid in routes[v_idx]:
            route_dist += haversine_km(pos, coord[sid])
            pos = coord[sid]
        route_dist += haversine_km(pos, depot_xy) if routes[v_idx] else 0.0
        factor = 0.12 if veh.fuel_type.lower() == "ev" else 0.82
        co2 += route_dist * factor

    visited = sum(len(r) for r in routes)
    return {
        "routes": routes,
        "visited_count": visited,
        "dropped_count": len(inst.stops) - visited,
        "total_distance_km": total_distance,
        "total_co2_kg": co2,
    }


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=10)
    ap.add_argument("--stops", type=int, default=12)
    ap.add_argument("--vehicles", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--time_limit_quick", type=int, default=6)
    ap.add_argument("--time_limit_long", type=int, default=20)
    ap.add_argument("--speed_kmph", type=float, default=35.0)
    ap.add_argument("--allow_drop", type=int, default=1, help="1 or 0")

    # NEW: feasibility controls
    ap.add_argument("--veh_capacity", type=int, default=900)
    ap.add_argument("--demand_lo", type=int, default=80)
    ap.add_argument("--demand_hi", type=int, default=220)

    # NEW: penalty strength
    ap.add_argument("--drop_penalty_quick", type=int, default=5000)
    ap.add_argument("--drop_penalty_long", type=int, default=8000)

    # NEW: parallelism
    ap.add_argument("--jobs", type=int, default=1, help="parallel workers")
    
    # Access-aware evaluation
    ap.add_argument("--use_access", type=int, default=0, help="1 to enable access analysis")
    ap.add_argument("--access_penalty_weight", type=float, default=0.002)
    ap.add_argument("--drop_penalty_weight", type=float, default=0.02)

    ap.add_argument("--csv", type=str, default=None, help="path to write CSV report")
    args = ap.parse_args()

    rnd = random.Random(args.seed)
    rows = []

    print(f"Running {args.instances} instances…")
    for i in range(args.instances):
        inst = random_instance(
            n_stops=args.stops,
            n_vehicles=args.vehicles,
            seed=rnd.randint(1, 1_000_000),
            speed_kmph=args.speed_kmph,
            veh_capacity=args.veh_capacity,      # NEW
            demand_lo=args.demand_lo,            # NEW
            demand_hi=args.demand_hi,            # NEW
        )

        # greedy baseline
        base = greedy_baseline(inst)
        base_dist = base["total_distance_km"]

        # quick solve
        t0 = time.time()
        quick = solve_vrp(
            depot=asdict(inst.depot),
            stops=[asdict(s) for s in inst.stops],
            vehicles=[asdict(v) for v in inst.vehicles],
            speed_kmph=inst.speed_kmph,
            time_limit_sec=args.time_limit_quick,
            allow_drop=bool(args.allow_drop),
            default_service_min=5,
            drop_penalty_per_priority=args.drop_penalty_quick,   # NEW
            debug_log=False,
            use_access_scores=bool(args.use_access),
            access_penalty_weight=args.access_penalty_weight,
            drop_penalty_weight=args.drop_penalty_weight
        )
        t1 = time.time()
        if not quick.get("ok"):
            print(f"[{i+1}] quick solver failed: {quick}")
            continue
        quick_m = extract_metrics(quick, inst)
        quick_runtime = t1 - t0

        # long solve (best-known proxy)
        t2 = time.time()
        long = solve_vrp(
            depot=asdict(inst.depot),
            stops=[asdict(s) for s in inst.stops],
            vehicles=[asdict(v) for v in inst.vehicles],
            speed_kmph=inst.speed_kmph,
            time_limit_sec=args.time_limit_long,
            allow_drop=bool(args.allow_drop),
            default_service_min=5,
            drop_penalty_per_priority=args.drop_penalty_long,    # NEW
            debug_log=False,
            use_access_scores=bool(args.use_access),
            access_penalty_weight=args.access_penalty_weight,
            drop_penalty_weight=args.drop_penalty_weight
        )
        t3 = time.time()
        if not long.get("ok"):
            print(f"[{i+1}] long solver failed: {long}")
            continue
        long_m = extract_metrics(long, inst)
        long_runtime = t3 - t2

        # gaps & improvements
        opt_gap = None
        if long_m["total_distance_km"] > 1e-6:
            opt_gap = 100.0 * (quick_m["total_distance_km"] - long_m["total_distance_km"]) / long_m["total_distance_km"]
        improv_vs_greedy = None
        if base_dist > 1e-6:
            improv_vs_greedy = 100.0 * (base_dist - quick_m["total_distance_km"]) / base_dist

        # Track which stops were dropped for clarity
        all_ids = {s.id for s in inst.stops}
        visited_ids = set()
        for rte in quick["routes"]:
            for st in rte["stops"]:
                if st["node"] > 0:
                    visited_ids.add(inst.stops[st["node"]-1].id)
        dropped_ids = sorted(all_ids - visited_ids)

        row = {
            "idx": i+1,
            "stops": args.stops,
            "vehicles": args.vehicles,
            "quick_distance_km": quick_m["total_distance_km"],
            "long_distance_km": long_m["total_distance_km"],
            "greedy_distance_km": round(base_dist, 2),
            "quick_served_rate": quick_m["served_stop_rate"],
            "quick_on_time_rate": quick_m["on_time_rate"],
            "quick_dropped": quick_m["dropped_count"],
            "quick_total_co2_kg": quick_m["total_co2_kg"],
            "quick_runtime_sec": round(quick_runtime, 3),
            "long_runtime_sec": round(long_runtime, 3),
            "opt_gap_percent": round(opt_gap, 2) if opt_gap is not None else None,
            "improv_vs_greedy_percent": round(improv_vs_greedy, 2) if improv_vs_greedy is not None else None,
            "quick_dropped_ids": "|".join(dropped_ids[:10]),  # limit for CSV
        }
        rows.append(row)
        print(f"[{i+1}] quick {row['quick_distance_km']} km in {row['quick_runtime_sec']}s | "
              f"served {row['quick_served_rate']*100:.0f}% | on-time {row['quick_on_time_rate']*100:.0f}% "
              f"| gap {row['opt_gap_percent']}% | vs greedy +{row['improv_vs_greedy_percent']}%")

    if not rows:
        print("No successful runs.")
        return

    # summary
    def col(name): 
        return [r[name] for r in rows if r[name] is not None]

    def mean(name): 
        v = col(name); 
        return round(statistics.mean(v), 3) if v else None

    print("\n=== Summary ===")
    print(f"instances: {len(rows)}")
    print(f"avg quick distance (km): {mean('quick_distance_km')}")
    print(f"avg quick runtime (s):  {mean('quick_runtime_sec')}")
    print(f"avg served rate:        {mean('quick_served_rate')}")
    print(f"avg on-time rate:       {mean('quick_on_time_rate')}")
    print(f"avg dropped stops:      {mean('quick_dropped')}")
    print(f"avg CO2 (kg):           {mean('quick_total_co2_kg')}")
    print(f"avg opt gap (%):        {mean('opt_gap_percent')}")
    print(f"avg improve vs greedy:  {mean('improv_vs_greedy_percent')}")

    # CSV
    out_csv = args.csv or f"eval_report_{int(time.time())}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_csv}")
    print("Done.")
    

if __name__ == "__main__":
    main()

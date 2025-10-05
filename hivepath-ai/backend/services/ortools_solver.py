from typing import List, Dict, Any, Tuple, Optional
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from datetime import datetime, timezone
import math

CO2_KG_PER_KM = {"diesel":0.82, "gas":0.75, "ev":0.12, "default":0.80}

def _haversine_km(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    R=6371.0
    lat1, lon1 = map(math.radians, a); lat2, lon2 = map(math.radians, b)
    dlat = lat2-lat1; dlon = lon2-lon1
    x = (math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
    return 2*R*math.asin(math.sqrt(x))

def _parse_iso_minutes(base_start: datetime, iso_str: str) -> int:
    """Return minutes from base_start to the given ISO string.
       If string omits date, interpret as TODAY in base_start's timezone."""
    s = iso_str
    if "T" not in s:  # e.g., "12:00:00" -> add today's date
        s = base_start.date().isoformat() + "T" + s
    try:
        t = datetime.fromisoformat(s.replace("Z","+00:00"))
        return max(0, int((t - base_start).total_seconds() // 60))
    except:
        # If parsing fails, return a reasonable default
        return 0

def _time_window_minutes(base_start: datetime, stop: Dict[str, Any]) -> Tuple[int,int]:
    tw = stop.get("time_window")
    if not tw: return (0, 24*60)
    start = tw.get("start"); end = tw.get("end")
    if not start or not end: return (0, 24*60)
    return (_parse_iso_minutes(base_start, start), _parse_iso_minutes(base_start, end))

def _warm_assignment_from_routes(routing, manager, routes: list[list[int]]):
    """Convert node-index routes to OR-Tools assignment for warm start."""
    # routes = [[0, 5, 2, 7, 0], [0, 3, 6, 1, 0], ...] node indices incl depot
    return routing.ReadAssignmentFromRoutes(routes, True)

def plan_to_routes_for_warm_start(plan: dict) -> list[list[int]]:
    """Convert PlanV1 to node lists for warm start."""
    # PlanV1 uses 0 = depot, others are already node ids in our build.
    routes = []
    for r in plan["routes"]:
        path = [s["node"] for s in r["stops"]]
        if path and path[0] != 0:
            path = [0] + path
        if not path or path[-1] != 0:
            path = path + [0]
        routes.append(path)
    return routes

def solve_vrp(
    depot: Dict[str, Any],
    stops: List[Dict[str, Any]],
    vehicles: List[Dict[str, Any]],
    speed_kmph: float = 40.0,
    blocked_stop_ids: Optional[set[str]] = None,
    # NEW knobs:
    time_limit_sec: int = 8,
    num_workers: int = 0,          # NEW: 0 = auto
    default_service_min: int = 5,
    allow_drop: bool = True,
    drop_penalty_per_priority: int = 5000,
    debug_log: bool = False,
    initial_routes: Optional[List[List[int]]] = None,  # NEW: warm start
    # Access-aware parameters
    use_access_scores: bool = True,
    access_penalty_weight: float = 0.002,  # γ in objective
    drop_penalty_weight: float = 0.02,     # δ in objective
) -> Dict[str, Any]:
    if not vehicles: return {"ok": False, "error": "no_vehicles"}
    if not stops: return {"ok": False, "error": "no_stops"}

    blocked_stop_ids = blocked_stop_ids or set()
    base_start = datetime.now(timezone.utc).replace(microsecond=0)

    all_nodes = [depot] + stops
    n_nodes, n_vehicles, depot_index = len(all_nodes), len(vehicles), 0

    # Distances & travel times
    dist_km = [[0]*n_nodes for _ in range(n_nodes)]
    time_min = [[0]*n_nodes for _ in range(n_nodes)]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i==j: continue
            d = _haversine_km((all_nodes[i]["lat"], all_nodes[i]["lng"]),
                              (all_nodes[j]["lat"], all_nodes[j]["lng"]))
            dist_km[i][j] = d
            time_min[i][j] = max(1, int((d / max(1e-6, speed_kmph)) * 60))

    # Demands & per-node service time
    demand = [0] + [int(s.get("demand",0)) for s in stops]
    service_time = [0] + [int(stops[k].get("service_min", default_service_min)) for k in range(len(stops))]

    # Time windows
    TW = [(0, 24*60)]
    for s in stops:
        if s["id"] in blocked_stop_ids:
            TW.append((23*60+50, 23*60+50))  # effectively impossible
        else:
            TW.append(_time_window_minutes(base_start, s))

    # Model
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # Transit callback includes travel + service time at FROM node
    def time_cb(from_i, to_i):
        from_node = manager.IndexToNode(from_i)
        to_node   = manager.IndexToNode(to_i)
        return time_min[from_node][to_node] + service_time[from_node]
    transit_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Time dimension
    routing.AddDimension(
        transit_idx,
        60,            # slack
        24*60,         # max per vehicle (one-day horizon)
        False,
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Capacity dimension
    def demand_cb(idx):
        node = manager.IndexToNode(idx)
        return demand[node]
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [int(v.get("capacity", 1000)) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, capacities, True, "Capacity")

    # Apply time windows on customer nodes
    for node in range(1, n_nodes):
        index = manager.NodeToIndex(node)
        start, end = TW[node]
        time_dim.CumulVar(index).SetRange(start, end)

    # Vehicles start at depot at time 0
    for v in range(n_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(0, 0)

    # Allow dropping customers (with penalty) so we don't get "no_solution"
    if allow_drop:
        for node in range(1, n_nodes):
            stop = stops[node-1]
            priority = int(max(1, stop.get("priority", 1)))
            base_penalty = drop_penalty_per_priority * priority
            
            # Add access-based penalty if available
            if use_access_scores and "access_score" in stop:
                access_score = int(stop.get("access_score", 50))
                access_penalty = int(access_penalty_weight * (100 - access_score))
                penalty = base_penalty + access_penalty
            else:
                penalty = base_penalty
                
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search params
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromSeconds(time_limit_sec)
        # Note: number_of_search_workers not available in this OR-Tools version
        if debug_log:
            params.log_search = True

    # Try warm start if provided
    assignment = None
    if initial_routes:
        assignment = _warm_assignment_from_routes(routing, manager, initial_routes)

    if assignment:
        solution = routing.SolveFromAssignmentWithParameters(assignment, params)
    else:
        solution = routing.SolveWithParameters(params)
    if not solution:
        return {"ok": False, "error": "no_solution"}

    # Build plan
    plan = {"ok": True, "routes": [], "summary": {}}
    total_km = total_min = total_demand = 0.0

    def v_co2_factor(v): return CO2_KG_PER_KM.get((v.get("fuel_type") or "default").lower(), CO2_KG_PER_KM["default"])

    for v in range(n_vehicles):
        index = routing.Start(v)
        route_nodes, route_km, route_min, load = [], 0.0, 0, 0
        while not routing.IsEnd(index):
            node_id = manager.IndexToNode(index)
            t_cumul = solution.Value(time_dim.CumulVar(index))
            route_nodes.append({"node": node_id, "t_min": int(t_cumul)})
            nxt = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(nxt):
                next_id = manager.IndexToNode(nxt)
                route_km += dist_km[node_id][next_id]
                route_min += time_min[node_id][next_id]
                if next_id > 0: load += demand[next_id]
            index = nxt

        plan["routes"].append({
            "vehicle_id": vehicles[v]["id"],
            "stops": route_nodes,
            "distance_km": round(route_km, 2),
            "drive_min": int(route_min),
            "load": int(load),
            "co2_kg": round(route_km * v_co2_factor(vehicles[v]), 2)
        })
        total_km += route_km; total_min += route_min; total_demand += load

    plan["summary"] = {
        "total_distance_km": round(total_km, 2),
        "total_drive_min": int(total_min),
        "total_served_demand": int(total_demand),
        "start_iso": base_start.isoformat(),
        "dropped_allowed": allow_drop
    }
    
    # Add telemetry for ML training
    plan["_telemetry"] = {
        "time_limit_sec": time_limit_sec,
        "num_workers": num_workers,
        "allow_drop": allow_drop,
        "drop_penalty_per_priority": drop_penalty_per_priority,
        "nodes": len(stops),
        "vehicles": len(vehicles)
    }
    
    return plan
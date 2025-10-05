"""
Enhanced VRP solver for multi-location routing scenarios
"""
from typing import List, Dict, Any, Tuple, Optional
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from datetime import datetime, timezone
import math
from functools import lru_cache

def solve_multi_location_vrp(
    locations: List[Dict[str, Any]],
    vehicles: List[Dict[str, Any]],
    speed_kmph: float = 40.0,
    pickup_delivery_pairs: List[Dict[str, str]] = [],
    sequences: List[Dict[str, Any]] = [],
    time_limit_sec: int = 8,
    num_workers: int = 0,
    allow_drop: bool = True,
    drop_penalty_per_priority: int = 5000,
    debug_log: bool = False
) -> Dict[str, Any]:
    """
    Solve multi-location VRP with pickup-delivery pairs and location dependencies
    """
    
    if not vehicles:
        return {"ok": False, "error": "no_vehicles"}
    if not locations:
        return {"ok": False, "error": "no_locations"}
    
    # Separate depots and stops
    depots = [loc for loc in locations if loc.get("location_type") == "depot"]
    stops = [loc for loc in locations if loc.get("location_type") != "depot"]
    
    if not depots:
        return {"ok": False, "error": "no_depot"}
    
    # Use first depot as main depot
    main_depot = depots[0]
    all_nodes = [main_depot] + stops
    n_nodes = len(all_nodes)
    n_vehicles = len(vehicles)
    depot_index = 0
    
    # Compute distance matrix
    dist_km, time_min = _compute_distance_matrix_cached(all_nodes, speed_kmph)
    
    # Create pickup-delivery constraints
    pickup_delivery_constraints = _create_pickup_delivery_constraints(
        pickup_delivery_pairs, stops, all_nodes
    )
    
    # Create location dependencies
    dependency_constraints = _create_dependency_constraints(stops, all_nodes)
    
    # Setup OR-Tools model
    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)
    
    # Time callback
    def time_cb(from_i, to_i):
        from_node = manager.IndexToNode(from_i)
        to_node = manager.IndexToNode(to_i)
        service_time = stops[from_node-1].get("service_min", 5) if from_node > 0 else 0
        return time_min[from_node][to_node] + service_time
    
    transit_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)
    
    # Time dimension
    routing.AddDimension(
        transit_idx,
        60,  # slack
        24*60,  # max time
        False,
        "Time"
    )
    time_dim = routing.GetDimensionOrDie("Time")
    
    # Capacity dimension
    def demand_cb(idx):
        node = manager.IndexToNode(idx)
        if node == 0:
            return 0
        return stops[node-1].get("demand", 0)
    
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [int(v.get("capacity", 1000)) for v in vehicles]
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, capacities, True, "Capacity")
    
    # Apply time windows
    for node in range(1, n_nodes):
        index = manager.NodeToIndex(node)
        stop = stops[node-1]
        tw = stop.get("time_window")
        if tw:
            start_min = _parse_time_to_minutes(tw.get("start", "00:00:00"))
            end_min = _parse_time_to_minutes(tw.get("end", "23:59:59"))
            time_dim.CumulVar(index).SetRange(start_min, end_min)
        else:
            time_dim.CumulVar(index).SetRange(0, 24*60)
    
    # Vehicles start at depot
    for v in range(n_vehicles):
        time_dim.CumulVar(routing.Start(v)).SetRange(0, 0)
    
    # Add pickup-delivery constraints
    for pickup_idx, delivery_idx in pickup_delivery_constraints:
        pickup_index = manager.NodeToIndex(pickup_idx)
        delivery_index = manager.NodeToIndex(delivery_idx)
        
        # Same vehicle constraint
        routing.AddPickupAndDelivery(pickup_index, delivery_index)
        
        # Precedence constraint (pickup before delivery)
        pickup_time = time_dim.CumulVar(pickup_index)
        delivery_time = time_dim.CumulVar(delivery_index)
        routing.solver().Add(pickup_time <= delivery_time)
    
    # Add dependency constraints
    for dependent_idx, dependency_idx in dependency_constraints:
        dependent_index = manager.NodeToIndex(dependent_idx)
        dependency_index = manager.NodeToIndex(dependency_idx)
        
        # Same vehicle constraint
        routing.AddPickupAndDelivery(dependency_index, dependent_index)
        
        # Precedence constraint
        dependency_time = time_dim.CumulVar(dependency_index)
        dependent_time = time_dim.CumulVar(dependent_index)
        routing.solver().Add(dependency_time <= dependent_time)
    
    # Allow dropping with penalties
    if allow_drop:
        for node in range(1, n_nodes):
            stop = stops[node-1]
            priority = stop.get("priority", 1)
            penalty = drop_penalty_per_priority * priority
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(time_limit_sec)
    
    if hasattr(params, 'number_of_search_workers'):
        params.number_of_search_workers = min(4, num_workers) if num_workers > 0 else 4
    
    if debug_log:
        params.log_search = True
    
    # Solve
    solution = routing.SolveWithParameters(params)
    if not solution:
        return {"ok": False, "error": "no_solution"}
    
    # Build result
    plan = {"ok": True, "routes": [], "summary": {}}
    total_km = total_min = total_demand = 0.0
    
    for v in range(n_vehicles):
        index = routing.Start(v)
        route_nodes, route_km, route_min, load = [], 0.0, 0, 0
        location_sequence = []
        
        while not routing.IsEnd(index):
            node_id = manager.IndexToNode(index)
            t_cumul = solution.Value(time_dim.CumulVar(index))
            
            if node_id == 0:
                location_info = {"node": node_id, "t_min": int(t_cumul), "location_type": "depot"}
            else:
                stop = stops[node_id-1]
                location_info = {
                    "node": node_id,
                    "t_min": int(t_cumul),
                    "location_type": stop.get("location_type", "service"),
                    "group_id": stop.get("group_id"),
                    "dependencies": stop.get("dependencies", [])
                }
            
            route_nodes.append(location_info)
            location_sequence.append(stop.get("id", f"node_{node_id}") if node_id > 0 else "depot")
            
            nxt = solution.Value(routing.NextVar(index))
            if not routing.IsEnd(nxt):
                next_id = manager.IndexToNode(nxt)
                route_km += dist_km[node_id][next_id]
                route_min += time_min[node_id][next_id]
                if next_id > 0:
                    load += stops[next_id-1].get("demand", 0)
            index = nxt
        
        plan["routes"].append({
            "vehicle_id": vehicles[v]["id"],
            "stops": route_nodes,
            "location_sequence": location_sequence,
            "distance_km": round(route_km, 2),
            "drive_min": int(route_min),
            "load": int(load)
        })
        total_km += route_km
        total_min += route_min
        total_demand += load
    
    plan["summary"] = {
        "total_distance_km": round(total_km, 2),
        "total_drive_min": int(total_min),
        "total_served_demand": int(total_demand),
        "pickup_delivery_pairs": len(pickup_delivery_pairs),
        "dependency_constraints": len(dependency_constraints)
    }
    
    return plan

@lru_cache(maxsize=10000)
def _haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Cached haversine distance calculation"""
    R = 6371.0
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    x = (math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(x))

def _compute_distance_matrix_cached(all_nodes: List[Dict[str, Any]], speed_kmph: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Compute distance and time matrices with caching"""
    n_nodes = len(all_nodes)
    dist_km = [[0]*n_nodes for _ in range(n_nodes)]
    time_min = [[0]*n_nodes for _ in range(n_nodes)]
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            d = _haversine_km((all_nodes[i]["lat"], all_nodes[i]["lng"]),
                            (all_nodes[j]["lat"], all_nodes[j]["lng"]))
            dist_km[i][j] = d
            time_min[i][j] = max(1, int((d / max(1e-6, speed_kmph)) * 60))
    
    return dist_km, time_min

def _create_pickup_delivery_constraints(pairs: List[Dict[str, str]], stops: List[Dict[str, Any]], all_nodes: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Create pickup-delivery constraint pairs"""
    constraints = []
    id_to_index = {node["id"]: i for i, node in enumerate(all_nodes)}
    
    for pair in pairs:
        pickup_id = pair.get("pickup")
        delivery_id = pair.get("delivery")
        
        if pickup_id in id_to_index and delivery_id in id_to_index:
            pickup_idx = id_to_index[pickup_id]
            delivery_idx = id_to_index[delivery_id]
            constraints.append((pickup_idx, delivery_idx))
    
    return constraints

def _create_dependency_constraints(stops: List[Dict[str, Any]], all_nodes: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Create location dependency constraints"""
    constraints = []
    id_to_index = {node["id"]: i for i, node in enumerate(all_nodes)}
    
    for stop in stops:
        stop_idx = id_to_index.get(stop["id"])
        if stop_idx and stop.get("dependencies"):
            for dep_id in stop["dependencies"]:
                dep_idx = id_to_index.get(dep_id)
                if dep_idx:
                    constraints.append((dep_idx, stop_idx))
    
    return constraints

def _parse_time_to_minutes(time_str: str) -> int:
    """Parse time string to minutes since midnight"""
    try:
        h, m, s = map(int, time_str.split(":"))
        return h * 60 + m
    except:
        return 0



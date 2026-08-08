"""Warm-start route construction.

OR-Tools' ``ReadAssignmentFromRoutes`` expects each route to list **only the
customer nodes**; the start and end depot are implicit in the routing model.
The previous implementation prepended and appended node ``0`` to every route,
so OR-Tools rejected the assignment with ``Index 0 is used multiple times`` and
returned ``None``. Because the caller treated ``None`` as "no warm start
available", every solve silently ran cold and the failure never surfaced.
"""

from __future__ import annotations

import math
from typing import Sequence

from hivepath.domain import Depot, Stop, Vehicle
from hivepath.logging_config import get_logger

logger = get_logger(__name__)

DEPOT_NODE = 0


def strip_depot(route: Sequence[int]) -> list[int]:
    """Remove depot sentinels from a route, preserving customer order."""
    return [node for node in route if node != DEPOT_NODE]


def validate_routes(routes: Sequence[Sequence[int]], node_count: int) -> list[list[int]]:
    """Return routes safe to hand to ``ReadAssignmentFromRoutes``.

    Drops the depot, rejects out-of-range nodes, and de-duplicates: OR-Tools
    requires that each customer appear at most once across all routes.
    """
    seen: set[int] = set()
    cleaned: list[list[int]] = []

    for route in routes:
        kept: list[int] = []
        for node in strip_depot(route):
            if not 1 <= node < node_count:
                logger.warning("warm start dropped out-of-range node %s", node)
                continue
            if node in seen:
                logger.warning("warm start dropped duplicate node %s", node)
                continue
            seen.add(node)
            kept.append(node)
        cleaned.append(kept)

    return cleaned


def sweep_routes(depot: Depot, stops: Sequence[Stop], vehicle_count: int) -> list[list[int]]:
    """Classic sweep heuristic: order stops by bearing, deal into vehicles.

    Cheap, dependency-free, and a decent starting point for the local search.
    """
    if vehicle_count < 1:
        return []
    if not stops:
        return [[] for _ in range(vehicle_count)]

    node_of = {stop.id: index + 1 for index, stop in enumerate(stops)}

    def bearing(stop: Stop) -> float:
        return math.atan2(stop.lat - depot.lat, stop.lng - depot.lng)

    ordered = sorted(stops, key=bearing)
    # Contiguous angular slices keep each vehicle in one sector, which is a
    # better seed than dealing round-robin across the whole circle.
    per_vehicle = math.ceil(len(ordered) / vehicle_count)
    routes: list[list[int]] = []
    for index in range(vehicle_count):
        chunk = ordered[index * per_vehicle : (index + 1) * per_vehicle]
        routes.append([node_of[stop.id] for stop in chunk])
    return routes


def capacity_aware_routes(
    depot: Depot, stops: Sequence[Stop], vehicles: Sequence[Vehicle]
) -> list[list[int]]:
    """Sweep, then repair any route that exceeds its vehicle's capacity.

    A warm start that violates capacity is still legal input to OR-Tools, but a
    feasible seed converges faster.
    """
    routes = sweep_routes(depot, stops, len(vehicles))
    demand_of = {index + 1: stop.demand for index, stop in enumerate(stops)}

    overflow: list[int] = []
    for route, vehicle in zip(routes, vehicles):
        load = 0
        kept: list[int] = []
        for node in route:
            if load + demand_of[node] <= vehicle.capacity:
                kept.append(node)
                load += demand_of[node]
            else:
                overflow.append(node)
        route[:] = kept

    # Re-home displaced stops wherever capacity remains.
    for node in list(overflow):
        for route, vehicle in zip(routes, vehicles):
            load = sum(demand_of[n] for n in route)
            if load + demand_of[node] <= vehicle.capacity:
                route.append(node)
                overflow.remove(node)
                break

    if overflow:
        # Genuinely more demand than fleet capacity. The solver will drop these
        # via disjunction; leaving them out of the seed is correct.
        logger.info(
            "warm start left %d stop(s) unassigned: fleet capacity is insufficient",
            len(overflow),
        )

    return validate_routes(routes, node_count=len(stops) + 1)


def routes_from_plan(plan_dict: dict) -> list[list[int]]:
    """Extract warm-start routes from a previously saved plan payload."""
    routes: list[list[int]] = []
    for route in plan_dict.get("routes", []):
        nodes = [s.get("node") for s in route.get("stops", []) if s.get("node") is not None]
        routes.append(strip_depot(nodes))
    return routes

"""Capacitated vehicle routing with time windows, solved with OR-Tools.

The objective is total vehicle time (travel plus service). Stops may be dropped
via disjunctions when the fleet cannot serve everything; the cost of dropping is
set by :mod:`hivepath.optimization.penalties`, which is where accessibility
enters the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from hivepath.config import Settings, get_settings
from hivepath.domain import (
    Depot,
    Plan,
    PlanSummary,
    Route,
    RouteStop,
    Stop,
    TimeWindow,
    Vehicle,
)
from hivepath.logging_config import get_logger
from hivepath.optimization.distance import DistanceMatrix, build_distance_matrix
from hivepath.optimization.penalties import drop_penalty
from hivepath.optimization.warm_start import capacity_aware_routes, validate_routes

logger = get_logger(__name__)

DEPOT_NODE = 0
MINUTES_PER_DAY = 24 * 60
#: Slack allowed at each node, in minutes - lets a vehicle wait for a window.
TIME_SLACK_MIN = 60


@dataclass(slots=True)
class SolverOptions:
    """Tunables for a single solve."""

    speed_kmph: float = 40.0
    time_limit_sec: int = 8
    num_workers: int = 8
    default_service_min: int = 5
    allow_drop: bool = True
    drop_penalty_per_priority: int = 5000
    use_access_scores: bool = True
    access_penalty_weight: float = 0.002
    use_google_maps: bool = False
    use_warm_start: bool = True
    debug_log: bool = False

    def __post_init__(self) -> None:
        if self.time_limit_sec < 1:
            raise ValueError(f"time_limit_sec must be >= 1, got {self.time_limit_sec}")
        if self.speed_kmph <= 0:
            raise ValueError(f"speed_kmph must be > 0, got {self.speed_kmph}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")

    @classmethod
    def from_settings(cls, settings: Settings | None = None, **overrides: Any) -> SolverOptions:
        """Build options from configuration, with per-request overrides."""
        settings = settings or get_settings()
        base = {
            "speed_kmph": settings.solver_default_speed_kmph,
            "time_limit_sec": settings.solver_time_limit_sec,
            "num_workers": settings.solver_num_workers,
            "default_service_min": settings.solver_default_service_min,
            "drop_penalty_per_priority": settings.solver_drop_penalty_per_priority,
            "access_penalty_weight": settings.solver_access_penalty_weight,
        }
        base.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**base)


def _build_search_parameters(options: SolverOptions) -> pywrapcp.RoutingSearchParameters:
    """Search configuration.

    Built unconditionally. It previously sat inside ``if allow_drop:``, so
    ``allow_drop=False`` reached the solve call with ``params`` unbound.

    On ``num_workers``: OR-Tools' routing local search (the guided local search
    used here) is **single-threaded**, and ``RoutingSearchParameters`` exposes no
    worker count. The only place a worker count applies is the CP-SAT backend,
    so that is where it is set. Unless CP-SAT is enabled, raising this value
    changes nothing - it is wired here so the setting is not silently discarded,
    not because it parallelises the default search.
    """
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(options.time_limit_sec)

    if options.num_workers > 0:
        params.sat_parameters.num_workers = options.num_workers

    params.log_search = options.debug_log
    return params


def _service_minutes(stop: Stop, default_service_min: int) -> int:
    return int(stop.service_min if stop.service_min is not None else default_service_min)


def solve_vrp(
    depot: Depot,
    stops: Sequence[Stop],
    vehicles: Sequence[Vehicle],
    options: SolverOptions | None = None,
    *,
    blocked_stop_ids: set[str] | None = None,
    warm_start_routes: Sequence[Sequence[int]] | None = None,
    distance_matrix: DistanceMatrix | None = None,
    settings: Settings | None = None,
) -> Plan:
    """Plan routes for ``vehicles`` over ``stops`` from ``depot``.

    Blocked stops are excluded from the model outright and reported in
    :attr:`Plan.dropped_stop_ids`, rather than being given an unsatisfiable time
    window - the latter risks rendering the whole model infeasible.
    """
    options = options or SolverOptions.from_settings(settings)
    settings = settings or get_settings()
    blocked = set(blocked_stop_ids or ())

    if not vehicles:
        return Plan.failed("no_vehicles")
    if not stops:
        return Plan.failed("no_stops")

    routable = [s for s in stops if s.id not in blocked]
    excluded = [s.id for s in stops if s.id in blocked]
    if not routable:
        return Plan.failed("all_stops_blocked", blocked=excluded)

    started_at = datetime.now(timezone.utc).replace(microsecond=0)
    nodes: list[Depot | Stop] = [depot, *routable]
    node_count = len(nodes)
    vehicle_count = len(vehicles)

    matrix = distance_matrix or build_distance_matrix(
        [(n.lat, n.lng) for n in nodes],
        options.speed_kmph,
        use_google_maps=options.use_google_maps,
        settings=settings,
    )
    if matrix.size != node_count:
        raise ValueError(
            f"distance matrix has {matrix.size} nodes, expected {node_count}"
        )

    demands = [0] + [s.demand for s in routable]
    service = [0] + [_service_minutes(s, options.default_service_min) for s in routable]

    manager = pywrapcp.RoutingIndexManager(node_count, vehicle_count, DEPOT_NODE)
    routing = pywrapcp.RoutingModel(manager)

    # Arc cost is travel time plus the service performed at the origin node.
    def time_callback(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return matrix.duration_min[i][j] + service[i]

    transit_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)

    routing.AddDimension(
        transit_index,
        TIME_SLACK_MIN,
        MINUTES_PER_DAY,
        False,  # vehicles do not all start at time zero cumulatively
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    def demand_callback(index: int) -> int:
        return demands[manager.IndexToNode(index)]

    demand_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [v.capacity for v in vehicles],
        True,
        "Capacity",
    )

    for node in range(1, node_count):
        stop = routable[node - 1]
        window = stop.time_window or TimeWindow.full_day()
        time_dimension.CumulVar(manager.NodeToIndex(node)).SetRange(
            window.start_min, window.end_min
        )

    for vehicle_index in range(vehicle_count):
        time_dimension.CumulVar(routing.Start(vehicle_index)).SetRange(0, 0)

    if options.allow_drop:
        for node in range(1, node_count):
            stop = routable[node - 1]
            penalty = drop_penalty(
                stop.priority,
                penalty_per_priority=options.drop_penalty_per_priority,
                access_score=stop.access_score if options.use_access_scores else None,
                access_weight=options.access_penalty_weight,
            )
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    params = _build_search_parameters(options)

    assignment = None
    if options.use_warm_start:
        seed = (
            validate_routes(warm_start_routes, node_count)
            if warm_start_routes is not None
            else capacity_aware_routes(depot, routable, vehicles)
        )
        # OR-Tools needs exactly one route per vehicle.
        seed = (seed + [[] for _ in range(vehicle_count)])[:vehicle_count]
        if any(seed):
            assignment = routing.ReadAssignmentFromRoutes(seed, True)
            if assignment is None:
                logger.warning("warm start rejected by OR-Tools; solving from scratch")

    solution = (
        routing.SolveFromAssignmentWithParameters(assignment, params)
        if assignment is not None
        else routing.SolveWithParameters(params)
    )

    if solution is None:
        logger.warning(
            "no solution found",
            extra={"stops": len(routable), "vehicles": vehicle_count},
        )
        return Plan.failed(
            "no_solution",
            stops=len(routable),
            vehicles=vehicle_count,
            time_limit_sec=options.time_limit_sec,
        )

    plan = _extract_plan(
        routing=routing,
        manager=manager,
        solution=solution,
        time_dimension=time_dimension,
        matrix=matrix,
        routable=routable,
        vehicles=vehicles,
        demands=demands,
        started_at=started_at,
    )

    plan.dropped_stop_ids.extend(excluded)
    plan.summary.dropped_stops = len(plan.dropped_stop_ids)
    plan.telemetry = {
        "matrix_source": matrix.source,
        "warm_started": assignment is not None,
        "time_limit_sec": options.time_limit_sec,
        "num_workers": options.num_workers,
        "allow_drop": options.allow_drop,
        "drop_penalty_per_priority": options.drop_penalty_per_priority,
        "access_scores_applied": options.use_access_scores,
        "nodes": len(routable),
        "vehicles": vehicle_count,
        "blocked_stops": len(excluded),
    }

    logger.info(
        "solved %d/%d stops across %d vehicles (%.2f km, source=%s, warm=%s)",
        plan.summary.served_stops,
        len(stops),
        vehicle_count,
        plan.summary.total_distance_km,
        matrix.source,
        assignment is not None,
        extra={"telemetry": plan.telemetry},
    )
    return plan


def _extract_plan(
    *,
    routing: pywrapcp.RoutingModel,
    manager: pywrapcp.RoutingIndexManager,
    solution: Any,
    time_dimension: Any,
    matrix: DistanceMatrix,
    routable: Sequence[Stop],
    vehicles: Sequence[Vehicle],
    demands: Sequence[int],
    started_at: datetime,
) -> Plan:
    """Read an OR-Tools solution into the domain model."""
    routes: list[Route] = []
    visited: set[int] = set()
    summary = PlanSummary(start_iso=started_at.isoformat())

    for vehicle_index, vehicle in enumerate(vehicles):
        route = Route(vehicle_id=vehicle.id)
        index = routing.Start(vehicle_index)

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            visited.add(node)
            route.stops.append(
                RouteStop(
                    node=node,
                    stop_id=routable[node - 1].id if node != DEPOT_NODE else None,
                    arrival_min=int(solution.Value(time_dimension.CumulVar(index))),
                )
            )

            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            route.distance_km += matrix.distance_km[node][next_node]
            route.drive_min += matrix.duration_min[node][next_node]
            if next_node != DEPOT_NODE:
                route.load += demands[next_node]
            index = next_index

        route.co2_kg = route.distance_km * vehicle.co2_kg_per_km
        routes.append(route)

        summary.total_distance_km += route.distance_km
        summary.total_drive_min += route.drive_min
        summary.total_served_demand += route.load
        summary.total_co2_kg += route.co2_kg
        summary.served_stops += route.served_count

    dropped = [
        stop.id for node, stop in enumerate(routable, start=1) if node not in visited
    ]
    summary.dropped_stops = len(dropped)

    return Plan(ok=True, routes=routes, summary=summary, dropped_stop_ids=dropped)

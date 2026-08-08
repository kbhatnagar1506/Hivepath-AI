"""Warm-start construction, including the depot bug that disabled it."""

from __future__ import annotations

from hivepath.domain import Depot, Stop, Vehicle
from hivepath.optimization.warm_start import (
    capacity_aware_routes,
    routes_from_plan,
    strip_depot,
    sweep_routes,
    validate_routes,
)


class TestStripDepot:
    def test_removes_leading_and_trailing_depot(self):
        assert strip_depot([0, 3, 1, 2, 0]) == [3, 1, 2]

    def test_removes_interior_depot_visits(self):
        assert strip_depot([0, 1, 0, 2, 0]) == [1, 2]

    def test_route_of_only_depot_becomes_empty(self):
        assert strip_depot([0, 0]) == []


class TestValidateRoutes:
    def test_strips_depot_so_ortools_accepts_the_seed(self):
        """Regression: OR-Tools rejects routes containing the depot with
        'Index 0 is used multiple times', silently discarding the warm start."""
        assert validate_routes([[0, 1, 2, 0], [0, 3, 0]], node_count=4) == [[1, 2], [3]]

    def test_drops_duplicate_nodes_across_routes(self):
        # A stop may appear at most once in the whole assignment.
        assert validate_routes([[0, 1, 2, 0], [0, 2, 3, 0]], node_count=4) == [[1, 2], [3]]

    def test_drops_out_of_range_nodes(self):
        assert validate_routes([[0, 1, 99, 0]], node_count=3) == [[1]]

    def test_no_depot_survives_validation(self):
        for route in validate_routes([[0, 1, 0, 2, 0]], node_count=3):
            assert 0 not in route


class TestSweepRoutes:
    def test_every_stop_assigned_exactly_once(self):
        depot = Depot(id="d", lat=42.36, lng=-71.06)
        stops = [
            Stop(id=f"s{i}", lat=42.36 + i * 0.01, lng=-71.06 + (i % 4) * 0.01)
            for i in range(12)
        ]
        routes = sweep_routes(depot, stops, vehicle_count=3)
        assigned = [node for route in routes for node in route]
        assert sorted(assigned) == list(range(1, 13))

    def test_one_route_per_vehicle(self):
        depot = Depot(id="d", lat=42.36, lng=-71.06)
        stops = [Stop(id="s1", lat=42.37, lng=-71.06)]
        assert len(sweep_routes(depot, stops, vehicle_count=4)) == 4

    def test_no_stops_yields_empty_routes(self):
        depot = Depot(id="d", lat=42.36, lng=-71.06)
        assert sweep_routes(depot, [], vehicle_count=2) == [[], []]


class TestCapacityAwareRoutes:
    def test_respects_vehicle_capacity(self):
        depot = Depot(id="d", lat=42.36, lng=-71.06)
        stops = [
            Stop(id=f"s{i}", lat=42.36 + i * 0.01, lng=-71.06, demand=60) for i in range(6)
        ]
        vehicles = [Vehicle(id="v1", capacity=120), Vehicle(id="v2", capacity=120)]

        routes = capacity_aware_routes(depot, stops, vehicles)
        demand_of = {i + 1: s.demand for i, s in enumerate(stops)}
        for route, vehicle in zip(routes, vehicles):
            assert sum(demand_of[n] for n in route) <= vehicle.capacity

    def test_excess_demand_is_left_unassigned_rather_than_overloading(self):
        depot = Depot(id="d", lat=42.36, lng=-71.06)
        stops = [Stop(id=f"s{i}", lat=42.36 + i * 0.01, lng=-71.06, demand=100) for i in range(5)]
        vehicles = [Vehicle(id="v1", capacity=100)]

        routes = capacity_aware_routes(depot, stops, vehicles)
        assert sum(len(r) for r in routes) == 1


class TestRoutesFromPlan:
    def test_extracts_and_strips_depot(self):
        plan = {
            "routes": [
                {"stops": [{"node": 0}, {"node": 2}, {"node": 1}]},
                {"stops": [{"node": 0}, {"node": 3}]},
            ]
        }
        assert routes_from_plan(plan) == [[2, 1], [3]]

    def test_empty_plan_yields_no_routes(self):
        assert routes_from_plan({}) == []

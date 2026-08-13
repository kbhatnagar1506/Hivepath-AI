"""Solver behaviour and constraint enforcement."""

from __future__ import annotations

import pytest

from hivepath.domain import Stop, TimeWindow, Vehicle
from hivepath.optimization.distance import DistanceMatrix
from hivepath.optimization.solver import SolverOptions, solve_vrp

FAST = dict(time_limit_sec=2, use_warm_start=True)


class TestSolverContract:
    def test_rejects_empty_fleet(self, depot, stops):
        plan = solve_vrp(depot, stops, [], SolverOptions(**FAST))
        assert not plan.ok and plan.error == "no_vehicles"

    def test_rejects_empty_stop_list(self, depot, vehicles):
        plan = solve_vrp(depot, [], vehicles, SolverOptions(**FAST))
        assert not plan.ok and plan.error == "no_stops"

    def test_solves_a_feasible_problem(self, depot, stops, vehicles):
        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.ok
        assert plan.summary.served_stops == len(stops)
        assert plan.summary.total_distance_km > 0

    def test_allow_drop_false_does_not_crash(self, depot, stops, vehicles):
        """Regression: search parameters were built inside ``if allow_drop:``,
        so this path raised UnboundLocalError on ``params``."""
        plan = solve_vrp(depot, stops, vehicles, SolverOptions(allow_drop=False, **FAST))
        assert plan.ok

    def test_rejects_mismatched_distance_matrix(self, depot, stops, vehicles):
        wrong = DistanceMatrix([[0.0, 1.0], [1.0, 0.0]], [[0, 1], [1, 0]], "haversine")
        with pytest.raises(ValueError, match="expected"):
            solve_vrp(depot, stops, vehicles, SolverOptions(**FAST), distance_matrix=wrong)


class TestConstraints:
    def test_capacity_is_never_exceeded(self, depot):
        stops = [
            Stop(id=f"s{i}", lat=42.36 + i * 0.01, lng=-71.06, demand=50) for i in range(6)
        ]
        vehicles = [Vehicle(id="v1", capacity=100), Vehicle(id="v2", capacity=100)]

        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.ok
        for route, vehicle in zip(plan.routes, vehicles, strict=True):
            assert route.load <= vehicle.capacity

    def test_excess_demand_is_dropped_not_crammed(self, depot):
        stops = [Stop(id=f"s{i}", lat=42.36 + i * 0.01, lng=-71.06, demand=90) for i in range(5)]
        vehicles = [Vehicle(id="v1", capacity=100)]

        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.ok
        assert plan.summary.dropped_stops > 0
        assert plan.summary.served_stops + plan.summary.dropped_stops == len(stops)

    def test_arrival_respects_time_windows(self, depot, vehicles):
        stops = [
            Stop(
                id=f"s{i}",
                lat=42.36 + i * 0.01,
                lng=-71.06,
                demand=10,
                time_window=TimeWindow(60, 240),
            )
            for i in range(4)
        ]
        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.ok
        for route in plan.routes:
            for visit in route.stops:
                if visit.node != 0:
                    assert 60 <= visit.arrival_min <= 240

    def test_blocked_stops_are_excluded_and_reported(self, depot, stops, vehicles):
        plan = solve_vrp(
            depot, stops, vehicles, SolverOptions(**FAST), blocked_stop_ids={"s0", "s1"}
        )
        assert plan.ok
        assert {"s0", "s1"} <= set(plan.dropped_stop_ids)
        visited = {v.stop_id for r in plan.routes for v in r.stops if v.stop_id}
        assert not visited & {"s0", "s1"}

    def test_all_stops_blocked_is_reported_not_crashed(self, depot, stops, vehicles):
        plan = solve_vrp(
            depot,
            stops,
            vehicles,
            SolverOptions(**FAST),
            blocked_stop_ids={s.id for s in stops},
        )
        assert not plan.ok and plan.error == "all_stops_blocked"


class TestAccessibilityInfluencesRouting:
    def test_inaccessible_stop_is_preferred_when_only_one_can_be_served(self, depot):
        """Capacity forces exactly one drop between two equidistant stops.

        Before the penalty fix both scored an identical drop cost, so
        accessibility had no effect on the outcome.
        """
        stops = [
            Stop(id="hard", lat=42.3701, lng=-71.0589, demand=60, access_score=10),
            Stop(id="easy", lat=42.3501, lng=-71.0589, demand=60, access_score=95),
        ]
        vehicles = [Vehicle(id="v1", capacity=60)]

        plan = solve_vrp(
            depot, stops, vehicles, SolverOptions(use_access_scores=True, **FAST)
        )
        served = {v.stop_id for r in plan.routes for v in r.stops if v.stop_id}
        assert served == {"hard"}

    def test_disabling_access_scores_removes_the_preference(self, depot):
        stops = [
            Stop(id="hard", lat=42.3701, lng=-71.0589, demand=60, access_score=10),
            Stop(id="easy", lat=42.3501, lng=-71.0589, demand=60, access_score=95),
        ]
        vehicles = [Vehicle(id="v1", capacity=60)]

        plan = solve_vrp(
            depot, stops, vehicles, SolverOptions(use_access_scores=False, **FAST)
        )
        assert plan.ok
        assert plan.summary.served_stops == 1


class TestTelemetry:
    def test_reports_haversine_without_credentials(self, depot, stops, vehicles):
        """Regression: telemetry claimed google_maps even after falling back."""
        plan = solve_vrp(
            depot, stops, vehicles, SolverOptions(use_google_maps=True, **FAST)
        )
        assert plan.telemetry["matrix_source"] == "haversine"

    def test_reports_warm_start_actually_applied(self, depot, stops, vehicles):
        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.telemetry["warm_started"] is True

    def test_records_solve_shape(self, depot, stops, vehicles):
        plan = solve_vrp(depot, stops, vehicles, SolverOptions(**FAST))
        assert plan.telemetry["nodes"] == len(stops)
        assert plan.telemetry["vehicles"] == len(vehicles)


class TestSolverOptions:
    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"time_limit_sec": 0}, "time_limit_sec"),
            ({"speed_kmph": 0}, "speed_kmph"),
            ({"num_workers": -1}, "num_workers"),
        ],
    )
    def test_invalid_options_rejected(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            SolverOptions(**kwargs)

    def test_from_settings_applies_overrides(self, settings):
        options = SolverOptions.from_settings(settings, time_limit_sec=3)
        assert options.time_limit_sec == 3
        assert options.drop_penalty_per_priority == settings.solver_drop_penalty_per_priority

    def test_from_settings_ignores_none_overrides(self, settings):
        options = SolverOptions.from_settings(settings, time_limit_sec=None)
        assert options.time_limit_sec == settings.solver_time_limit_sec

    def test_num_workers_reaches_the_only_field_that_accepts_it(self):
        """RoutingSearchParameters has no worker count; only CP-SAT does.

        Wired so the setting is not silently discarded. It does not parallelise
        the default guided local search.
        """
        from hivepath.optimization.solver import _build_search_parameters

        params = _build_search_parameters(SolverOptions(num_workers=6))
        assert params.sat_parameters.num_workers == 6

    def test_time_limit_reaches_the_solver(self):
        from hivepath.optimization.solver import _build_search_parameters

        params = _build_search_parameters(SolverOptions(time_limit_sec=5))
        assert params.time_limit.seconds == 5


class TestEmissions:
    def test_electric_vehicles_emit_less_than_diesel(self, depot):
        stops = [Stop(id=f"s{i}", lat=42.36 + i * 0.02, lng=-71.06, demand=10) for i in range(4)]

        diesel = solve_vrp(
            depot, stops, [Vehicle(id="v", capacity=500, fuel_type="diesel")],
            SolverOptions(**FAST),
        )
        electric = solve_vrp(
            depot, stops, [Vehicle(id="v", capacity=500, fuel_type="ev")],
            SolverOptions(**FAST),
        )
        assert diesel.ok and electric.ok
        assert electric.summary.total_co2_kg < diesel.summary.total_co2_kg

"""Domain model validation and conversions."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from hivepath.domain import Plan, Route, RouteStop, Stop, TimeWindow, Vehicle
from hivepath.domain.models import co2_factor


class TestStop:
    def test_rejects_priority_below_one(self):
        with pytest.raises(ValueError, match="priority"):
            Stop(id="s", lat=0, lng=0, priority=0)

    def test_rejects_negative_demand(self):
        with pytest.raises(ValueError, match="demand"):
            Stop(id="s", lat=0, lng=0, demand=-1)

    @pytest.mark.parametrize("score", [-1, 101])
    def test_rejects_out_of_range_access_score(self, score):
        with pytest.raises(ValueError, match="access_score"):
            Stop(id="s", lat=0, lng=0, access_score=score)

    def test_access_fraction_converts_to_model_scale(self):
        """The domain uses 0-100; the models were trained on 0-1."""
        assert Stop(id="s", lat=0, lng=0, access_score=75).access_fraction == 0.75

    def test_unassessed_stop_uses_training_median(self):
        assert Stop(id="s", lat=0, lng=0).access_fraction == 0.6

    def test_from_dict_round_trips_fields(self):
        stop = Stop.from_dict(
            {"id": "s1", "lat": 42.0, "lng": -71.0, "demand": 25, "priority": 3}
        )
        assert (stop.id, stop.demand, stop.priority) == ("s1", 25, 3)


class TestVehicle:
    def test_rejects_nonpositive_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            Vehicle(id="v", capacity=0)

    def test_electric_emits_less_per_km_than_diesel(self):
        assert Vehicle(id="v", fuel_type="ev").co2_kg_per_km < Vehicle(
            id="v", fuel_type="diesel"
        ).co2_kg_per_km

    def test_unknown_fuel_type_uses_default(self):
        assert co2_factor("hydrogen") == co2_factor(None)


class TestTimeWindow:
    def test_rejects_inverted_window(self):
        with pytest.raises(ValueError, match="after end"):
            TimeWindow(500, 100)

    def test_parses_bare_time_against_base_date(self):
        base = datetime(2026, 8, 8, 8, 0, tzinfo=timezone.utc)
        window = TimeWindow.from_iso(base, {"start": "09:00:00", "end": "11:00:00"})
        assert (window.start_min, window.end_min) == (60, 180)

    def test_unparseable_input_widens_to_full_day(self):
        """Collapsing to zero would silently make the stop unservable."""
        base = datetime(2026, 8, 8, 8, 0, tzinfo=timezone.utc)
        window = TimeWindow.from_iso(base, {"start": "not-a-time", "end": "also-not"})
        assert (window.start_min, window.end_min) == (0, 24 * 60)

    def test_missing_window_is_full_day(self):
        base = datetime(2026, 8, 8, 8, 0, tzinfo=timezone.utc)
        assert TimeWindow.from_iso(base, None) == TimeWindow.full_day()

    def test_reversed_bounds_are_normalised(self):
        base = datetime(2026, 8, 8, 8, 0, tzinfo=timezone.utc)
        window = TimeWindow.from_iso(base, {"start": "11:00:00", "end": "09:00:00"})
        assert window.start_min < window.end_min


class TestRouteAndPlan:
    def test_served_count_excludes_depot(self):
        route = Route(
            vehicle_id="v1",
            stops=[
                RouteStop(0, None, 0),
                RouteStop(1, "s1", 10),
                RouteStop(2, "s2", 20),
            ],
        )
        assert route.served_count == 2

    def test_failed_plan_serialises_its_error(self):
        payload = Plan.failed("no_solution", stops=5).to_dict()
        assert payload["ok"] is False
        assert payload["error"] == "no_solution"
        assert payload["telemetry"]["stops"] == 5

    def test_plan_to_dict_has_stable_shape(self):
        payload = Plan(ok=True).to_dict()
        assert set(payload) >= {
            "ok",
            "routes",
            "summary",
            "dropped_stop_ids",
            "telemetry",
        }

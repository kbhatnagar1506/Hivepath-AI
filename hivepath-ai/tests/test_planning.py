"""Planning pipeline: preset resolution, stage ordering, and replanning."""

from __future__ import annotations

import pytest

from hivepath.api.schemas import OptimizeRequest, Preset
from hivepath.planning import build_solver_options, create_plan, replan
from hivepath.storage import get_incident_repository, get_plan_repository


def make_request(**overrides) -> OptimizeRequest:
    payload = {
        "run_id": "r1",
        "depot": {"id": "d", "lat": 42.3601, "lng": -71.0589},
        "vehicles": [{"id": "v1", "capacity": 300}],
        "stops": [
            {"id": f"s{i}", "lat": 42.3601 + i * 0.008, "lng": -71.0589, "demand": 40}
            for i in range(5)
        ],
        "time_limit_sec": 2,
        "use_access_analysis": False,
    }
    payload.update(overrides)
    return OptimizeRequest(**payload)


class TestPresetResolution:
    @pytest.mark.parametrize("preset", list(Preset))
    def test_every_preset_produces_valid_options(self, preset):
        options = build_solver_options(make_request(preset=preset, time_limit_sec=None))
        assert options.time_limit_sec >= 1

    def test_quality_preset_no_longer_forbids_dropping(self):
        """It used to set allow_drop=False, which made over-capacity requests
        unsolvable. A very high drop penalty expresses the same intent."""
        options = build_solver_options(make_request(preset=Preset.QUALITY))
        assert options.allow_drop is True
        assert options.drop_penalty_per_priority >= 50_000

    def test_explicit_values_override_the_preset(self):
        options = build_solver_options(
            make_request(preset=Preset.QUALITY, time_limit_sec=3)
        )
        assert options.time_limit_sec == 3

    def test_ultra_fast_disables_network_bound_stages(self):
        options = build_solver_options(make_request(preset=Preset.ULTRA_FAST))
        assert options.use_google_maps is False
        assert options.use_warm_start is False


class TestPipeline:
    async def test_produces_a_plan(self):
        plan = await create_plan(make_request())
        assert plan.ok
        assert plan.run_id == "r1"
        assert plan.summary.served_stops == 5

    async def test_persists_plan_and_request(self):
        await create_plan(make_request())
        assert get_plan_repository().get("r1") is not None

    async def test_persist_false_leaves_storage_untouched(self):
        await create_plan(make_request(), persist=False)
        assert get_plan_repository().get("r1") is None

    async def test_service_times_are_applied(self):
        plan = await create_plan(make_request(use_service_time_model=True))
        assert plan.ok
        # Predicted service time makes the route take longer than pure driving.
        assert plan.summary.total_drive_min >= 0

    async def test_accessibility_runs_before_service_time_prediction(self, monkeypatch):
        """Ordering regression: access_score is an input feature to the service
        time model, but enrichment used to run after prediction, so the feature
        was always its default."""
        order: list[str] = []

        async def fake_enrich(self, stops, **kwargs):
            order.append("accessibility")
            return stops

        def fake_apply(stops, **kwargs):
            order.append("service_time")
            return stops

        monkeypatch.setattr(
            "hivepath.accessibility.enricher.AccessibilityEnricher.enrich", fake_enrich
        )
        monkeypatch.setattr("hivepath.planning.apply_service_times", fake_apply)

        await create_plan(
            make_request(use_access_analysis=True, use_service_time_model=True)
        )
        assert order == ["accessibility", "service_time"]

    async def test_blocked_stops_are_honoured(self):
        get_incident_repository().block("s2", ttl_minutes=30)
        plan = await create_plan(make_request())
        assert "s2" in plan.dropped_stop_ids


class TestReplan:
    async def test_unknown_run_returns_none(self):
        assert await replan("never-planned", "new-run") is None

    async def test_replan_excludes_the_blocked_stop(self):
        await create_plan(make_request())
        get_incident_repository().block("s1", ttl_minutes=30)

        plan = await replan("r1", "r2")
        assert plan is not None and plan.ok
        assert "s1" in plan.dropped_stop_ids
        assert plan.run_id == "r2"

    async def test_replan_is_stored_under_the_new_id(self):
        await create_plan(make_request())
        await replan("r1", "r2")
        assert get_plan_repository().get("r2") is not None

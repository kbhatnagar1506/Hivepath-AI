"""HTTP contract: status codes, validation, and payload shape."""

from __future__ import annotations

import pytest

API = "/api/v1"


class TestHealth:
    def test_health_reports_available_features(self, client):
        response = client.get(f"{API}/health")
        assert response.status_code == 200

        body = response.json()
        assert body["status"] == "healthy"
        # No credentials in the test environment, so integrations are off.
        assert body["features"]["google_maps_distances"] is False
        assert body["features"]["accessibility_analysis"] is False

    def test_health_also_served_unprefixed_for_orchestrators(self, client):
        assert client.get("/health").status_code == 200

    def test_index_banner(self, client):
        assert client.get("/").json()["docs"] == "/docs"

    def test_openapi_schema_builds(self, client):
        """Catches route/response-model mismatches that only surface on render."""
        assert client.get("/openapi.json").status_code == 200


class TestOptimize:
    def test_returns_a_plan(self, client, optimize_payload):
        response = client.post(f"{API}/optimize/routes", json=optimize_payload)
        assert response.status_code == 200

        body = response.json()
        assert body["ok"] is True
        assert body["run_id"] == "test-run"
        assert body["summary"]["served_stops"] == 8
        assert body["telemetry"]["matrix_source"] == "haversine"

    @pytest.mark.parametrize(
        "preset", ["ultra_fast", "fast", "balanced", "quality"]
    )
    def test_every_preset_succeeds(self, client, optimize_payload, preset):
        """Regression: preset=quality set allow_drop=False and returned 500."""
        payload = {**optimize_payload, "preset": preset, "run_id": f"run-{preset}"}
        response = client.post(f"{API}/optimize/routes", json=payload)
        assert response.status_code == 200, response.text
        assert response.json()["ok"] is True

    def test_google_maps_flag_is_accepted(self, client, optimize_payload):
        """Regression: this kwarg was passed to a solver that never declared it."""
        payload = {**optimize_payload, "use_google_maps": True}
        response = client.post(f"{API}/optimize/routes", json=payload)
        assert response.status_code == 200
        # No credentials, so it must report the fallback honestly.
        assert response.json()["telemetry"]["matrix_source"] == "haversine"

    def test_rejects_empty_stop_list(self, client, optimize_payload):
        payload = {**optimize_payload, "stops": []}
        assert client.post(f"{API}/optimize/routes", json=payload).status_code == 422

    def test_rejects_duplicate_stop_ids(self, client, optimize_payload):
        duplicated = [optimize_payload["stops"][0], optimize_payload["stops"][0]]
        payload = {**optimize_payload, "stops": duplicated}
        response = client.post(f"{API}/optimize/routes", json=payload)
        assert response.status_code == 422
        assert "duplicate" in response.text.lower()

    def test_rejects_out_of_range_coordinates(self, client, optimize_payload):
        payload = {**optimize_payload}
        payload["depot"] = {"id": "d", "lat": 999, "lng": -71.0}
        assert client.post(f"{API}/optimize/routes", json=payload).status_code == 422

    def test_rejects_unknown_fields(self, client, optimize_payload):
        payload = {**optimize_payload, "totally_made_up": True}
        assert client.post(f"{API}/optimize/routes", json=payload).status_code == 422

    def test_rejects_negative_demand(self, client, optimize_payload):
        payload = {**optimize_payload}
        payload["stops"] = [{**payload["stops"][0], "demand": -5}]
        assert client.post(f"{API}/optimize/routes", json=payload).status_code == 422

    def test_over_capacity_returns_a_partial_plan_not_an_error(
        self, client, optimize_payload
    ):
        payload = {**optimize_payload}
        payload["vehicles"] = [{"id": "v1", "capacity": 40}]
        response = client.post(f"{API}/optimize/routes", json=payload)

        assert response.status_code == 200
        body = response.json()
        assert body["ok"] is True
        assert body["summary"]["dropped_stops"] > 0


class TestPlans:
    def test_fetch_after_optimize(self, client, optimize_payload):
        client.post(f"{API}/optimize/routes", json=optimize_payload)
        response = client.get(f"{API}/plans/test-run")
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_unknown_plan_is_404(self, client):
        assert client.get(f"{API}/plans/nope").status_code == 404

    def test_metrics(self, client, optimize_payload):
        client.post(f"{API}/optimize/routes", json=optimize_payload)
        body = client.get(f"{API}/plans/test-run/metrics").json()

        assert body["ok"] is True
        assert body["served_stops"] == 8
        assert body["service_rate"] == 1.0
        assert body["total_co2_kg"] > 0

    def test_metrics_for_unknown_plan_is_404(self, client):
        assert client.get(f"{API}/plans/nope/metrics").status_code == 404


class TestIncidents:
    def test_report_blocks_a_stop(self, client):
        response = client.post(
            f"{API}/incidents",
            json={"id": "i1", "stop_id": "s3", "ttl_minutes": 30},
        )
        assert response.status_code == 201
        assert "s3" in response.json()["blocked_stops"]

    def test_listing_reflects_the_block(self, client):
        client.post(f"{API}/incidents", json={"id": "i1", "stop_id": "s3"})
        blocked = client.get(f"{API}/incidents").json()["blocked"]
        assert [b["stop_id"] for b in blocked] == ["s3"]

    def test_incident_triggers_replan(self, client, optimize_payload):
        client.post(f"{API}/optimize/routes", json=optimize_payload)

        response = client.post(
            f"{API}/incidents",
            json={
                "id": "i1",
                "stop_id": "s3",
                "replan_from_run_id": "test-run",
                "new_run_id": "test-run-2",
            },
        )
        assert response.status_code == 201

        body = response.json()
        assert body["new_run_id"] == "test-run-2"
        assert "s3" in body["plan"]["dropped_stop_ids"]

    def test_replan_requires_both_ids(self, client):
        response = client.post(
            f"{API}/incidents",
            json={"id": "i1", "stop_id": "s3", "replan_from_run_id": "test-run"},
        )
        assert response.status_code == 422

    def test_replan_of_unknown_run_is_404(self, client):
        response = client.post(
            f"{API}/incidents",
            json={
                "id": "i1",
                "stop_id": "s3",
                "replan_from_run_id": "never-existed",
                "new_run_id": "x",
            },
        )
        assert response.status_code == 404

    def test_clear_block(self, client):
        client.post(f"{API}/incidents", json={"id": "i1", "stop_id": "s3"})
        assert client.delete(f"{API}/incidents/s3").status_code == 200
        assert client.delete(f"{API}/incidents/s3").status_code == 404


class TestAccessibilityEndpoint:
    def test_returns_503_without_credentials(self, client):
        """Better than a neutral score that reads as a real assessment."""
        response = client.post(
            f"{API}/accessibility/analyze", json={"lat": 42.36, "lng": -71.05}
        )
        assert response.status_code == 503
        assert "OPENAI_API_KEY" in response.json()["detail"]

    def test_rejects_invalid_headings(self, client):
        response = client.post(
            f"{API}/accessibility/analyze",
            json={"lat": 42.36, "lng": -71.05, "headings": [0, 400]},
        )
        assert response.status_code == 422

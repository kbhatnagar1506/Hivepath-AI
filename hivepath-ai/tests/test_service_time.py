"""Service-time prediction, model selection, and the callable surface."""

from __future__ import annotations

from hivepath.domain import Stop
from hivepath.ml.service_time import (
    MAX_SERVICE_MINUTES,
    MIN_SERVICE_MINUTES,
    HeuristicServiceTimeModel,
    apply_service_times,
    get_service_time_model,
    predict_service_minutes,
)


class TestPublicSurface:
    def test_predict_service_minutes_is_importable_and_callable(self):
        """Regression: the router imported a name that did not exist.

        ``from services.service_time_model import predict_minutes`` raised
        ImportError into a bare ``except: pass``, so the model never ran and
        nothing reported it.
        """
        result = predict_service_minutes([Stop(id="s1", lat=42.0, lng=-71.0, demand=100)])
        assert set(result) == {"s1"}
        assert isinstance(result["s1"], float)

    def test_returns_mapping_keyed_by_stop_id(self):
        stops = [Stop(id=f"s{i}", lat=42.0, lng=-71.0, demand=i * 10) for i in range(3)]
        assert set(predict_service_minutes(stops)) == {"s0", "s1", "s2"}

    def test_empty_input_returns_empty_mapping(self):
        assert predict_service_minutes([]) == {}


class TestHeuristic:
    def test_larger_demand_takes_longer(self):
        model = HeuristicServiceTimeModel()
        small, large = model.predict(
            [
                Stop(id="a", lat=42.0, lng=-71.0, demand=10),
                Stop(id="b", lat=42.0, lng=-71.0, demand=500),
            ]
        )
        assert large > small

    def test_poor_accessibility_takes_longer(self):
        model = HeuristicServiceTimeModel()
        hard, easy = model.predict(
            [
                Stop(id="a", lat=42.0, lng=-71.0, demand=50, access_score=0),
                Stop(id="b", lat=42.0, lng=-71.0, demand=50, access_score=100),
            ]
        )
        assert hard > easy

    def test_accessibility_uses_the_0_100_domain_scale(self):
        """Regression: passing a 0-100 score into a 0-1 formula produced
        ``5 * (1 - 50) = -245``, clamping every stop to the floor."""
        model = HeuristicServiceTimeModel()
        predictions = model.predict(
            [Stop(id=f"s{s}", lat=42.0, lng=-71.0, demand=100, access_score=s) for s in (0, 50, 100)]
        )
        assert len(set(predictions)) == 3, "scores must remain distinguishable"
        assert all(p > MIN_SERVICE_MINUTES for p in predictions)

    def test_predictions_are_clamped_to_a_sane_range(self):
        model = HeuristicServiceTimeModel()
        predictions = model.predict(
            [
                Stop(id="tiny", lat=42.0, lng=-71.0, demand=0, access_score=100),
                Stop(id="huge", lat=42.0, lng=-71.0, demand=100_000, access_score=0),
            ]
        )
        assert all(MIN_SERVICE_MINUTES <= p <= MAX_SERVICE_MINUTES for p in predictions)


class TestModelSelection:
    def test_falls_back_to_heuristic_without_a_checkpoint(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
        from hivepath.config import get_settings

        get_settings.cache_clear()
        get_service_time_model.cache_clear()

        assert get_service_time_model().name == "heuristic"

    def test_model_is_resolved_once(self):
        assert get_service_time_model() is get_service_time_model()


class TestApplyServiceTimes:
    def test_populates_missing_service_minutes(self):
        stops = [Stop(id="s1", lat=42.0, lng=-71.0, demand=100)]
        apply_service_times(stops)
        assert stops[0].service_min is not None and stops[0].service_min >= 1

    def test_respects_explicit_service_minutes(self):
        stops = [Stop(id="s1", lat=42.0, lng=-71.0, demand=100, service_min=17)]
        apply_service_times(stops)
        assert stops[0].service_min == 17

    def test_overwrite_replaces_explicit_values(self):
        stops = [Stop(id="s1", lat=42.0, lng=-71.0, demand=100, service_min=17)]
        apply_service_times(stops, overwrite=True)
        assert stops[0].service_min != 17

"""Accessibility policy, response validation, and enrichment gating."""

from __future__ import annotations

import pytest

from hivepath.accessibility.enricher import AccessibilityEnricher
from hivepath.accessibility.policy import AccessThresholds, evaluate
from hivepath.domain import Stop
from hivepath.integrations.vision import NEUTRAL_RESULT, validate_analysis


def assessed(**overrides) -> dict:
    return {
        "access_score": 80,
        "service_time_sec": 240,
        "findings": [],
        "hazards": [],
        "notes": "",
        "assessed": True,
        **overrides,
    }


class TestPolicy:
    def test_accessible_location_is_not_blocked(self):
        decision = evaluate(assessed(access_score=90))
        assert not decision.blocked and not decision.warning

    def test_low_score_blocks(self):
        decision = evaluate(assessed(access_score=20))
        assert decision.blocked
        assert "below block threshold" in decision.reason

    def test_middling_score_warns_without_blocking(self):
        decision = evaluate(assessed(access_score=45))
        assert decision.warning and not decision.blocked

    def test_critical_hazard_blocks_regardless_of_score(self):
        decision = evaluate(
            assessed(access_score=95, hazards=[{"label": "blocked entrance", "severity": "critical"}])
        )
        assert decision.blocked
        assert "critical hazard" in decision.reason

    def test_minor_hazard_does_not_block(self):
        decision = evaluate(
            assessed(access_score=80, hazards=[{"label": "cones", "severity": "minor"}])
        )
        assert not decision.blocked

    def test_unassessed_location_is_never_blocked(self):
        """Absence of evidence is not evidence of inaccessibility."""
        decision = evaluate(dict(NEUTRAL_RESULT))
        assert not decision.blocked
        assert decision.reason == "not assessed"

    def test_thresholds_are_configurable(self):
        strict = AccessThresholds(block_below_score=90, warn_below_score=95)
        assert evaluate(assessed(access_score=80), strict).blocked


class TestVisionResponseValidation:
    def test_clamps_out_of_range_score(self):
        assert validate_analysis({"access_score": 5000})["access_score"] == 100
        assert validate_analysis({"access_score": -20})["access_score"] == 0

    def test_missing_fields_fall_back_to_neutral(self):
        result = validate_analysis({})
        assert result["access_score"] == 50
        assert result["service_time_sec"] == 240
        assert result["assessed"] is True

    def test_non_numeric_score_does_not_raise(self):
        assert validate_analysis({"access_score": "very bad"})["access_score"] == 50

    def test_critical_hazard_caps_the_score(self):
        """Enforced here rather than trusted to the model."""
        result = validate_analysis(
            {"access_score": 99, "hazards": [{"label": "x", "severity": "critical"}]}
        )
        assert result["access_score"] <= 35

    def test_unknown_severity_downgrades_to_minor(self):
        result = validate_analysis({"hazards": [{"label": "x", "severity": "apocalyptic"}]})
        assert result["hazards"][0]["severity"] == "minor"

    def test_malformed_entries_are_discarded(self):
        result = validate_analysis({"findings": ["not a dict", {"no_label": 1}]})
        assert result["findings"] == []

    def test_confidence_is_clamped(self):
        result = validate_analysis(
            {"findings": [{"label": "ramp", "present": True, "confidence": 12}]}
        )
        assert result["findings"][0]["confidence"] == 1.0


class TestEnricherGating:
    async def test_skipped_without_credentials(self, settings):
        """Stops must stay unassessed, not be stamped with a fake average."""
        stops = [Stop(id="s1", lat=42.0, lng=-71.0)]
        result = await AccessibilityEnricher(settings=settings).enrich(stops)
        assert result[0].access_score is None

    async def test_empty_input_is_a_noop(self, settings):
        assert await AccessibilityEnricher(settings=settings).enrich([]) == []

    async def test_existing_scores_are_preserved(self, settings):
        stops = [Stop(id="s1", lat=42.0, lng=-71.0, access_score=42)]
        result = await AccessibilityEnricher(settings=settings).enrich(stops)
        assert result[0].access_score == 42


class TestAnalyzerFallback:
    async def test_failure_yields_neutral_unassessed_result(self, monkeypatch, settings):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        from hivepath.config import get_settings

        get_settings.cache_clear()

        from hivepath.accessibility.analyzer import AccessibilityAnalyzer

        async def boom(*args, **kwargs):
            raise RuntimeError("street view down")

        monkeypatch.setattr("hivepath.accessibility.analyzer.fetch_panorama", boom)

        result = await AccessibilityAnalyzer().analyze(42.0, -71.0)
        assert result["assessed"] is False
        assert result["access_score"] == 50

    @pytest.mark.parametrize("lat,lng", [(42.0, -71.0), (0.0, 0.0)])
    async def test_no_imagery_is_not_an_error(self, monkeypatch, lat, lng):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "k")
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        from hivepath.config import get_settings

        get_settings.cache_clear()

        from hivepath.accessibility.analyzer import AccessibilityAnalyzer

        async def empty(*args, **kwargs):
            return []

        monkeypatch.setattr("hivepath.accessibility.analyzer.fetch_panorama", empty)

        result = await AccessibilityAnalyzer().analyze(lat, lng)
        assert result["assessed"] is False

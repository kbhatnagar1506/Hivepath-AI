"""Configuration loading and credential reconciliation."""

from __future__ import annotations

import pytest

from hivepath.config import Environment, Settings, get_settings


class TestDefaults:
    def test_starts_without_any_credentials(self, settings):
        assert settings.has_maps_credentials is False
        assert settings.has_vlm_credentials is False

    def test_sensible_solver_defaults(self, settings):
        assert settings.solver_time_limit_sec >= 1
        assert settings.solver_drop_penalty_per_priority > 0
        assert settings.solver_access_penalty_weight > 0


class TestCredentialReconciliation:
    def test_street_view_falls_back_to_the_maps_key(self, monkeypatch):
        """The old code read two different names for the same credential."""
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "shared-key")
        monkeypatch.delenv("GOOGLE_STREET_VIEW_API_KEY", raising=False)
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.street_view_key == "shared-key"
        assert settings.has_street_view_credentials

    def test_dedicated_street_view_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "maps-key")
        monkeypatch.setenv("GOOGLE_STREET_VIEW_API_KEY", "sv-key")
        get_settings.cache_clear()

        settings = get_settings()
        assert settings.street_view_key == "sv-key"
        assert settings.maps_key == "maps-key"


class TestSecretHandling:
    def test_secrets_are_not_exposed_in_repr(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "super-secret-value")
        monkeypatch.setenv("OPENAI_API_KEY", "another-secret")
        get_settings.cache_clear()

        rendered = repr(get_settings())
        assert "super-secret-value" not in rendered
        assert "another-secret" not in rendered

    def test_secret_is_still_readable_through_the_accessor(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "readable")
        get_settings.cache_clear()
        assert get_settings().maps_key == "readable"


class TestParsing:
    def test_cors_origins_split_from_csv(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.test, http://b.test ,")
        get_settings.cache_clear()
        assert get_settings().cors_origins == ["http://a.test", "http://b.test"]

    def test_log_level_is_normalised(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "debug")
        get_settings.cache_clear()
        assert get_settings().log_level == "DEBUG"

    def test_invalid_log_level_rejected(self, monkeypatch):
        monkeypatch.setenv("LOG_LEVEL", "chatty")
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="log_level"):
            get_settings()

    def test_environment_flag(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.environment is Environment.PRODUCTION
        assert settings.is_production


class TestCaching:
    def test_settings_are_cached(self):
        assert get_settings() is get_settings()

    def test_cache_clear_reloads(self, monkeypatch):
        first = get_settings()
        monkeypatch.setenv("SERVICE_NAME", "renamed")
        get_settings.cache_clear()
        assert get_settings() is not first
        assert get_settings().service_name == "renamed"


class TestPaths:
    def test_artefact_paths_are_absolute(self):
        settings = Settings()
        assert settings.models_dir.is_absolute()
        assert settings.artifacts_dir.is_absolute()

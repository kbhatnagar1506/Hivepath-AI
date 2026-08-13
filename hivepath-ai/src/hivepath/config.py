"""Application configuration.

Every environment-dependent value in the service is resolved here, from a single
``.env`` file, and nowhere else. Modules must not call ``os.getenv`` directly:
doing so at import time (as the previous layout did) freezes the value before
tests or a process manager can influence it, and makes the real configuration
surface impossible to discover.

Secrets are held as :class:`~pydantic.SecretStr` so they do not appear in logs,
tracebacks, or ``repr`` output.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _find_project_root() -> Path:
    """Walk up from this file until a directory containing pyproject.toml is found."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Runtime configuration, populated from environment variables and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- Service -----------------------------------------------------------
    service_name: str = "hivepath"
    environment: Environment = Environment.DEVELOPMENT
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    #: Origins permitted by CORS. Comma-separated in the environment.
    #: ``NoDecode`` suppresses pydantic-settings' automatic JSON decoding of
    #: complex types, which would otherwise reject a plain CSV string before
    #: the validator below ever runs.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:3000"]
    )

    # ---- Credentials -------------------------------------------------------
    google_maps_api_key: SecretStr = SecretStr("")
    #: Optional. Street View and Distance Matrix are both Google Maps Platform
    #: products and normally share one key; set this only to scope them apart.
    google_street_view_api_key: SecretStr | None = None
    openai_api_key: SecretStr = SecretStr("")
    vlm_model: str = "gpt-4o-mini"

    # ---- Infrastructure ----------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    base_backend_url: str = "http://localhost:8000"

    # ---- Solver defaults ---------------------------------------------------
    solver_time_limit_sec: int = 8
    solver_num_workers: int = 8
    solver_default_speed_kmph: float = 40.0
    solver_default_service_min: int = 5
    solver_drop_penalty_per_priority: int = 5000
    #: Fraction of the base drop penalty applied per point of missing
    #: accessibility (0-100 scale). See :mod:`hivepath.optimization.penalties`.
    solver_access_penalty_weight: float = 0.002

    # ---- Artefacts ---------------------------------------------------------
    # Named to avoid pydantic's protected "model_" namespace.
    models_dir: Path = PROJECT_ROOT / "models"
    artifacts_dir: Path = PROJECT_ROOT / "mlartifacts"
    data_dir: Path = PROJECT_ROOT / "data"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    # ---- Derived -----------------------------------------------------------
    @property
    def street_view_key(self) -> str:
        """Street View key, falling back to the shared Maps key when unset."""
        if self.google_street_view_api_key is not None:
            return self.google_street_view_api_key.get_secret_value()
        return self.google_maps_api_key.get_secret_value()

    @property
    def maps_key(self) -> str:
        return self.google_maps_api_key.get_secret_value()

    @property
    def openai_key(self) -> str:
        return self.openai_api_key.get_secret_value()

    @property
    def has_maps_credentials(self) -> bool:
        return bool(self.maps_key)

    @property
    def has_street_view_credentials(self) -> bool:
        return bool(self.street_view_key)

    @property
    def has_vlm_credentials(self) -> bool:
        return bool(self.openai_key)

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the ``.env`` file is read once. Tests call
    ``get_settings.cache_clear()`` after patching the environment.
    """
    return Settings()

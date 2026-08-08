"""Request and response models for the HTTP API.

These are the validation boundary: anything that reaches the domain layer has
already been checked here, which is why the solver can assume well-formed input.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from hivepath.domain import Depot, Stop, TimeWindow, Vehicle


def _reject_duplicates(ids: list[str], label: str) -> None:
    """Raise if any id repeats. Duplicate ids silently corrupt node indexing."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in ids:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {sorted(duplicates)}")


class Preset(str, Enum):
    """Named speed/quality trade-offs."""

    ULTRA_FAST = "ultra_fast"
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"


class TimeWindowSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str = Field(..., description="ISO timestamp, or HH:MM:SS for today")
    end: str = Field(..., description="ISO timestamp, or HH:MM:SS for today")


class DepotSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)

    def to_domain(self) -> Depot:
        return Depot(id=self.id, lat=self.lat, lng=self.lng)


class VehicleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    capacity: int = Field(1000, gt=0)
    fuel_type: str = Field("diesel")

    def to_domain(self) -> Vehicle:
        return Vehicle(id=self.id, capacity=self.capacity, fuel_type=self.fuel_type)


class StopSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    demand: int = Field(0, ge=0)
    priority: int = Field(1, ge=1, le=100)
    time_window: TimeWindowSchema | None = None
    service_min: int | None = Field(None, ge=0)
    access_score: float | None = Field(
        None, ge=0, le=100, description="0-100; higher is more accessible"
    )

    def to_domain(self, base: datetime) -> Stop:
        return Stop(
            id=self.id,
            lat=self.lat,
            lng=self.lng,
            demand=self.demand,
            priority=self.priority,
            time_window=(
                TimeWindow.from_iso(base, self.time_window.model_dump())
                if self.time_window
                else None
            ),
            service_min=self.service_min,
            access_score=self.access_score,
        )


class OptimizeRequest(BaseModel):
    """A routing problem to solve."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(..., min_length=1, description="Identifier for this plan")
    depot: DepotSchema
    vehicles: list[VehicleSchema] = Field(..., min_length=1)
    stops: list[StopSchema] = Field(..., min_length=1)

    preset: Preset | None = Field(
        None, description="Applied first; explicit fields below still win"
    )

    speed_kmph: float | None = Field(None, gt=0, le=200)
    time_limit_sec: int | None = Field(None, ge=1, le=300)
    num_workers: int | None = Field(None, ge=0, le=64)
    default_service_min: int | None = Field(None, ge=0)
    allow_drop: bool | None = None
    drop_penalty_per_priority: int | None = Field(None, ge=0)

    use_service_time_model: bool = True
    use_access_analysis: bool = True
    use_warm_start: bool = True
    use_google_maps: bool = False
    access_penalty_weight: float | None = Field(None, ge=0, le=1)
    debug_log: bool = False

    @field_validator("stops")
    @classmethod
    def _unique_stop_ids(cls, stops: list[StopSchema]) -> list[StopSchema]:
        _reject_duplicates([s.id for s in stops], "stop")
        return stops

    @field_validator("vehicles")
    @classmethod
    def _unique_vehicle_ids(cls, vehicles: list[VehicleSchema]) -> list[VehicleSchema]:
        _reject_duplicates([v.id for v in vehicles], "vehicle")
        return vehicles

    @model_validator(mode="after")
    def _warn_on_infeasible_capacity(self) -> OptimizeRequest:
        # Not an error: the solver drops stops it cannot serve. Rejecting here
        # would prevent the partial plan the caller probably still wants.
        return self

    def domain_stops(self, base: datetime | None = None) -> list[Stop]:
        base = base or datetime.now(timezone.utc)
        return [s.to_domain(base) for s in self.stops]

    def domain_vehicles(self) -> list[Vehicle]:
        return [v.to_domain() for v in self.vehicles]


class IncidentRequest(BaseModel):
    """Report a disruption, optionally triggering a replan."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    type: str = Field("blocked_stop")
    stop_id: str = Field(..., min_length=1)
    severity: float = Field(1.0, ge=0, le=1)
    ttl_minutes: int = Field(90, gt=0, le=1440)
    note: str | None = None
    replan_from_run_id: str | None = None
    new_run_id: str | None = None

    @model_validator(mode="after")
    def _replan_needs_both_ids(self) -> IncidentRequest:
        if bool(self.replan_from_run_id) != bool(self.new_run_id):
            raise ValueError(
                "replan_from_run_id and new_run_id must be provided together"
            )
        return self


class AccessibilityRequest(BaseModel):
    """Assess a single location's kerbside accessibility."""

    model_config = ConfigDict(extra="forbid")

    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    stop_id: str | None = None
    vehicle_desc: str = "26-ft box truck"
    headings: list[int] | None = Field(None, max_length=8)

    @field_validator("headings")
    @classmethod
    def _valid_headings(cls, headings: list[int] | None) -> list[int] | None:
        if headings is None:
            return None
        if any(not 0 <= h < 360 for h in headings):
            raise ValueError("headings must be within 0-359 degrees")
        return headings


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    features: dict[str, bool]


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None


class PlanResponse(BaseModel):
    """Loosely typed on purpose: the plan payload is built by the domain layer."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    run_id: str | None = None
    routes: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    dropped_stop_ids: list[str] = Field(default_factory=list)
    telemetry: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

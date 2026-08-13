"""Domain models.

The previous implementation threaded plain dictionaries through every layer,
which is how a caller came to pass ``use_google_maps=`` to a solver that had no
such parameter, and how CSV export drifted out of step with its own column
headers. These types make the shape of the data explicit and checkable.

Accessibility scores are on a **0-100** scale throughout the domain and the API.
The ML models were trained on a 0-1 scale; :meth:`Stop.access_fraction` is the
single place that conversion happens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Kilograms of CO2 per kilometre travelled, by vehicle fuel type.
CO2_KG_PER_KM: dict[str, float] = {
    "diesel": 0.82,
    "gas": 0.75,
    "ev": 0.12,
    "default": 0.80,
}

MINUTES_PER_DAY = 24 * 60


def co2_factor(fuel_type: str | None) -> float:
    """CO2 kg/km for a fuel type, falling back to the fleet default."""
    if not fuel_type:
        return CO2_KG_PER_KM["default"]
    return CO2_KG_PER_KM.get(fuel_type.lower(), CO2_KG_PER_KM["default"])


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A delivery window, stored as minutes from the plan's start instant."""

    start_min: int
    end_min: int

    def __post_init__(self) -> None:
        if self.start_min > self.end_min:
            raise ValueError(
                f"time window start ({self.start_min}) is after end ({self.end_min})"
            )

    @classmethod
    def full_day(cls) -> TimeWindow:
        return cls(0, MINUTES_PER_DAY)

    @classmethod
    def from_iso(cls, base: datetime, raw: dict[str, str] | None) -> TimeWindow:
        """Build from ``{"start": ISO, "end": ISO}``, relative to ``base``.

        A bare time such as ``"12:00:00"`` is interpreted as that time on
        ``base``'s date. Unparseable input widens to the full day rather than
        silently collapsing to zero, which would make the stop unservable.
        """
        if not raw:
            return cls.full_day()
        start, end = raw.get("start"), raw.get("end")
        if not start or not end:
            return cls.full_day()
        start_min = _iso_to_minutes(base, start)
        end_min = _iso_to_minutes(base, end)
        if start_min is None or end_min is None:
            return cls.full_day()
        if start_min > end_min:
            start_min, end_min = end_min, start_min
        return cls(start_min, end_min)


def _iso_to_minutes(base: datetime, raw: str) -> int | None:
    """Minutes from ``base`` to ``raw``; ``None`` when unparseable."""
    text = raw if "T" in raw else f"{base.date().isoformat()}T{raw}"
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=base.tzinfo or UTC)
    return max(0, int((moment - base).total_seconds() // 60))


@dataclass(frozen=True, slots=True)
class Depot:
    id: str
    lat: float
    lng: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Depot:
        return cls(id=str(data["id"]), lat=float(data["lat"]), lng=float(data["lng"]))


@dataclass(slots=True)
class Stop:
    """A delivery location.

    ``access_score`` is 0-100 where 100 is trivially accessible. ``None`` means
    accessibility has not been assessed, which is distinct from "assessed and
    found to be average" - only the former is skipped by the penalty model.
    """

    id: str
    lat: float
    lng: float
    demand: int = 0
    priority: int = 1
    time_window: TimeWindow | None = None
    service_min: int | None = None
    access_score: float | None = None

    def __post_init__(self) -> None:
        if self.priority < 1:
            raise ValueError(f"stop {self.id!r}: priority must be >= 1, got {self.priority}")
        if self.demand < 0:
            raise ValueError(f"stop {self.id!r}: demand must be >= 0, got {self.demand}")
        if self.access_score is not None and not 0 <= self.access_score <= 100:
            raise ValueError(
                f"stop {self.id!r}: access_score must be within 0-100, got {self.access_score}"
            )

    @property
    def access_fraction(self) -> float:
        """Accessibility on the 0-1 scale the ML models were trained against."""
        if self.access_score is None:
            return 0.6  # training-set median, used when unassessed
        return self.access_score / 100.0

    @classmethod
    def from_dict(cls, data: dict[str, Any], base: datetime | None = None) -> Stop:
        base = base or datetime.now(UTC)
        window = data.get("time_window")
        return cls(
            id=str(data["id"]),
            lat=float(data["lat"]),
            lng=float(data["lng"]),
            demand=int(data.get("demand", 0)),
            priority=int(data.get("priority", 1) or 1),
            time_window=TimeWindow.from_iso(base, window) if window else None,
            service_min=(
                int(data["service_min"]) if data.get("service_min") is not None else None
            ),
            access_score=(
                float(data["access_score"]) if data.get("access_score") is not None else None
            ),
        )


@dataclass(frozen=True, slots=True)
class Vehicle:
    id: str
    capacity: int = 1000
    fuel_type: str = "diesel"

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"vehicle {self.id!r}: capacity must be > 0, got {self.capacity}")

    @property
    def co2_kg_per_km(self) -> float:
        return co2_factor(self.fuel_type)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vehicle:
        return cls(
            id=str(data["id"]),
            capacity=int(data.get("capacity", 1000)),
            fuel_type=str(data.get("fuel_type") or "diesel"),
        )


@dataclass(frozen=True, slots=True)
class RouteStop:
    """One visit on a route. ``node`` is 0 for the depot, 1..N for stops."""

    node: int
    stop_id: str | None
    arrival_min: int


@dataclass(slots=True)
class Route:
    vehicle_id: str
    stops: list[RouteStop] = field(default_factory=list)
    distance_km: float = 0.0
    drive_min: int = 0
    load: int = 0
    co2_kg: float = 0.0

    @property
    def served_count(self) -> int:
        """Number of customer visits, excluding the depot."""
        return sum(1 for s in self.stops if s.node != 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vehicle_id": self.vehicle_id,
            "stops": [
                {"node": s.node, "stop_id": s.stop_id, "t_min": s.arrival_min}
                for s in self.stops
            ],
            "distance_km": round(self.distance_km, 2),
            "drive_min": self.drive_min,
            "load": self.load,
            "co2_kg": round(self.co2_kg, 2),
        }


@dataclass(slots=True)
class PlanSummary:
    total_distance_km: float = 0.0
    total_drive_min: int = 0
    total_served_demand: int = 0
    total_co2_kg: float = 0.0
    served_stops: int = 0
    dropped_stops: int = 0
    start_iso: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_distance_km": round(self.total_distance_km, 2),
            "total_drive_min": self.total_drive_min,
            "total_served_demand": self.total_served_demand,
            "total_co2_kg": round(self.total_co2_kg, 2),
            "served_stops": self.served_stops,
            "dropped_stops": self.dropped_stops,
            "start_iso": self.start_iso,
        }


@dataclass(slots=True)
class Plan:
    """The result of a solve. ``ok`` is False when no plan could be produced."""

    ok: bool
    routes: list[Route] = field(default_factory=list)
    summary: PlanSummary = field(default_factory=PlanSummary)
    dropped_stop_ids: list[str] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    run_id: str | None = None

    @classmethod
    def failed(cls, error: str, **telemetry: Any) -> Plan:
        return cls(ok=False, error=error, telemetry=telemetry)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "routes": [r.to_dict() for r in self.routes],
            "summary": self.summary.to_dict(),
            "dropped_stop_ids": list(self.dropped_stop_ids),
            "telemetry": dict(self.telemetry),
        }
        if self.error:
            payload["error"] = self.error
        if self.run_id:
            payload["run_id"] = self.run_id
        return payload

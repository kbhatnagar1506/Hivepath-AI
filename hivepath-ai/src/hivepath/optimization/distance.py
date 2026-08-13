"""Travel distance and duration matrices.

Two sources are supported: great-circle (haversine) arithmetic, which needs no
credentials, and the Google Distance Matrix API, which reflects real roads and
live traffic. The chosen source is recorded on the result so a plan can never
claim road distances it did not actually use - the previous implementation
reported ``google_maps`` even when it had silently fallen back.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from hivepath.config import Settings, get_settings
from hivepath.logging_config import get_logger

logger = get_logger(__name__)

MatrixSource = Literal["haversine", "google_maps"]

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lng) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


@dataclass(slots=True)
class DistanceMatrix:
    """Square distance (km) and duration (whole minutes) matrices."""

    distance_km: list[list[float]]
    duration_min: list[list[int]]
    source: MatrixSource

    def __post_init__(self) -> None:
        n = len(self.distance_km)
        if len(self.duration_min) != n:
            raise ValueError("distance and duration matrices differ in size")
        if any(len(row) != n for row in self.distance_km):
            raise ValueError("distance matrix is not square")
        if any(len(row) != n for row in self.duration_min):
            raise ValueError("duration matrix is not square")

    @property
    def size(self) -> int:
        return len(self.distance_km)


def _haversine_matrix(
    points: Sequence[tuple[float, float]], speed_kmph: float
) -> DistanceMatrix:
    n = len(points)
    distance = [[0.0] * n for _ in range(n)]
    duration = [[0] * n for _ in range(n)]
    safe_speed = max(1e-6, speed_kmph)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            km = haversine_km(points[i], points[j])
            distance[i][j] = km
            # At least one minute, so no arc is free and the solver cannot
            # collapse distinct stops into a zero-cost cluster.
            duration[i][j] = max(1, int((km / safe_speed) * 60))
    return DistanceMatrix(distance, duration, "haversine")


def _google_matrix(
    points: Sequence[tuple[float, float]], speed_kmph: float, settings: Settings
) -> DistanceMatrix:
    """Fetch a matrix from Google. Raises on any failure; the caller falls back."""
    from hivepath.integrations.google_maps import fetch_distance_matrix

    raw = fetch_distance_matrix(points, api_key=settings.maps_key)
    n = len(points)
    distance = [[0.0] * n for _ in range(n)]
    duration = [[0] * n for _ in range(n)]
    safe_speed = max(1e-6, speed_kmph)

    degraded = 0
    for i, row in enumerate(raw):
        for j, element in enumerate(row):
            if i == j:
                continue
            if element is None:
                # Google could not route this pair; fill the single cell rather
                # than discarding the whole matrix.
                degraded += 1
                km = haversine_km(points[i], points[j])
                distance[i][j] = km
                duration[i][j] = max(1, int((km / safe_speed) * 60))
            else:
                distance[i][j] = element.distance_m / 1000.0
                duration[i][j] = max(1, round(element.duration_s / 60))

    if degraded:
        logger.warning(
            "google distance matrix returned %d unroutable pairs; those cells use haversine",
            degraded,
            extra={"unroutable_pairs": degraded, "matrix_size": n},
        )
    return DistanceMatrix(distance, duration, "google_maps")


def build_distance_matrix(
    points: Sequence[tuple[float, float]],
    speed_kmph: float,
    *,
    use_google_maps: bool = False,
    settings: Settings | None = None,
) -> DistanceMatrix:
    """Build a travel matrix, preferring Google when asked and able.

    Falls back to haversine - and says so in :attr:`DistanceMatrix.source` - if
    Google is not requested, has no credentials, or errors.
    """
    settings = settings or get_settings()

    if not use_google_maps:
        return _haversine_matrix(points, speed_kmph)

    if not settings.has_maps_credentials:
        logger.info(
            "google maps requested but GOOGLE_MAPS_API_KEY is not set; using haversine distances"
        )
        return _haversine_matrix(points, speed_kmph)

    try:
        return _google_matrix(points, speed_kmph, settings)
    except Exception:
        logger.warning(
            "google distance matrix failed; falling back to haversine", exc_info=True
        )
        return _haversine_matrix(points, speed_kmph)


def matrix_from_locations(
    locations: Sequence[Any],
    speed_kmph: float,
    *,
    use_google_maps: bool = False,
    settings: Settings | None = None,
) -> DistanceMatrix:
    """Convenience wrapper for objects exposing ``.lat`` / ``.lng``."""
    points = [(float(loc.lat), float(loc.lng)) for loc in locations]
    return build_distance_matrix(
        points, speed_kmph, use_google_maps=use_google_maps, settings=settings
    )

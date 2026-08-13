"""Google Distance Matrix client.

The API caps a single request at 25 origins, 25 destinations, and 100 elements,
so an N x N matrix is fetched in tiles. The previous implementation issued one
unbounded request and relied on ``lru_cache`` over tuple arguments, which meant
any fleet above ten stops silently exceeded the element limit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import httpx

from hivepath.logging_config import get_logger

logger = get_logger(__name__)

DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

MAX_ORIGINS_PER_REQUEST = 25
MAX_DESTINATIONS_PER_REQUEST = 25
MAX_ELEMENTS_PER_REQUEST = 100

DEFAULT_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class MatrixElement:
    distance_m: int
    duration_s: int


class GoogleMapsError(RuntimeError):
    """Raised when the Distance Matrix API rejects a request outright."""


def _format_points(points: Sequence[tuple[float, float]]) -> str:
    return "|".join(f"{lat},{lng}" for lat, lng in points)


def _tile_sizes(n: int) -> tuple[int, int]:
    """Choose origin/destination tile sizes respecting all three API caps."""
    destinations = min(MAX_DESTINATIONS_PER_REQUEST, n) or 1
    origins = max(1, min(MAX_ORIGINS_PER_REQUEST, MAX_ELEMENTS_PER_REQUEST // destinations))
    return origins, destinations


def fetch_distance_matrix(
    points: Sequence[tuple[float, float]],
    *,
    api_key: str,
    client: httpx.Client | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    departure_time: str = "now",
) -> list[list[MatrixElement | None]]:
    """Fetch an N x N travel matrix.

    Returns a grid where ``None`` marks a pair Google could not route; callers
    decide how to fill those cells. Raises :class:`GoogleMapsError` only when a
    request fails as a whole.
    """
    if not api_key:
        raise GoogleMapsError("GOOGLE_MAPS_API_KEY is not configured")

    n = len(points)
    result: list[list[MatrixElement | None]] = [[None] * n for _ in range(n)]
    origin_tile, destination_tile = _tile_sizes(n)

    owns_client = client is None
    client = client or httpx.Client(timeout=timeout)
    requests_made = 0

    try:
        for o_start in range(0, n, origin_tile):
            origins = points[o_start : o_start + origin_tile]
            for d_start in range(0, n, destination_tile):
                destinations = points[d_start : d_start + destination_tile]

                response = client.get(
                    DISTANCE_MATRIX_URL,
                    params={
                        "origins": _format_points(origins),
                        "destinations": _format_points(destinations),
                        "mode": "driving",
                        "units": "metric",
                        "departure_time": departure_time,
                        "traffic_model": "best_guess",
                        "key": api_key,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                requests_made += 1

                status = payload.get("status")
                if status != "OK":
                    raise GoogleMapsError(
                        f"Distance Matrix returned {status}: "
                        f"{payload.get('error_message', 'no detail')}"
                    )

                for row_offset, row in enumerate(payload.get("rows", [])):
                    for col_offset, element in enumerate(row.get("elements", [])):
                        if element.get("status") != "OK":
                            continue
                        # duration_in_traffic is only present with a departure
                        # time and a traffic model; fall back to free-flow.
                        duration = element.get("duration_in_traffic") or element["duration"]
                        result[o_start + row_offset][d_start + col_offset] = MatrixElement(
                            distance_m=int(element["distance"]["value"]),
                            duration_s=int(duration["value"]),
                        )
    finally:
        if owns_client:
            client.close()

    logger.debug(
        "fetched %dx%d distance matrix in %d request(s)", n, n, requests_made
    )
    return result

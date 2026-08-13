"""Google Street View Static API client.

Consolidates what were three separate implementations (``streetview_client``,
``vlm_client``, and ``streetscout``) reading two different environment variable
names for the same credential.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from hivepath.config import Settings, get_settings
from hivepath.logging_config import get_logger

logger = get_logger(__name__)

STREET_VIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
STREET_VIEW_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"

#: North, east, south, west - enough to characterise a kerbside.
DEFAULT_HEADINGS: tuple[int, ...] = (0, 90, 180, 270)
DEFAULT_TIMEOUT_S = 20.0


class StreetViewError(RuntimeError):
    """Raised when Street View imagery cannot be retrieved."""


def build_image_url(
    lat: float,
    lng: float,
    *,
    api_key: str,
    heading: int = 0,
    fov: int = 90,
    pitch: int = -10,
    size: str = "640x640",
) -> str:
    return (
        f"{STREET_VIEW_URL}?size={size}&location={lat},{lng}"
        f"&heading={heading}&fov={fov}&pitch={pitch}&key={api_key}"
    )


async def fetch_image(
    lat: float,
    lng: float,
    *,
    heading: int = 0,
    client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> bytes:
    """Fetch a single Street View frame as JPEG bytes."""
    settings = settings or get_settings()
    if not settings.has_street_view_credentials:
        raise StreetViewError(
            "Street View needs GOOGLE_MAPS_API_KEY (or GOOGLE_STREET_VIEW_API_KEY)"
        )

    url = build_image_url(lat, lng, api_key=settings.street_view_key, heading=heading)
    response = await client.get(url)
    response.raise_for_status()

    # Street View bills for and returns a grey "no imagery" placeholder rather
    # than a 404, so check the payload actually looks like a photo.
    if not response.content or len(response.content) < 1024:
        raise StreetViewError(f"no imagery available at {lat},{lng} heading {heading}")
    return response.content


async def fetch_panorama(
    lat: float,
    lng: float,
    *,
    headings: Sequence[int] = DEFAULT_HEADINGS,
    settings: Settings | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> list[bytes]:
    """Fetch several headings concurrently, skipping any that fail.

    Returns an empty list when nothing could be retrieved; the caller decides
    whether that is fatal.
    """
    settings = settings or get_settings()

    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(
            *(
                fetch_image(lat, lng, heading=h, client=client, settings=settings)
                for h in headings
            ),
            return_exceptions=True,
        )

    images: list[bytes] = []
    for heading, result in zip(headings, results, strict=True):
        if isinstance(result, BaseException):
            logger.debug("street view heading %s unavailable: %s", heading, result)
        else:
            images.append(result)
    return images

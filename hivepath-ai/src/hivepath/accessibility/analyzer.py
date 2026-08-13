"""Assess a location's kerbside accessibility from Street View imagery."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from hivepath.config import Settings, get_settings
from hivepath.integrations.street_view import DEFAULT_HEADINGS, fetch_panorama
from hivepath.integrations.vision import NEUTRAL_RESULT, analyze_images
from hivepath.logging_config import get_logger

logger = get_logger(__name__)


def _cache_key(lat: float, lng: float, headings: Sequence[int]) -> str:
    # Six decimal places is ~0.1m, finer than Street View can resolve.
    return f"{lat:.6f},{lng:.6f}@{'-'.join(map(str, sorted(headings)))}"


class AccessibilityAnalyzer:
    """Analyses locations, memoising results for the process lifetime.

    Concurrency is bounded so a large fleet cannot open hundreds of simultaneous
    connections to Street View and the vision model.
    """

    def __init__(
        self,
        *,
        max_concurrent: int = 8,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    async def analyze(
        self,
        lat: float,
        lng: float,
        *,
        headings: Sequence[int] = DEFAULT_HEADINGS,
        vehicle_desc: str = "26-ft box truck",
        use_cache: bool = True,
    ) -> dict[str, Any]:
        """Return an accessibility analysis, or the neutral default.

        Never raises: any failure downgrades to :data:`NEUTRAL_RESULT` with
        ``assessed=False`` so callers can distinguish "average" from "unknown".
        """
        key = _cache_key(lat, lng, headings)
        if use_cache and key in self._cache:
            return dict(self._cache[key])

        async with self._semaphore:
            try:
                images = await fetch_panorama(
                    lat, lng, headings=headings, settings=self._settings
                )
                if not images:
                    logger.info("no street view imagery at %s,%s", lat, lng)
                    return dict(NEUTRAL_RESULT)

                result = await analyze_images(
                    images,
                    lat,
                    lng,
                    vehicle_desc=vehicle_desc,
                    settings=self._settings,
                )
            except Exception:
                logger.warning(
                    "accessibility analysis failed at %s,%s; using neutral default",
                    lat,
                    lng,
                    exc_info=True,
                )
                return dict(NEUTRAL_RESULT)

        if use_cache:
            self._cache[key] = dict(result)
        return result

    async def analyze_many(
        self,
        locations: Sequence[tuple[float, float]],
        *,
        headings: Sequence[int] = DEFAULT_HEADINGS,
        vehicle_desc: str = "26-ft box truck",
        use_cache: bool = True,
    ) -> list[dict[str, Any]]:
        """Analyse locations concurrently, preserving input order."""
        return list(
            await asyncio.gather(
                *(
                    self.analyze(
                        lat,
                        lng,
                        headings=headings,
                        vehicle_desc=vehicle_desc,
                        use_cache=use_cache,
                    )
                    for lat, lng in locations
                )
            )
        )

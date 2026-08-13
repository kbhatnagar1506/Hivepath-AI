"""Attach accessibility scores to stops before they reach the solver."""

from __future__ import annotations

from collections.abc import Sequence

from hivepath.accessibility.analyzer import AccessibilityAnalyzer
from hivepath.config import Settings, get_settings
from hivepath.domain import Stop
from hivepath.integrations.street_view import DEFAULT_HEADINGS
from hivepath.logging_config import get_logger

logger = get_logger(__name__)


class AccessibilityEnricher:
    """Populates ``Stop.access_score`` and ``Stop.service_min``."""

    def __init__(
        self,
        analyzer: AccessibilityAnalyzer | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._analyzer = analyzer or AccessibilityAnalyzer(settings=self._settings)

    async def enrich(
        self,
        stops: Sequence[Stop],
        *,
        headings: Sequence[int] = DEFAULT_HEADINGS,
        vehicle_desc: str = "26-ft box truck",
        overwrite: bool = False,
    ) -> list[Stop]:
        """Return stops with accessibility applied.

        Stops that already carry a score are left alone unless ``overwrite``.
        When no credentials are configured the stops are returned untouched -
        deliberately leaving ``access_score`` as ``None`` so the penalty model
        treats them as unassessed rather than as average.
        """
        stops = list(stops)
        if not stops:
            return stops

        if not (
            self._settings.has_street_view_credentials and self._settings.has_vlm_credentials
        ):
            logger.info(
                "accessibility analysis skipped: Street View and/or vision credentials absent"
            )
            return stops

        pending = [
            stop for stop in stops if overwrite or stop.access_score is None
        ]
        if not pending:
            return stops

        results = await self._analyzer.analyze_many(
            [(stop.lat, stop.lng) for stop in pending],
            headings=headings,
            vehicle_desc=vehicle_desc,
        )

        assessed = 0
        for stop, analysis in zip(pending, results, strict=True):
            if not analysis.get("assessed", False):
                continue
            assessed += 1
            stop.access_score = float(analysis["access_score"])
            if stop.service_min is None:
                stop.service_min = max(1, round(int(analysis["service_time_sec"]) / 60))

        logger.info("accessibility assessed for %d/%d stop(s)", assessed, len(pending))
        return stops


async def enrich_stops(
    stops: Sequence[Stop],
    *,
    settings: Settings | None = None,
    **kwargs: object,
) -> list[Stop]:
    """Module-level convenience wrapper around :class:`AccessibilityEnricher`."""
    return await AccessibilityEnricher(settings=settings).enrich(stops, **kwargs)  # type: ignore[arg-type]

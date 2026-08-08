"""Planning orchestration.

One place where accessibility analysis, service-time prediction, and the solver
are composed, so the HTTP routers stay thin and the same pipeline is reachable
from tests and scripts without going through HTTP.

Order matters: accessibility runs **before** service-time prediction, because
``access_score`` is an input feature to that model. The previous implementation
had them the other way round, so the feature was always its default.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from hivepath.accessibility import AccessibilityEnricher
from hivepath.api.schemas import OptimizeRequest, Preset
from hivepath.config import Settings, get_settings
from hivepath.domain import Plan
from hivepath.logging_config import get_logger
from hivepath.ml.service_time import apply_service_times
from hivepath.optimization.solver import SolverOptions, solve_vrp
from hivepath.optimization.warm_start import routes_from_plan
from hivepath.storage import (
    get_incident_repository,
    get_plan_repository,
    get_request_repository,
)

logger = get_logger(__name__)


#: Solver overrides per preset. Anything the caller sets explicitly wins over
#: these, so a preset is a starting point rather than a straitjacket.
PRESET_OPTIONS: dict[Preset, dict[str, Any]] = {
    Preset.ULTRA_FAST: {
        "time_limit_sec": 2,
        "drop_penalty_per_priority": 25_000,
        "use_google_maps": False,
        "use_warm_start": False,
    },
    Preset.FAST: {
        "time_limit_sec": 4,
        "drop_penalty_per_priority": 20_000,
    },
    Preset.BALANCED: {
        "time_limit_sec": 6,
        "drop_penalty_per_priority": 25_000,
    },
    Preset.QUALITY: {
        "time_limit_sec": 12,
        "drop_penalty_per_priority": 50_000,
        # Historically this preset set allow_drop=False, which made the request
        # unsolvable whenever demand exceeded fleet capacity. A very high drop
        # penalty expresses the same intent without the cliff.
        "allow_drop": True,
    },
}

#: Features disabled by the fastest preset, since each costs a network round
#: trip or a model load.
_ULTRA_FAST_DISABLES = ("use_service_time_model", "use_access_analysis")


def build_solver_options(
    request: OptimizeRequest, settings: Settings | None = None
) -> SolverOptions:
    """Resolve preset defaults and explicit overrides into solver options."""
    settings = settings or get_settings()

    overrides: dict[str, Any] = {}
    if request.preset is not None:
        overrides.update(PRESET_OPTIONS.get(request.preset, {}))

    explicit = {
        "speed_kmph": request.speed_kmph,
        "time_limit_sec": request.time_limit_sec,
        "num_workers": request.num_workers,
        "default_service_min": request.default_service_min,
        "allow_drop": request.allow_drop,
        "drop_penalty_per_priority": request.drop_penalty_per_priority,
        "access_penalty_weight": request.access_penalty_weight,
    }
    overrides.update({k: v for k, v in explicit.items() if v is not None})

    overrides["use_access_scores"] = request.use_access_analysis
    overrides["use_warm_start"] = overrides.get("use_warm_start", request.use_warm_start)
    overrides["use_google_maps"] = overrides.get("use_google_maps", request.use_google_maps)
    overrides["debug_log"] = request.debug_log

    return SolverOptions.from_settings(settings, **overrides)


def _feature_enabled(request: OptimizeRequest, name: str) -> bool:
    if request.preset is Preset.ULTRA_FAST and name in _ULTRA_FAST_DISABLES:
        return False
    return bool(getattr(request, name))


async def create_plan(
    request: OptimizeRequest,
    *,
    settings: Settings | None = None,
    persist: bool = True,
) -> Plan:
    """Run the full planning pipeline for a request."""
    settings = settings or get_settings()
    base = datetime.now(timezone.utc)

    depot = request.depot.to_domain()
    stops = request.domain_stops(base)
    vehicles = request.domain_vehicles()
    options = build_solver_options(request, settings)

    if _feature_enabled(request, "use_access_analysis"):
        try:
            stops = await AccessibilityEnricher(settings=settings).enrich(stops)
        except Exception:
            logger.warning(
                "accessibility enrichment failed; routing without access scores",
                exc_info=True,
            )

    if _feature_enabled(request, "use_service_time_model"):
        try:
            stops = apply_service_times(stops)
        except Exception:
            logger.warning(
                "service time prediction failed; using default service times",
                exc_info=True,
            )

    blocked = get_incident_repository().active_ids()

    # OR-Tools is CPU-bound and releases no GIL-friendly await points, so keep
    # it off the event loop or it will stall every other in-flight request.
    plan = await asyncio.to_thread(
        solve_vrp,
        depot,
        stops,
        vehicles,
        options,
        blocked_stop_ids=blocked,
        settings=settings,
    )
    plan.run_id = request.run_id

    if persist:
        get_plan_repository().save(request.run_id, plan.to_dict())
        get_request_repository().save(request.run_id, request.model_dump(mode="json"))

    return plan


async def replan(
    original_run_id: str,
    new_run_id: str,
    *,
    settings: Settings | None = None,
) -> Plan | None:
    """Re-solve a stored request against the current set of blocked stops.

    Returns ``None`` when the original request is no longer held in memory.
    The previous plan seeds the warm start, so the new plan stays close to the
    old one - drivers should not see routes reshuffle wholesale after one
    blocked dock.
    """
    settings = settings or get_settings()

    stored = get_request_repository().get(original_run_id)
    if stored is None:
        logger.warning("cannot replan %s: original request not found", original_run_id)
        return None

    request = OptimizeRequest.model_validate({**stored, "run_id": new_run_id})
    previous = get_plan_repository().get(original_run_id)

    base = datetime.now(timezone.utc)
    depot = request.depot.to_domain()
    stops = request.domain_stops(base)
    vehicles = request.domain_vehicles()
    options = build_solver_options(request, settings)

    warm = routes_from_plan(previous) if previous and previous.get("ok") else None

    plan = await asyncio.to_thread(
        solve_vrp,
        depot,
        stops,
        vehicles,
        options,
        blocked_stop_ids=get_incident_repository().active_ids(),
        warm_start_routes=warm,
        settings=settings,
    )
    plan.run_id = new_run_id

    get_plan_repository().save(new_run_id, plan.to_dict())
    get_request_repository().save(new_run_id, request.model_dump(mode="json"))
    return plan

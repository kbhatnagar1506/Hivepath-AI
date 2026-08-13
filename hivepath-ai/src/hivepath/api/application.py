"""FastAPI application factory.

A factory rather than a module-level ``app`` so tests can build an application
against a specific configuration instead of whatever the environment happened
to hold at import time.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from hivepath import __version__
from hivepath.api.routes import accessibility, health, incidents, optimization, plans
from hivepath.config import Settings, get_settings
from hivepath.logging_config import configure_logging, get_logger

logger = get_logger(__name__)

API_PREFIX = "/api/v1"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "%s v%s starting in %s mode",
        settings.service_name,
        __version__,
        settings.environment.value,
    )
    if not settings.has_maps_credentials:
        logger.info("GOOGLE_MAPS_API_KEY unset: distances will use haversine estimates")
    if not settings.has_vlm_credentials:
        logger.info("OPENAI_API_KEY unset: accessibility analysis is disabled")
    yield
    logger.info("%s shutting down", settings.service_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application."""
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="HivePath AI",
        description=(
            "Accessibility-aware logistics route optimization. "
            "Plans vehicle routes with time windows and capacity limits, and "
            "prefers to serve hard-to-reach stops rather than skip them."
        ),
        version=__version__,
        default_response_class=ORJSONResponse,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(optimization.router, prefix=API_PREFIX)
    app.include_router(plans.router, prefix=API_PREFIX)
    app.include_router(incidents.router, prefix=API_PREFIX)
    app.include_router(accessibility.router, prefix=API_PREFIX)

    # Kept unprefixed: container orchestrators probe these paths by convention.
    app.include_router(health.router)

    @app.get("/", tags=["health"], summary="Service banner")
    def index() -> dict[str, str]:
        return {
            "service": settings.service_name,
            "version": __version__,
            "docs": "/docs",
            "api": API_PREFIX,
        }

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        if response.status_code >= 500:
            logger.error(
                "%s %s -> %d", request.method, request.url.path, response.status_code
            )
        return response

    return app


#: Module-level instance for ``uvicorn hivepath.api.application:app``.
app = create_app()

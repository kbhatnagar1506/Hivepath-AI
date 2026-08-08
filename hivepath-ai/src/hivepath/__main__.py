"""Entry point: ``python -m hivepath`` or ``hivepath-api``."""

from __future__ import annotations

import uvicorn

from hivepath.config import get_settings
from hivepath.logging_config import configure_logging


def main() -> None:
    settings = get_settings()
    configure_logging(settings)
    uvicorn.run(
        "hivepath.api.application:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()

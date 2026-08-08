"""Route optimization endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from hivepath.api.schemas import ErrorResponse, OptimizeRequest, PlanResponse
from hivepath.logging_config import get_logger
from hivepath.planning import create_plan

logger = get_logger(__name__)

router = APIRouter(prefix="/optimize", tags=["optimization"])


@router.post(
    "/routes",
    response_model=PlanResponse,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Plan vehicle routes",
)
async def optimize_routes(request: OptimizeRequest) -> PlanResponse:
    """Solve a routing problem and store the resulting plan.

    Returns 200 with ``ok=false`` when the problem is well-formed but has no
    solution (for example, every stop blocked). A 5xx means the service itself
    failed.
    """
    try:
        plan = await create_plan(request)
    except ValueError as exc:
        # Domain-level validation the schema cannot express.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("optimization failed for run_id=%s", request.run_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"optimization failed: {exc}",
        ) from exc

    return PlanResponse(**plan.to_dict())

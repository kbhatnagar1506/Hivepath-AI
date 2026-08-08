"""Drop-penalty model.

In an OR-Tools disjunction the penalty is what the objective pays for leaving a
stop unserved, so a *higher* penalty makes a stop *more* likely to be visited.

Accessibility raises the penalty: a stop that is hard to reach is exactly the
one a purely distance-driven objective would skip first, and skipping it is the
outcome the platform exists to avoid.

The original formula was ``int(weight * (100 - access_score))`` with
``weight=0.002``. Its largest possible value was ``0.2``, which ``int()``
truncates to ``0`` - so accessibility never altered a single routing decision.
Scaling by the base penalty keeps the weight meaningful and makes it track
whatever ``drop_penalty_per_priority`` the caller chose.
"""

from __future__ import annotations

from typing import Final

#: Accessibility is expressed 0-100 across the domain and the API.
ACCESS_SCORE_MAX: Final = 100.0
ACCESS_SCORE_MIN: Final = 0.0


def clamp_access_score(score: float) -> float:
    """Constrain a score to 0-100."""
    return max(ACCESS_SCORE_MIN, min(ACCESS_SCORE_MAX, float(score)))


def access_penalty(
    base_penalty: int,
    access_score: float | None,
    weight: float,
) -> int:
    """Extra drop cost attributable to poor accessibility.

    Returns 0 when the stop has not been assessed, so an unassessed stop is
    treated the same as it was before accessibility analysis existed.

    Args:
        base_penalty: Priority-derived cost of dropping the stop.
        access_score: 0-100, higher is more accessible. ``None`` if unassessed.
        weight: Fraction of ``base_penalty`` charged per point of missing
            accessibility. At the 0.002 default, a fully inaccessible stop
            costs 20% more to drop than a fully accessible one.
    """
    if access_score is None:
        return 0
    if weight < 0:
        raise ValueError(f"access penalty weight must be >= 0, got {weight}")
    shortfall = ACCESS_SCORE_MAX - clamp_access_score(access_score)
    return int(weight * base_penalty * shortfall)


def drop_penalty(
    priority: int,
    *,
    penalty_per_priority: int,
    access_score: float | None = None,
    access_weight: float = 0.0,
) -> int:
    """Total cost of leaving a stop unserved.

    Always at least 1: a zero penalty would let the solver drop the stop for
    free, which is never the intent when the stop was submitted for routing.
    """
    if priority < 1:
        raise ValueError(f"priority must be >= 1, got {priority}")
    if penalty_per_priority < 0:
        raise ValueError(
            f"penalty_per_priority must be >= 0, got {penalty_per_priority}"
        )

    base = penalty_per_priority * priority
    total = base + access_penalty(base, access_score, access_weight)
    return max(1, total)

"""Policy translating an accessibility assessment into a routing decision.

Scores here are 0-100, matching the domain and the API. The previous module
mixed 0-1 probabilities with metre measurements in one threshold set, so a
score of ``50`` (mid-scale on one convention) read as a catastrophic failure on
the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AccessThresholds:
    """Cutoffs for deciding a location cannot be served as planned."""

    #: Below this, the location is treated as unusable for the vehicle.
    block_below_score: float = 35.0
    #: Below this, the location is usable but flagged for review.
    warn_below_score: float = 55.0
    #: A hazard at or above this severity blocks regardless of score.
    blocking_severities: frozenset[str] = frozenset({"critical"})


DEFAULT_THRESHOLDS = AccessThresholds()


@dataclass(frozen=True, slots=True)
class AccessDecision:
    blocked: bool
    warning: bool
    severity: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "warning": self.warning,
            "severity": self.severity,
            "reason": self.reason,
        }


def evaluate(
    analysis: dict[str, Any],
    thresholds: AccessThresholds = DEFAULT_THRESHOLDS,
) -> AccessDecision:
    """Decide whether a stop should be blocked, flagged, or left alone.

    An unassessed location is never blocked - absence of evidence is not
    evidence of inaccessibility.
    """
    if not analysis.get("assessed", False):
        return AccessDecision(
            blocked=False,
            warning=False,
            severity=0.0,
            reason="not assessed",
        )

    score = float(analysis.get("access_score", 50))
    hazards = analysis.get("hazards") or []
    blocking = [
        h for h in hazards if str(h.get("severity", "")).lower() in thresholds.blocking_severities
    ]

    if blocking:
        labels = ", ".join(sorted({str(h.get("label", "hazard")) for h in blocking}))
        return AccessDecision(
            blocked=True,
            warning=True,
            severity=0.95,
            reason=f"critical hazard: {labels}",
        )

    if score < thresholds.block_below_score:
        return AccessDecision(
            blocked=True,
            warning=True,
            severity=round(1.0 - score / 100.0, 2),
            reason=f"access score {score:.0f} below block threshold {thresholds.block_below_score:.0f}",
        )

    if score < thresholds.warn_below_score:
        return AccessDecision(
            blocked=False,
            warning=True,
            severity=round(1.0 - score / 100.0, 2),
            reason=f"access score {score:.0f} below warn threshold {thresholds.warn_below_score:.0f}",
        )

    return AccessDecision(
        blocked=False,
        warning=False,
        severity=round(max(0.0, 1.0 - score / 100.0), 2),
        reason="accessible",
    )

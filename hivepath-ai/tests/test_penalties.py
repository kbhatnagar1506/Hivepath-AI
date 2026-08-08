"""Drop-penalty model, including the regression that made it inert."""

from __future__ import annotations

import pytest

from hivepath.optimization.penalties import access_penalty, clamp_access_score, drop_penalty


class TestAccessPenalty:
    def test_unassessed_stop_gets_no_access_penalty(self):
        assert access_penalty(5000, None, 0.002) == 0

    def test_fully_accessible_stop_gets_no_access_penalty(self):
        assert access_penalty(5000, 100, 0.002) == 0

    def test_penalty_is_nonzero_at_default_weight(self):
        """Regression: the original formula truncated to 0 for every score.

        ``int(0.002 * (100 - score))`` peaks at 0.2, so accessibility never
        altered a routing decision.
        """
        assert access_penalty(5000, 0, 0.002) > 0

    def test_penalty_rises_as_accessibility_falls(self):
        weights = [access_penalty(5000, score, 0.002) for score in (100, 75, 50, 25, 0)]
        assert weights == sorted(weights)
        assert len(set(weights)) == len(weights), "each score must be distinguishable"

    def test_penalty_scales_with_base(self):
        """A preset that raises the base penalty raises the access term too."""
        assert access_penalty(50_000, 0, 0.002) == 10 * access_penalty(5000, 0, 0.002)

    @pytest.mark.parametrize(
        "score,expected", [(-10, 0.0), (0, 0.0), (50, 50.0), (150, 100.0)]
    )
    def test_scores_are_clamped(self, score, expected):
        assert clamp_access_score(score) == expected

    def test_negative_weight_rejected(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            access_penalty(5000, 50, -1)


class TestDropPenalty:
    def test_scales_with_priority(self):
        low = drop_penalty(1, penalty_per_priority=5000)
        high = drop_penalty(3, penalty_per_priority=5000)
        assert high == 3 * low

    def test_inaccessible_stop_costs_more_to_drop_than_accessible(self):
        """The whole point: hard-to-reach stops should be kept, not skipped."""
        hard = drop_penalty(1, penalty_per_priority=5000, access_score=10, access_weight=0.002)
        easy = drop_penalty(1, penalty_per_priority=5000, access_score=95, access_weight=0.002)
        assert hard > easy

    def test_never_free_to_drop(self):
        assert drop_penalty(1, penalty_per_priority=0) >= 1

    def test_zero_weight_disables_accessibility(self):
        with_score = drop_penalty(
            1, penalty_per_priority=5000, access_score=0, access_weight=0.0
        )
        without = drop_penalty(1, penalty_per_priority=5000)
        assert with_score == without

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValueError, match="priority must be >= 1"):
            drop_penalty(0, penalty_per_priority=5000)

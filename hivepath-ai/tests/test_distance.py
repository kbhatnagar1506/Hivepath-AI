"""Distance matrix construction and source reporting."""

from __future__ import annotations

import pytest

from hivepath.integrations.google_maps import GoogleMapsError, _tile_sizes
from hivepath.optimization.distance import (
    DistanceMatrix,
    build_distance_matrix,
    haversine_km,
)

BOSTON = (42.3601, -71.0589)
NEW_YORK = (40.7128, -74.0060)


class TestHaversine:
    def test_zero_for_identical_points(self):
        assert haversine_km(BOSTON, BOSTON) == pytest.approx(0.0, abs=1e-9)

    def test_matches_known_distance(self):
        # Boston to New York is ~306 km great-circle.
        assert haversine_km(BOSTON, NEW_YORK) == pytest.approx(306, abs=5)

    def test_symmetric(self):
        assert haversine_km(BOSTON, NEW_YORK) == pytest.approx(haversine_km(NEW_YORK, BOSTON))

    def test_antipodal_points_do_not_produce_nan(self):
        # Guards the domain error that an unclamped asin argument would raise.
        assert haversine_km((0.0, 0.0), (0.0, 180.0)) > 20_000


class TestBuildDistanceMatrix:
    def test_haversine_matrix_is_square_with_zero_diagonal(self):
        points = [BOSTON, NEW_YORK, (41.0, -72.0)]
        matrix = build_distance_matrix(points, 40.0)

        assert matrix.size == 3
        assert matrix.source == "haversine"
        for i in range(3):
            assert matrix.distance_km[i][i] == 0
            assert matrix.duration_min[i][i] == 0

    def test_every_offdiagonal_arc_costs_at_least_one_minute(self):
        # Two points ~10m apart would otherwise round to a free arc.
        points = [(42.3601, -71.0589), (42.3602, -71.0589)]
        matrix = build_distance_matrix(points, 40.0)
        assert matrix.duration_min[0][1] >= 1

    def test_lower_speed_yields_longer_durations(self):
        points = [BOSTON, NEW_YORK]
        slow = build_distance_matrix(points, 20.0)
        fast = build_distance_matrix(points, 80.0)
        assert slow.duration_min[0][1] > fast.duration_min[0][1]

    def test_falls_back_to_haversine_without_credentials(self, settings):
        """The source label must reflect what was actually used."""
        matrix = build_distance_matrix([BOSTON, NEW_YORK], 40.0, use_google_maps=True)
        assert matrix.source == "haversine"

    def test_falls_back_when_google_client_raises(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key")
        from hivepath.config import get_settings

        get_settings.cache_clear()

        def boom(*args, **kwargs):
            raise GoogleMapsError("quota exceeded")

        monkeypatch.setattr(
            "hivepath.integrations.google_maps.fetch_distance_matrix", boom
        )
        matrix = build_distance_matrix([BOSTON, NEW_YORK], 40.0, use_google_maps=True)
        assert matrix.source == "haversine"


class TestDistanceMatrixValidation:
    def test_rejects_non_square_distance(self):
        with pytest.raises(ValueError, match="not square"):
            DistanceMatrix([[0.0, 1.0]], [[0]], "haversine")

    def test_rejects_size_mismatch(self):
        with pytest.raises(ValueError, match="differ in size"):
            DistanceMatrix([[0.0, 1.0], [1.0, 0.0]], [[0, 1]], "haversine")


class TestGoogleTiling:
    @pytest.mark.parametrize("n", [1, 5, 10, 25, 40, 100])
    def test_tiles_respect_api_limits(self, n):
        """Google caps a request at 25 origins, 25 destinations, 100 elements."""
        origins, destinations = _tile_sizes(n)
        assert 1 <= origins <= 25
        assert 1 <= destinations <= 25
        assert origins * destinations <= 100

    def test_missing_key_is_an_error_not_a_silent_empty_matrix(self):
        from hivepath.integrations.google_maps import fetch_distance_matrix

        with pytest.raises(GoogleMapsError, match="not configured"):
            fetch_distance_matrix([BOSTON], api_key="")

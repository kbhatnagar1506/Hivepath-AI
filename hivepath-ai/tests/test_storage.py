"""Repository behaviour: bounds, expiry, and thread safety."""

from __future__ import annotations

import threading
import time

import pytest

from hivepath.storage import repositories
from hivepath.storage.repositories import (
    IncidentRepository,
    PlanRepository,
    RequestRepository,
)


class _FakeClock:
    """Stands in for the ``time`` module so expiry can be tested without sleeping."""

    def __init__(self, now: float) -> None:
        self._now = now

    def time(self) -> float:
        return self._now


class TestPlanRepository:
    def test_round_trip(self):
        repo = PlanRepository()
        repo.save("run-1", {"ok": True})
        assert repo.get("run-1") == {"ok": True}

    def test_missing_key_returns_none(self):
        assert PlanRepository().get("absent") is None

    def test_evicts_oldest_beyond_capacity(self):
        """A long-lived process must not grow without bound."""
        repo = PlanRepository(max_entries=3)
        for i in range(5):
            repo.save(f"run-{i}", {"i": i})

        assert len(repo) == 3
        assert repo.get("run-0") is None
        assert repo.get("run-4") == {"i": 4}

    def test_overwrite_does_not_count_against_capacity(self):
        repo = PlanRepository(max_entries=2)
        repo.save("a", {"v": 1})
        repo.save("b", {"v": 2})
        repo.save("a", {"v": 3})
        assert len(repo) == 2
        assert repo.get("a") == {"v": 3}

    def test_delete(self):
        repo = PlanRepository()
        repo.save("run-1", {})
        assert repo.delete("run-1") is True
        assert repo.delete("run-1") is False


class TestRequestRepository:
    def test_round_trip(self):
        repo = RequestRepository()
        repo.save("run-1", {"run_id": "run-1"})
        assert repo.get("run-1")["run_id"] == "run-1"


class TestIncidentRepository:
    def test_block_and_query(self):
        repo = IncidentRepository()
        repo.block("s1", ttl_minutes=10)
        assert repo.is_blocked("s1")
        assert repo.active_ids() == {"s1"}

    def test_expired_blocks_are_purged(self, monkeypatch):
        repo = IncidentRepository()
        repo.block("s1", ttl_minutes=1)
        assert repo.is_blocked("s1")

        # Advance the repository's clock past the TTL rather than sleeping.
        monkeypatch.setattr(repositories, "time", _FakeClock(time.time() + 3600))

        assert repo.active_ids() == set()
        assert not repo.is_blocked("s1")

    def test_unexpired_blocks_survive(self, monkeypatch):
        repo = IncidentRepository()
        repo.block("s1", ttl_minutes=60)

        monkeypatch.setattr(repositories, "time", _FakeClock(time.time() + 60))
        assert repo.is_blocked("s1")

    def test_unblock(self):
        repo = IncidentRepository()
        repo.block("s1")
        assert repo.unblock("s1") is True
        assert repo.unblock("s1") is False

    def test_rejects_nonpositive_ttl(self):
        with pytest.raises(ValueError, match="ttl_minutes"):
            IncidentRepository().block("s1", ttl_minutes=0)

    def test_concurrent_blocks_are_all_recorded(self):
        """FastAPI serves sync endpoints from a thread pool, so this is real."""
        repo = IncidentRepository()

        def worker(index: int) -> None:
            for j in range(20):
                repo.block(f"stop-{index}-{j}", ttl_minutes=60)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(repo.active_ids()) == 8 * 20


class TestConcurrentPlanWrites:
    def test_no_lost_updates_under_contention(self):
        repo = PlanRepository(max_entries=1000)

        def worker(index: int) -> None:
            for j in range(25):
                repo.save(f"run-{index}-{j}", {"i": index})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(repo) == 8 * 25

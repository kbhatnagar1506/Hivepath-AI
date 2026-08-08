"""In-memory repositories.

These replace three near-identical modules that each kept a bare module-level
dict. State is per-process and lost on restart, which is fine for planning runs
but is stated plainly here rather than implied to be Redis-backed.

Access is guarded by a lock: FastAPI serves synchronous endpoints from a thread
pool, so these are genuinely reached concurrently.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from hivepath.logging_config import get_logger

logger = get_logger(__name__)

#: Plans are bounded so a long-lived process cannot grow without limit.
DEFAULT_MAX_ENTRIES = 1000


class _BoundedStore:
    """Thread-safe dict with FIFO eviction once ``max_entries`` is exceeded."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key not in self._data and len(self._data) >= self._max_entries:
                oldest = next(iter(self._data))
                del self._data[oldest]
                logger.debug("evicted %s to stay within %d entries", oldest, self._max_entries)
            self._data[key] = value

    def get(self, key: str) -> Any | None:
        with self._lock:
            return self._data.get(key)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class PlanRepository:
    """Stores solved plans by run id."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._store = _BoundedStore(max_entries)

    def save(self, run_id: str, plan: dict[str, Any]) -> None:
        self._store.put(run_id, plan)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get(run_id)

    def delete(self, run_id: str) -> bool:
        return self._store.delete(run_id)

    def run_ids(self) -> list[str]:
        return self._store.keys()

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class RequestRepository:
    """Stores the original request for each run, so a replan can reuse it."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._store = _BoundedStore(max_entries)

    def save(self, run_id: str, payload: dict[str, Any]) -> None:
        self._store.put(run_id, payload)

    def get(self, run_id: str) -> dict[str, Any] | None:
        return self._store.get(run_id)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class IncidentRepository:
    """Tracks temporarily blocked stops with expiry."""

    def __init__(self) -> None:
        self._blocks: dict[str, float] = {}
        self._lock = threading.Lock()

    def block(self, stop_id: str, ttl_minutes: int = 90) -> float:
        """Block a stop; returns the expiry timestamp."""
        if ttl_minutes <= 0:
            raise ValueError(f"ttl_minutes must be > 0, got {ttl_minutes}")
        expires_at = time.time() + ttl_minutes * 60
        with self._lock:
            self._blocks[stop_id] = expires_at
        logger.info("blocked stop %s for %d minute(s)", stop_id, ttl_minutes)
        return expires_at

    def unblock(self, stop_id: str) -> bool:
        with self._lock:
            return self._blocks.pop(stop_id, None) is not None

    def is_blocked(self, stop_id: str) -> bool:
        return stop_id in self.active()

    def active(self) -> dict[str, float]:
        """Currently blocked stops, purging any that have expired."""
        now = time.time()
        with self._lock:
            expired = [k for k, v in self._blocks.items() if v <= now]
            for key in expired:
                del self._blocks[key]
            return dict(self._blocks)

    def active_ids(self) -> set[str]:
        return set(self.active())

    def clear(self) -> None:
        with self._lock:
            self._blocks.clear()


_plans = PlanRepository()
_requests = RequestRepository()
_incidents = IncidentRepository()


def get_plan_repository() -> PlanRepository:
    return _plans


def get_request_repository() -> RequestRepository:
    return _requests


def get_incident_repository() -> IncidentRepository:
    return _incidents


def reset_repositories() -> None:
    """Clear all shared state. Used by test fixtures."""
    _plans.clear()
    _requests.clear()
    _incidents.clear()

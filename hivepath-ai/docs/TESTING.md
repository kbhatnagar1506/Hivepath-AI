# Testing

190 tests, 85% line coverage, ~70 seconds. This document is the reasoning
behind the suite — how it's isolated, what each file actually verifies, and
what it deliberately does not test.

```bash
pytest                      # everything
pytest -v --durations=12    # see the slowest tests
pytest -k accessibility     # by keyword
pytest tests/test_solver.py -v
python -m coverage run --source=src/hivepath -m pytest && python -m coverage report --show-missing
```

---

## Philosophy

**Hermetic by default.** An autouse fixture in
[`conftest.py`](../tests/conftest.py) blanks `GOOGLE_MAPS_API_KEY`,
`GOOGLE_STREET_VIEW_API_KEY`, and `OPENAI_API_KEY` before every test, and
clears every `@lru_cache` (`get_settings`, `get_service_time_model`) and the
shared in-memory repositories. No test reaches the network, and no test can
see state another test left behind. This means the suite is safe to run
offline, in CI with no secrets configured, and in any order.

**Named after what they guard against.** A meaningful fraction of these tests
exist because something specific broke in this exact codebase. Rather than a
generic `test_solver_works`, they're written as
`test_penalty_is_nonzero_at_default_weight` or
`test_strips_depot_so_ortools_accepts_the_seed` — the name states the
regression, and the docstring explains the mechanism, so a future change that
reintroduces the bug fails with a comprehensible message instead of a bare
assertion diff.

**Real HTTP, not mocked routers.** [`test_api.py`](../tests/test_api.py) uses
FastAPI's `TestClient` against the actual `create_app()` factory — full
middleware stack, real Pydantic validation, real status codes. Nothing about
the HTTP layer is stubbed.

**Domain tests never import FastAPI.** `test_solver.py`, `test_penalties.py`,
`test_warm_start.py`, `test_domain.py` exercise the optimization core directly,
with no HTTP involved. This is what keeps them fast enough to run in under a
second each despite driving a real OR-Tools solve.

---

## Suite inventory

| File | Tests | What it verifies |
|---|---:|---|
| [`test_api.py`](../tests/test_api.py) | 25 | HTTP contract end to end: health, every preset via real POST, validation 422s, incident-triggered replan, 503 on missing accessibility credentials |
| [`test_solver.py`](../tests/test_solver.py) | 21 | Capacity/time-window enforcement, `allow_drop=False` no longer crashes, accessibility changes which stop is kept under scarcity, telemetry honesty |
| [`test_accessibility.py`](../tests/test_accessibility.py) | 19 | Block/warn policy thresholds, vision-response validation and clamping, enricher's credential gating, analyzer's fail-open behavior |
| [`test_domain.py`](../tests/test_domain.py) | 17 | Model invariants (`priority ≥ 1`, `capacity > 0`), time-window parsing and its fallback-to-full-day behavior, the 0–100 access-score conversion |
| [`test_warm_start.py`](../tests/test_warm_start.py) | 14 | Depot-stripping (the OR-Tools rejection bug), sweep heuristic coverage, capacity repair, duplicate/out-of-range node rejection |
| [`test_config.py`](../tests/test_config.py) | 13 | `.env` parsing, the `GOOGLE_STREET_VIEW_API_KEY` → `GOOGLE_MAPS_API_KEY` fallback, secret redaction in `repr()`, cache invalidation |
| [`test_distance.py`](../tests/test_distance.py) | 13 | Haversine correctness against a known distance, Google-matrix tiling stays within API limits at every fleet size, fallback-on-failure |
| [`test_planning.py`](../tests/test_planning.py) | 13 | Preset resolution and override precedence, **pipeline stage ordering**, replan behavior, persistence toggling |
| [`test_storage.py`](../tests/test_storage.py) | 13 | Bounded-store eviction, incident TTL expiry, **concurrent writes under real threads** |
| [`test_service_time.py`](../tests/test_service_time.py) | 12 | Heuristic model behavior, the 0–100→0–1 scale-conversion bug, lazy model selection |
| [`test_penalties.py`](../tests/test_penalties.py) | 12 | The drop-penalty formula, including the exact truncation-to-zero regression |

Totals above are undecorated test functions; the collected count (190) is
higher because several are `@pytest.mark.parametrize`d across multiple inputs —
for example every solver preset, or every invalid-settings case.

---

## Coverage, by module

```
Name                                       Stmts   Miss  Cover
------------------------------------------------------------------
src/hivepath/api/schemas.py                  135      2    99%
src/hivepath/config.py                        80      1    99%
src/hivepath/optimization/solver.py          145      3    98%
src/hivepath/domain/models.py                140      3    98%
src/hivepath/storage/repositories.py         102      4    96%
src/hivepath/optimization/penalties.py        21      1    95%
src/hivepath/planning.py                      75      4    95%
src/hivepath/optimization/warm_start.py       69      4    94%
src/hivepath/api/routes/plans.py              24      1    96%
src/hivepath/api/routes/health.py             13      1    92%
src/hivepath/api/application.py               45      8    82%
src/hivepath/accessibility/analyzer.py        39      8    79%
src/hivepath/logging_config.py                34      8    76%
src/hivepath/api/routes/optimization.py       17      5    71%
src/hivepath/api/routes/accessibility.py      22      6    73%
src/hivepath/optimization/distance.py         81     22    73%
src/hivepath/ml/service_time.py               86     28    67%
src/hivepath/integrations/vision.py           57     20    65%
src/hivepath/accessibility/enricher.py        35     15    57%
src/hivepath/integrations/google_maps.py      53     29    45%
src/hivepath/integrations/street_view.py      34     19    44%
src/hivepath/__main__.py                      10     10     0%
------------------------------------------------------------------
TOTAL                                       1397    203    85%
```

**The pattern is deliberate, not accidental.** Everything the solver's
correctness depends on — schemas, config, the solver itself, domain models,
storage, penalties, planning, warm start — sits at 94–99%. The lower numbers
cluster entirely in `integrations/` (the network clients) and the neural-model
branch of `ml/service_time.py`, which the hermetic fixture specifically
prevents from running: there is no credential in the test environment for them
to succeed with. `__main__.py` at 0% is the `uvicorn.run()` entry point, which
starting a real server would be needed to exercise and which none of the unit
tests should be doing.

If you add real credentials to a local `.env` and want to raise those numbers,
that's what `pytest -m integration` is reserved for — the marker exists in
`pyproject.toml` (`markers = ["integration: tests that require external
network credentials"]`) even though no test currently claims it. Nothing in
this suite talks to the internet today.

---

## Ten regressions worth reading

Every one of these is a test whose failure would mean a real defect came back.
Reading them is a faster way to understand what actually went wrong in the
codebase this replaced than reading a changelog.

1. **`test_penalty_is_nonzero_at_default_weight`**
   ([`test_penalties.py`](../tests/test_penalties.py)) — the accessibility
   penalty used to be `int(0.002 × (100 − score))`, whose max value before
   truncation is `0.2`. `int()` floored it to zero for every input, so
   accessibility never changed a single routing decision.

2. **`test_inaccessible_stop_is_preferred_when_only_one_can_be_served`**
   ([`test_solver.py`](../tests/test_solver.py)) — behavioral, not just
   arithmetic: with capacity forcing exactly one of two equidistant stops to
   be dropped, asserts the *harder-to-reach* one is the one kept.

3. **`test_strips_depot_so_ortools_accepts_the_seed`**
   ([`test_warm_start.py`](../tests/test_warm_start.py)) — OR-Tools rejects
   `ReadAssignmentFromRoutes` input containing the depot node with
   `Index 0 is used multiple times`. The old warm-start builder included it,
   so every warm start was silently rejected and every solve ran cold.

4. **`test_allow_drop_false_does_not_crash`**
   ([`test_solver.py`](../tests/test_solver.py)) — the OR-Tools search
   parameters used to be constructed inside `if allow_drop:`, so
   `allow_drop=False` (which `preset=quality` used to force) reached
   `SolveWithParameters(params)` with `params` unbound.

5. **`test_every_preset_succeeds`**
   ([`test_api.py`](../tests/test_api.py), parametrized) — the HTTP-level
   version of #4: every preset returns 200, run through the real API.

6. **`test_google_maps_flag_is_accepted`**
   ([`test_api.py`](../tests/test_api.py)) — `solve_vrp()` used to have no
   `use_google_maps` parameter at all; the router passed it unconditionally,
   so every request raised `TypeError` before reaching the solver.

7. **`test_reports_haversine_without_credentials`**
   ([`test_solver.py`](../tests/test_solver.py)) — a plan must never claim
   `matrix_source: "google_maps"` after silently falling back to haversine.

8. **`test_accessibility_uses_the_0_100_domain_scale`**
   ([`test_service_time.py`](../tests/test_service_time.py)) — the heuristic
   service-time formula was written for a 0–1 access score but fed a 0–100
   one, computing `5 × (1 - 50) = -245` and clamping every prediction to the
   same floor regardless of actual input.

9. **`test_accessibility_runs_before_service_time_prediction`**
   ([`test_planning.py`](../tests/test_planning.py)) — pipeline ordering, not
   a unit bug: `access_score` is an input feature to the service-time model,
   but enrichment used to run *after* prediction, so the feature was always
   unset when the model read it.

10. **`test_unassessed_location_is_never_blocked`**
    ([`test_accessibility.py`](../tests/test_accessibility.py)) — asserts
    that a stop with no accessibility data is never treated as *known to be*
    inaccessible. Absence of evidence is not evidence of inaccessibility, and
    conflating the two would have silently degraded coverage everywhere
    credentials happen to be unset.

---

## Concurrency tests

Two tests spin up real OS threads rather than relying on `asyncio` alone,
because the repositories are reached from FastAPI's sync-endpoint thread pool,
not just from async code:

- `TestIncidentRepository::test_concurrent_blocks_are_all_recorded`
- `TestConcurrentPlanWrites::test_no_lost_updates_under_contention`

Both hammer a shared repository from 8 threads × 20–25 writes and assert the
final count matches exactly — proving the `threading.Lock` around
`_BoundedStore` actually prevents lost updates, rather than merely existing.

---

## What isn't tested, and why that's a decision

- **Live Google Maps / Street View / OpenAI calls.** Deliberately excluded by
  the autouse credential-blanking fixture. Testing against a real third-party
  API in a unit suite means flaky CI, rate-limit exposure, and a dependency on
  secrets existing in every environment that runs `pytest`. The *contract*
  with those services — request shape, tiling limits, response validation,
  and fallback behavior — is tested with fakes; the actual network call is
  not.
- **The neural service-time model's forward pass.** `torch` is an optional
  extra; the suite runs (and must run) with it absent. What *is* tested is
  that its absence degrades cleanly to the heuristic, not that the neural
  model's numerical output is correct — that belongs in the training
  pipeline's own validation, not here.
- **Load and throughput.** This suite verifies correctness, not how many
  requests per second the service sustains. There is no load test in this
  repository yet.
- **The Next.js dashboard.** Out of scope for this suite entirely; it is a
  separate application under `integrated_dashboard/`.

---

## Adding a test

Two rules, taken from the existing suite:

1. If you're fixing a bug, name the test after the bug
   (`test_<what_would_have_failed>`), and put the mechanism — not just the
   symptom — in a one-line docstring. The goal is that someone reading a
   failure six months from now understands *why* the assertion exists without
   archaeology.
2. If your code makes a network call, it goes through
   `hivepath.integrations`, and your test mocks at that boundary (see
   `test_falls_back_when_google_client_raises` in
   [`test_distance.py`](../tests/test_distance.py) for the pattern) — never
   by making the call for real.

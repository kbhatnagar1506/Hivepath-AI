#!/usr/bin/env python3
"""Measure what the optimizer actually changes, on synthetic scenarios.

Two questions, each answered by running the same solver twice and comparing:

1. **Equity.** When the fleet cannot serve every stop, does accessibility-aware
   routing keep hard-to-reach stops in the plan? Compared against the identical
   solver with ``use_access_scores=False``.

2. **Efficiency.** How much shorter is an optimized plan than serving the same
   stops in the order they were submitted? Distance drives fuel and CO2.

These are synthetic scenarios with a fixed seed, not field measurements. They
quantify solver behaviour, not real-world delivery outcomes.

    python scripts/benchmark_impact.py [--trials 40]
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hivepath.domain import Depot, Stop, Vehicle  # noqa: E402
from hivepath.domain.models import co2_factor  # noqa: E402
from hivepath.optimization.distance import build_distance_matrix, haversine_km  # noqa: E402
from hivepath.optimization.solver import SolverOptions, solve_vrp  # noqa: E402

DEPOT = Depot(id="hub", lat=42.3601, lng=-71.0589)

#: Below this score a location is materially hard to service - stairs only,
#: no legal stopping nearby, blocked loading bay.
HARD_TO_REACH = 35.0

#: Share of stops drawn from the hard-to-reach tail.
HARD_FRACTION = 0.22

SEED = 20260808


@dataclass
class Scenario:
    stops: list[Stop]
    vehicles: list[Vehicle]

    @property
    def hard_stops(self) -> list[Stop]:
        return [s for s in self.stops if (s.access_score or 100) < HARD_TO_REACH]


def make_scenario(rng: random.Random, n_stops: int, capacity_ratio: float) -> Scenario:
    """Stops scattered around the depot, with a fleet deliberately too small.

    ``capacity_ratio`` below 1.0 forces the solver to choose which stops to
    drop - which is the only situation where the accessibility weighting can
    change an outcome.
    """
    stops = []
    for i in range(n_stops):
        # Mixture: most kerbsides are workable, a persistent minority are not.
        # A single beta skewed too accessible to generate hard stops reliably,
        # leaving too small a sample to draw any conclusion from.
        if rng.random() < HARD_FRACTION:
            access = rng.uniform(5, HARD_TO_REACH - 1)
        else:
            access = min(100.0, max(HARD_TO_REACH, rng.betavariate(5, 2) * 100))

        # ~5km box around the depot.
        stops.append(
            Stop(
                id=f"s{i}",
                lat=DEPOT.lat + rng.uniform(-0.045, 0.045),
                lng=DEPOT.lng + rng.uniform(-0.045, 0.045),
                demand=rng.randint(20, 60),
                priority=1,
                access_score=access,
            )
        )

    total_demand = sum(s.demand for s in stops)
    n_vehicles = 3
    per_vehicle = int(total_demand * capacity_ratio / n_vehicles)
    vehicles = [
        Vehicle(id=f"v{v}", capacity=max(60, per_vehicle), fuel_type="diesel")
        for v in range(n_vehicles)
    ]
    return Scenario(stops=stops, vehicles=vehicles)


def served_ids(plan) -> set[str]:
    return {v.stop_id for r in plan.routes for v in r.stops if v.stop_id}


def naive_distance_km(scenario: Scenario) -> float:
    """Distance if vehicles serve stops in submitted order until full.

    The realistic no-optimizer baseline: a dispatcher works down the manifest.
    """
    matrix = build_distance_matrix(
        [(DEPOT.lat, DEPOT.lng)] + [(s.lat, s.lng) for s in scenario.stops], 40.0
    )
    total = 0.0
    index = 0
    for vehicle in scenario.vehicles:
        load = 0
        position = 0  # depot
        while index < len(scenario.stops):
            stop = scenario.stops[index]
            if load + stop.demand > vehicle.capacity:
                break
            total += matrix.distance_km[position][index + 1]
            position = index + 1
            load += stop.demand
            index += 1
        total += matrix.distance_km[position][0]  # return to depot
    return total


def run(trials: int) -> None:
    rng = random.Random(SEED)
    options = dict(time_limit_sec=3, use_warm_start=True)

    hard_with: list[float] = []
    hard_without: list[float] = []
    overall_with: list[float] = []
    distance_delta_pct: list[float] = []
    co2_saved_kg: list[float] = []
    optimized_km: list[float] = []
    naive_km: list[float] = []

    print(f"Running {trials} scenarios (seed {SEED})...\n")

    for trial in range(trials):
        scenario = make_scenario(rng, n_stops=rng.randint(14, 22), capacity_ratio=0.72)
        hard = scenario.hard_stops
        if not hard:
            continue

        aware = solve_vrp(
            DEPOT, scenario.stops, scenario.vehicles,
            SolverOptions(use_access_scores=True, **options),
        )
        blind = solve_vrp(
            DEPOT, scenario.stops, scenario.vehicles,
            SolverOptions(use_access_scores=False, **options),
        )
        if not (aware.ok and blind.ok):
            continue

        aware_served, blind_served = served_ids(aware), served_ids(blind)
        hard_ids = {s.id for s in hard}

        hard_with.append(len(hard_ids & aware_served) / len(hard_ids))
        hard_without.append(len(hard_ids & blind_served) / len(hard_ids))
        overall_with.append(len(aware_served) / len(scenario.stops))

        # Efficiency: optimized vs serving the manifest in order.
        naive = naive_distance_km(scenario)
        opt = aware.summary.total_distance_km
        if naive > 0:
            optimized_km.append(opt)
            naive_km.append(naive)
            distance_delta_pct.append((naive - opt) / naive * 100)
            co2_saved_kg.append((naive - opt) * co2_factor("diesel"))

        if (trial + 1) % 10 == 0:
            print(f"  {trial + 1}/{trials} scenarios complete")

    def pct(values: list[float]) -> str:
        return f"{statistics.mean(values) * 100:.1f}%"

    print("\n" + "=" * 68)
    print("EQUITY - service rate for hard-to-reach stops (access score < 35)")
    print("=" * 68)
    print(f"  accessibility weighting ON   {pct(hard_with)}")
    print(f"  accessibility weighting OFF  {pct(hard_without)}")
    delta = (statistics.mean(hard_with) - statistics.mean(hard_without)) * 100
    print(f"  difference                   {delta:+.1f} percentage points")
    print(f"  overall service rate (ON)    {pct(overall_with)}")

    print("\n" + "=" * 68)
    print("EFFICIENCY - optimized plan vs serving the manifest in order")
    print("=" * 68)
    print(f"  mean optimized distance      {statistics.mean(optimized_km):.1f} km")
    print(f"  mean unoptimized distance    {statistics.mean(naive_km):.1f} km")
    print(f"  mean reduction               {statistics.mean(distance_delta_pct):.1f}%")
    print(f"  mean CO2 avoided per run     {statistics.mean(co2_saved_kg):.2f} kg (diesel)")

    print("\n" + "=" * 68)
    print(f"  scenarios measured           {len(hard_with)}")
    print("  NOTE: synthetic scenarios, fixed seed. Measures solver behaviour,")
    print("        not field-observed delivery outcomes.")
    print("=" * 68)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    run(parser.parse_args().trials)

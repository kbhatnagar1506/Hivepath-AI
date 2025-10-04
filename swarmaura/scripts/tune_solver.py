#!/usr/bin/env python3
import optuna, json, time, statistics as st, sys, pathlib
from dataclasses import asdict

# Add backend to path
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
from services.ortools_solver import solve_vrp

# Import evaluation functions
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from evaluate_solver import random_instance, extract_metrics

def score(plan, inst):
    m = extract_metrics(plan, inst)
    # Composite objective (lower is better)
    return (
        m["total_distance_km"]
        + 5.0 * m.get("dropped_count", 0)    # heavy penalty on drops
        + 0.5 * (1.0 - m["on_time_rate"]) * 100.0
        + 0.05 * m["total_co2_kg"]
    )

def objective(trial):
    # Search space
    time_limit = trial.suggest_int("time_limit_sec", 4, 12)
    drop_pen   = trial.suggest_int("drop_penalty_per_priority", 8000, 40000, step=2000)
    svc_min    = trial.suggest_int("default_service_min", 3, 10)
    allow_drop = trial.suggest_categorical("allow_drop", [True, False])

    # Evaluate on K random instances
    vals=[]
    for k in range(6):
        inst = random_instance(n_stops=12, n_vehicles=3, seed=trial.number*1000+k,
                               speed_kmph=35.0, veh_capacity=1200, demand_lo=60, demand_hi=140)
        plan = solve_vrp(
            asdict(inst.depot),
            [asdict(s) for s in inst.stops],
            [asdict(v) for v in inst.vehicles],
            inst.speed_kmph,
            time_limit_sec=time_limit,
            drop_penalty_per_priority=drop_pen,
            default_service_min=svc_min,
            allow_drop=allow_drop,
            num_workers=8
        )
        if not plan.get("ok"):
            vals.append(1e9); continue
        vals.append(score(plan, inst))
    return st.mean(vals)

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    print("Best:", study.best_trial.params)
    with open("models/solver_presets.json","w") as f:
        json.dump(study.best_trial.params, f, indent=2)
    print("Saved models/solver_presets.json")

if __name__ == "__main__":
    main()

# Legacy code

Everything in this directory predates the `src/hivepath/` package and is **not
maintained or imported by the running service**. It is kept because it contains
data-collection and analysis work that has not been ported yet, not because it
is expected to run.

## `legacy/backend/`

The original FastAPI application. Fully superseded:

| Old | New |
|---|---|
| `backend/app.py` | `hivepath/api/application.py` (app factory) |
| `backend/routers/optimize.py` | `hivepath/api/routes/optimization.py` |
| `backend/routers/plan.py`, `metrics.py` | `hivepath/api/routes/plans.py` |
| `backend/routers/incidents.py` | `hivepath/api/routes/incidents.py` |
| `backend/routers/agents.py`, `streetscout.py` | `hivepath/api/routes/accessibility.py` |
| `backend/services/ortools_solver.py` | `hivepath/optimization/solver.py` |
| `backend/services/warmstart*.py` | `hivepath/optimization/warm_start.py` |
| `backend/services/google_maps_client.py` | `hivepath/integrations/google_maps.py` |
| `backend/services/streetview_client.py` | `hivepath/integrations/street_view.py` |
| `backend/services/vlm_client.py`, `streetscout.py` | `hivepath/integrations/vision.py` |
| `backend/services/access_enricher.py`, `access_policy.py` | `hivepath/accessibility/` |
| `backend/services/service_time_model.py` | `hivepath/ml/service_time.py` |
| `backend/services/plan_store.py`, `request_store.py`, `incident_store.py` | `hivepath/storage/repositories.py` |
| `backend/services/config.py` | `hivepath/config.py` |

Not ported, because nothing imported them:

- `services/risk_shaper.py` - edge-level risk model, never wired into the solver
- `services/warmstart.py` - torch clusterer, superseded by the sweep heuristic
  in `hivepath/optimization/warm_start.py`
- `services/multi_location_solver.py` and `routers/multi_location.py` - the
  router was never registered on the app, so these endpoints never existed
- `routers/integration.py` - likewise never registered
- `services/service_time_model.py`'s GNN branch specifically: it tried to load
  a `SAGEConv`-based checkpoint, but the training script that produced that
  checkpoint (`ml/train_service_time_gnn.py`, also not ported) never actually
  built a graph model despite its name - its own code comment says "here we
  keep simple MLP for speed." The new `hivepath/ml/service_time.py` is an
  honest version of that: a plain feedforward MLP, documented as such, with no
  graph layers and no `torch_geometric` dependency.

The `models/*.joblib` and `mlartifacts/*.pt` checkpoints these produced have
been removed from the repository - none of them could be loaded by the ported
code as it stands (wrong filename, or an architecture the new loader doesn't
build). See the root [`README.md`](../../README.md#machine-learning) for what
actually runs today.

## `legacy/scripts/`

Twenty-one standalone analysis, demo, and data-extraction scripts that were
sitting in the project root. They import the old `backend.services` modules
through `sys.path` manipulation and **will not run** against the new layout
without being ported.

Two cautions if you revisit them:

1. A live Google Maps API key was hardcoded in `enable_google_vision.py`,
   `enhanced_image_stats.py`, and `weather_traffic_integration.py`. Those
   literals now read `os.getenv("GOOGLE_MAPS_API_KEY", "")`. The key has since
   been purged from this repository's git history entirely (via
   `git filter-repo`, verified against every blob in a fresh clone); it should
   still be treated as compromised if you're auditing a mirror or fork made
   before that happened.
2. Several scripts report fabricated figures. `presentation_demo.py` generates
   predictions with `random.uniform()` and prints accuracy percentages as string
   literals. Do not cite numbers from these as measurements.

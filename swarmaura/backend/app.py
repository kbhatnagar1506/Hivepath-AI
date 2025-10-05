from fastapi import FastAPI
from fastapi.responses import ORJSONResponse

from routers import health, optimize, plan, incidents, agents, metrics, ops, streetscout

app = FastAPI(
    title="RouteLoom Optimizer API",
    version="1.0.0",
    default_response_class=ORJSONResponse,
)

app.include_router(health.router)
app.include_router(optimize.router, prefix="/api/v1/optimize", tags=["optimize"])
app.include_router(plan.router,     prefix="/api/v1",           tags=["plan"])
app.include_router(incidents.router, prefix="/api/v1/incidents", tags=["incidents"])
app.include_router(agents.router,    prefix="/api/v1/agents",    tags=["agents"])
app.include_router(metrics.router,   prefix="/api/v1",           tags=["metrics"])
app.include_router(ops.router,       prefix="/api/v1/ops",       tags=["ops"])
app.include_router(streetscout.router, prefix="/api/v1",         tags=["streetscout"])

@app.get("/")
def index():
    return {"ok": True, "service": "routeloom"}


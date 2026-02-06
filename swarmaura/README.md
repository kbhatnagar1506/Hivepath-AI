# RouteLoom (Product 1) — Quickstart

## Run with Docker
```bash
cd infra
docker compose up --build

Health
curl :8000/health

Create baseline plan
curl -X POST :8000/api/v1/optimize/routes -H 'content-type: application/json' -d @../demo_payload.json

Fetch plan
curl :8000/api/v1/plan/demo-001

Block a stop & auto-replan
curl -X POST :8000/api/v1/incidents/ingest -H 'content-type: application/json' -d '{
  "id":"I-90-closure",
  "type":"closure",
  "target":{"stop_id":"C3"},
  "severity":0.9,
  "ttl_minutes":90,
  "replan_from_run_id":"demo-001",
  "new_run_id":"demo-002"
}'


Then

curl :8000/api/v1/plan/demo-002

(Optional) StreetView analysis → decide block → auto-replan

Set keys in .env.example, then:

curl -X POST :8000/api/v1/agents/streetview-analyze -H 'content-type: application/json' -d '{
  "run_id":"demo-001",
  "stop_id":"C3",
  "lat":42.39,
  "lng":-71.02,
  "vehicle_desc":"26-ft box truck",
  "autoincident": true
}'


---

# 7) Run it now (commands)

```bash
# from repo root
cd infra
docker compose up --build

# new terminal:
curl :8000/health

# make a plan
curl -X POST :8000/api/v1/optimize/routes -H 'content-type: application/json' -d @../demo_payload.json

# fetch it
curl :8000/api/v1/plan/demo-001

# simulate incident + auto-replan
curl -X POST :8000/api/v1/incidents/ingest -H 'content-type: application/json' -d '{
  "id":"block-c3","type":"blocked_dock","target":{"stop_id":"C3"},
  "severity":0.8,"ttl_minutes":60,"replan_from_run_id":"demo-001","new_run_id":"demo-002"
}'

curl :8000/api/v1/plan/demo-002

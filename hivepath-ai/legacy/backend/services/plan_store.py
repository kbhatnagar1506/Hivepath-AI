import json
from typing import Any, Dict

# In-memory store for testing (replace with Redis in production)
_memory_store = {}

def save_plan(run_id: str, plan: Dict[str, Any]) -> None: 
    _memory_store[f"plan:{run_id}"] = json.dumps(plan)

def get_plan(run_id: str) -> Dict[str, Any] | None:
    s = _memory_store.get(f"plan:{run_id}")
    return json.loads(s) if s else None

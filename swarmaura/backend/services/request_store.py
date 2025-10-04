import json
from typing import Any, Dict

# In-memory store for testing (replace with Redis in production)
_memory_store = {}

def save_request(run_id: str, body: Dict[str, Any]) -> None: 
    _memory_store[f"req:{run_id}"] = json.dumps(body)

def get_request(run_id: str) -> Dict[str, Any] | None:
    s = _memory_store.get(f"req:{run_id}")
    return json.loads(s) if s else None

import time
from typing import Dict
_ACTIVE_BLOCKS: Dict[str, float] = {}
def block_stop(stop_id: str, ttl_minutes: int = 90):
    _ACTIVE_BLOCKS[stop_id] = time.time() + ttl_minutes*60
def is_blocked(stop_id: str) -> bool:
    now = time.time(); exp = _ACTIVE_BLOCKS.get(stop_id, 0)
    if exp and exp < now: del _ACTIVE_BLOCKS[stop_id]; return False
    return exp > now
def active_blocks() -> Dict[str, float]:
    now = time.time()
    for k,v in list(_ACTIVE_BLOCKS.items()):
        if v < now: del _ACTIVE_BLOCKS[k]
    return dict(_ACTIVE_BLOCKS)

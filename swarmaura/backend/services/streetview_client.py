import os, httpx
GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
def build_streetview_url(lat: float, lng: float, heading: int = 0, fov: int = 90, pitch: int = 0, size="640x640") -> str:
    base = "https://maps.googleapis.com/maps/api/streetview"
    return f"{base}?size={size}&location={lat},{lng}&heading={heading}&fov={fov}&pitch={pitch}&key={GOOGLE_KEY}"
async def fetch_streetview_image(lat: float, lng: float, heading: int = 0) -> bytes:
    if not GOOGLE_KEY: raise RuntimeError("GOOGLE_MAPS_API_KEY missing")
    url = build_streetview_url(lat, lng, heading=heading)
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url); r.raise_for_status(); return r.content
import os, base64, json, httpx
from typing import List, Dict, Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are StreetScout, a logistics curbside-access analyst.
You ONLY return strict JSON that validates against the provided schema.
If the image is insufficient or ambiguous, you must express uncertainty in scores and add a 'notes' explanation.
Do not include any text outside JSON.
"""

JSON_SCHEMA = {
  "type":"object",
  "properties":{
    "analysis":{
      "type":"object",
      "properties":{
        "legal_parking_likelihood":{"type":"number","minimum":0,"maximum":1},
        "curb_ramp_present":{"type":"number","minimum":0,"maximum":1},
        "clear_length_m":{"type":"number"},
        "lane_width_m":{"type":"number"},
        "obstruction_risk":{"type":"number","minimum":0,"maximum":1},
        "traffic_density":{"type":"number","minimum":0,"maximum":1},
        "overall_access":{"type":"number","minimum":0,"maximum":1}
      },
      "required":["legal_parking_likelihood","curb_ramp_present","clear_length_m","lane_width_m","obstruction_risk","traffic_density","overall_access"]
    },
    "recommendation":{
      "type":"object",
      "properties":{
        "service_time_minutes":{"type":"number"},
        "suggested_dropoff":{
          "type":"object",
          "properties":{
            "offset_meters":{"type":"number"},
            "relative_bearing_deg":{"type":"number"},
            "reason":{"type":"string"}
          },
          "required":["offset_meters","relative_bearing_deg","reason"]
        },
        "flags":{"type":"array","items":{"type":"string"}}
      },
      "required":["service_time_minutes","suggested_dropoff","flags"]
    },
    "notes":{"type":"string"}
  },
  "required":["analysis","recommendation","notes"]
}

def build_user_prompt(lat: float, lng: float, vehicle_desc: str) -> str:
    return f"""
Task: Assess curbside access for deliveries at approx lat={lat}, lng={lng} for a {vehicle_desc}.
Score each field 0..1 unless it's a physical length (meters). Use the following semantics:
- legal_parking_likelihood: probability stopping here is lawful and safe.
- curb_ramp_present: likelihood a ramp/curb cut exists nearby for carts.
- clear_length_m: estimated usable curb length free of hydrants/driveways.
- lane_width_m: estimated drivable lane width adjacent to stop area.
- obstruction_risk: likelihood of cones, construction, double-parked cars, or blocked loading.
- traffic_density: relative density now/typical at this location.
- overall_access: your final confidence that a 26-ft box truck can safely stop and unload.

Recommendation:
- service_time_minutes: extra minutes to budget here due to access.
- suggested_dropoff: if the exact point looks bad, propose a nearby offset and bearing + 'reason'.
- flags: short strings e.g. ["no_parking_sign","hydrant","narrow_lane","blocked_dock"].

JSON schema (return STRICT JSON only):
{json.dumps(JSON_SCHEMA)}
""".strip()

async def call_vlm_with_images(images: List[bytes], lat: float, lng: float, vehicle_desc: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY: raise RuntimeError("OPENAI_API_KEY missing")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    content = [{"type":"text","text": build_user_prompt(lat, lng, vehicle_desc)}]
    for img in images:
        b64 = base64.b64encode(img).decode("utf-8")
        content.append({"type":"image_url","image_url":{"url": f"data:image/jpeg;base64,{b64}"}})
    payload = {
        "model": VLM_MODEL,
        "messages": [
            {"role":"system","content": SYSTEM_PROMPT},
            {"role":"user","content": content}
        ],
        "temperature": 0.2,
        "response_format": {"type":"json_object"}
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return json.loads(text)
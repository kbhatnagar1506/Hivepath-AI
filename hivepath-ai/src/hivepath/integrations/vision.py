"""Vision-model scoring of kerbside accessibility.

Given Street View frames, ask a vision model how hard it is to stop, park, and
unload. The response is constrained to JSON and validated before use - an
unvalidated model response previously flowed straight into routing decisions.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Sequence
from typing import Any

import httpx

from hivepath.config import Settings, get_settings
from hivepath.logging_config import get_logger

logger = get_logger(__name__)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_TIMEOUT_S = 60.0

SYSTEM_PROMPT = (
    "You are a logistics kerbside-access analyst. Return ONLY strict JSON "
    "matching the requested schema. If imagery is ambiguous, express that "
    "through lower confidence values and explain in notes."
)

USER_PROMPT_TEMPLATE = """\
Assess kerbside delivery access at approximately lat={lat}, lng={lng} for a {vehicle}.

Return JSON with exactly these keys:
  access_score: integer 0-100, where 100 means trivially easy to stop and unload
  service_time_sec: integer, realistic seconds to complete a delivery here
  findings: array of {{"label": string, "present": boolean, "confidence": 0-1}}
  hazards: array of {{"label": string, "severity": "minor"|"major"|"critical"}}
  notes: string

Scoring guidance:
  - Dedicated loading zone, kerb cuts, or legal nearby parking: high score
  - Clear driveway or loading bay within 30m: raises score
  - Bus lane, hydrant, no-stopping signage, heavy obstruction, stairs-only
    access, or narrow lane: lowers score substantially
  - Any critical hazard caps access_score at 35

Service time guidance: baseline 240s; subtract up to 60s for a dedicated
loading zone; add up to 90s where there is no legal stopping within 60m; add
60s for stairs; add 45s for heavy pedestrian or traffic conflict.
"""

#: Returned when the model is unavailable or its output fails validation.
#: 50 is deliberately mid-scale: it neither promotes nor penalises the stop.
NEUTRAL_RESULT: dict[str, Any] = {
    "access_score": 50,
    "service_time_sec": 240,
    "findings": [],
    "hazards": [],
    "notes": "Accessibility not assessed; neutral default applied.",
    "assessed": False,
}


class VisionUnavailable(RuntimeError):
    """Raised when the vision model cannot be reached or is not configured."""


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return fallback


def validate_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Coerce a model response into the documented shape.

    Never raises: a malformed field falls back to its neutral value so one bad
    response cannot break a routing request.
    """
    findings = []
    for item in payload.get("findings") or []:
        if not isinstance(item, dict) or "label" not in item:
            continue
        findings.append(
            {
                "label": str(item["label"]),
                "present": bool(item.get("present", True)),
                "confidence": _clamp(item.get("confidence", 0.5), 0.0, 1.0, 0.5),
            }
        )

    hazards = []
    for item in payload.get("hazards") or []:
        if not isinstance(item, dict) or "label" not in item:
            continue
        severity = str(item.get("severity", "minor")).lower()
        if severity not in {"minor", "major", "critical"}:
            severity = "minor"
        hazards.append({"label": str(item["label"]), "severity": severity})

    score = int(_clamp(payload.get("access_score", 50), 0, 100, 50))
    # Enforce the documented cap rather than trusting the model to apply it.
    if any(h["severity"] == "critical" for h in hazards):
        score = min(score, 35)

    return {
        "access_score": score,
        "service_time_sec": int(_clamp(payload.get("service_time_sec", 240), 30, 3600, 240)),
        "findings": findings,
        "hazards": hazards,
        "notes": str(payload.get("notes", "")),
        "assessed": True,
    }


async def analyze_images(
    images: Sequence[bytes],
    lat: float,
    lng: float,
    *,
    vehicle_desc: str = "26-ft box truck",
    settings: Settings | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Score kerbside access from Street View frames."""
    settings = settings or get_settings()
    if not settings.has_vlm_credentials:
        raise VisionUnavailable("OPENAI_API_KEY is not configured")
    if not images:
        raise VisionUnavailable("no imagery supplied")

    content: list[dict[str, Any]] = [
        {"type": "text", "text": USER_PROMPT_TEMPLATE.format(lat=lat, lng=lng, vehicle=vehicle_desc)}
    ]
    for image in images:
        encoded = base64.b64encode(image).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "low"},
            }
        )

    payload = {
        "model": settings.vlm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            OPENAI_CHAT_URL,
            json=payload,
            headers={"Authorization": f"Bearer {settings.openai_key}"},
        )
        response.raise_for_status()
        body = response.json()

    try:
        raw = json.loads(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        raise VisionUnavailable(f"vision model returned unusable output: {exc}") from exc

    return validate_analysis(raw)

from dataclasses import dataclass

@dataclass
class AccessThresholds:
    min_total_score: float = 0.55
    min_legal_parking: float = 0.6
    min_lane_width_m: float = 3.0
    min_clear_length_m: float = 9.0
    max_obstruction_score: float = 0.4

DEFAULT_THRESHOLDS = AccessThresholds()

def decision_from_scores(scores: dict, th: AccessThresholds = DEFAULT_THRESHOLDS):
    overall = float(scores.get("overall_access", 0.0))
    legal   = float(scores.get("legal_parking_likelihood", 0.0))
    lane    = float(scores.get("lane_width_m", 0.0))
    clear   = float(scores.get("clear_length_m", 0.0))
    obstruct= float(scores.get("obstruction_risk", 1.0))
    block = (overall < th.min_total_score or legal < th.min_legal_parking or lane < th.min_lane_width_m or clear < th.min_clear_length_m or obstruct > th.max_obstruction_score)
    severity = 0.9 if block else max(0.1, 1 - overall)
    return {"block": block, "severity": round(severity, 2)}
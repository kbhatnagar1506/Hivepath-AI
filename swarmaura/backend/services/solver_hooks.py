"""
Solver Hooks for ML Integration
Integrates ML predictions into the OR-Tools solver pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from service_time_model import predictor_singleton
from datetime import datetime
from typing import List, Dict, Any

def enrich_service_times(stops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enrich stops with ML-predicted service times
    
    Args:
        stops: List of stop dictionaries
        
    Returns:
        List of stops with added service_min field
    """
    if not stops:
        return stops
    
    print(f"🧠 Enriching {len(stops)} stops with ML-predicted service times...")
    
    # Prepare features for ML prediction
    feats = []
    id2idx = {s["id"]: i+1 for i, s in enumerate(stops)}  # Map stop IDs to indices
    
    now = datetime.now()
    
    for s in stops:
        feats.append({
            "id": s["id"],
            "node_idx": id2idx.get(s["id"], 0),
            "demand": s.get("demand", 120),
            "access_score": s.get("access_score", 0.6),
            "hour": now.hour,
            "weekday": now.weekday()
        })
    
    # Get ML predictions
    predictions = predictor_singleton.predict_minutes(feats)
    
    # Add predictions to stops
    for s, pred_min in zip(stops, predictions):
        s["service_min"] = float(round(pred_min, 1))
    
    # Log some statistics
    if predictions:
        avg_service = sum(predictions) / len(predictions)
        min_service = min(predictions)
        max_service = max(predictions)
        
        print(f"   📊 Service time stats: avg={avg_service:.1f}min, range=[{min_service:.1f}-{max_service:.1f}]min")
        
        # Show first few predictions
        for i, (s, pred) in enumerate(zip(stops[:3], predictions[:3])):
            print(f"   📍 {s['id']}: {pred:.1f}min (demand={s.get('demand', 120)}, access={s.get('access_score', 0.6):.2f})")
        if len(stops) > 3:
            print(f"   ... and {len(stops)-3} more stops")
    
    return stops

def enrich_service_times_with_context(stops: List[Dict[str, Any]], 
                                    context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Enrich stops with ML-predicted service times using additional context
    
    Args:
        stops: List of stop dictionaries
        context: Optional context dict with keys like 'weather_impact', 'traffic_level', etc.
        
    Returns:
        List of stops with added service_min field
    """
    if not stops:
        return stops
    
    print(f"🧠 Enriching {len(stops)} stops with contextual ML predictions...")
    
    # Prepare features for ML prediction
    feats = []
    id2idx = {s["id"]: i+1 for i, s in enumerate(stops)}
    
    now = datetime.now()
    
    for s in stops:
        # Base features
        feat = {
            "id": s["id"],
            "node_idx": id2idx.get(s["id"], 0),
            "demand": s.get("demand", 120),
            "access_score": s.get("access_score", 0.6),
            "hour": now.hour,
            "weekday": now.weekday()
        }
        
        # Add context if available
        if context:
            # Weather impact could affect service time
            if "weather_impact" in context:
                # Adjust access score based on weather
                weather_factor = max(0.5, 1.0 - context["weather_impact"] / 100.0)
                feat["access_score"] = min(1.0, feat["access_score"] * weather_factor)
            
            # Traffic level could affect service time
            if "traffic_level" in context:
                # Adjust demand based on traffic (more traffic = more complex service)
                traffic_factor = {"light": 1.0, "moderate": 1.1, "heavy": 1.2, "severe": 1.3}
                feat["demand"] = int(feat["demand"] * traffic_factor.get(context["traffic_level"], 1.0))
        
        feats.append(feat)
    
    # Get ML predictions
    predictions = predictor_singleton.predict_minutes(feats)
    
    # Add predictions to stops
    for s, pred_min in zip(stops, predictions):
        s["service_min"] = float(round(pred_min, 1))
    
    # Log statistics
    if predictions:
        avg_service = sum(predictions) / len(predictions)
        print(f"   📊 Contextual service time avg: {avg_service:.1f}min")
    
    return stops

def get_service_time_model_info() -> Dict[str, Any]:
    """Get information about the service time prediction model"""
    return predictor_singleton.get_model_info()

def test_solver_hooks():
    """Test the solver hooks"""
    print("🧪 Testing Solver Hooks")
    print("=" * 30)
    
    # Test stops
    test_stops = [
        {
            "id": "S_A",
            "lat": 42.37,
            "lng": -71.05,
            "demand": 150,
            "access_score": 0.72,
            "priority": 1
        },
        {
            "id": "S_B",
            "lat": 42.34,
            "lng": -71.10,
            "demand": 140,
            "access_score": 0.61,
            "priority": 2
        },
        {
            "id": "S_C",
            "lat": 42.39,
            "lng": -71.02,
            "demand": 160,
            "access_score": 0.55,
            "priority": 1
        }
    ]
    
    print("📊 Original stops:")
    for s in test_stops:
        print(f"   📍 {s['id']}: demand={s['demand']}, access={s['access_score']:.2f}")
    
    # Test basic enrichment
    print("\n🧠 Testing basic service time enrichment...")
    enriched_stops = enrich_service_times(test_stops.copy())
    
    print("\n📊 Enriched stops:")
    for s in enriched_stops:
        print(f"   📍 {s['id']}: service_min={s['service_min']:.1f}min")
    
    # Test contextual enrichment
    print("\n🧠 Testing contextual service time enrichment...")
    context = {
        "weather_impact": 25,  # Moderate weather impact
        "traffic_level": "heavy"
    }
    
    contextual_stops = enrich_service_times_with_context(test_stops.copy(), context)
    
    print("\n📊 Contextual enriched stops:")
    for s in contextual_stops:
        print(f"   📍 {s['id']}: service_min={s['service_min']:.1f}min")
    
    # Show model info
    model_info = get_service_time_model_info()
    print(f"\n📊 Model Info:")
    print(f"   🎯 Type: {model_info['model_type']}")
    print(f"   📈 Mean Service Time: {model_info['y_mean']:.2f} minutes" if model_info['y_mean'] else "   📈 Mean Service Time: N/A")
    
    print("\n✅ Solver hooks ready!")

if __name__ == "__main__":
    test_solver_hooks()

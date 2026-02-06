"""
Access enricher service that integrates StreetScout analysis with VRP stops.
"""
import asyncio
from typing import List, Dict, Any, Optional
from services.streetscout import AnalyzeReq, analyze_location, batch_analyze_locations

class AccessEnricher:
    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self._cache = {}  # Simple in-memory cache
    
    def _cache_key(self, lat: float, lng: float, headings: List[int]) -> str:
        """Generate cache key for location and headings."""
        return f"{lat:.6f}_{lng:.6f}_{'-'.join(map(str, sorted(headings)))}"
    
    async def enrich_stops(self, stops: List[Dict[str, Any]], 
                          headings: Optional[List[int]] = None,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Enrich stops with access scores and predicted service times.
        
        Args:
            stops: List of stop dictionaries with lat/lng
            headings: Street View headings to analyze (default: [0, 90, 180, 270])
            use_cache: Whether to use cached results
            
        Returns:
            Enriched stops with access_score and pred_service_time_sec
        """
        if not stops:
            return stops
            
        headings = headings or [0, 90, 180, 270]
        enriched_stops = []
        
        # Check cache first
        if use_cache:
            for stop in stops:
                cache_key = self._cache_key(stop["lat"], stop["lng"], headings)
                if cache_key in self._cache:
                    enriched_stop = stop.copy()
                    enriched_stop.update(self._cache[cache_key])
                    enriched_stops.append(enriched_stop)
                else:
                    enriched_stops.append(stop)
        else:
            enriched_stops = [stop.copy() for stop in stops]
        
        # Find stops that need analysis
        to_analyze = []
        for i, stop in enumerate(enriched_stops):
            if "access_score" not in stop:
                to_analyze.append((i, stop))
        
        if not to_analyze:
            return enriched_stops
        
        # Prepare analysis requests
        requests = []
        for i, stop in to_analyze:
            requests.append(AnalyzeReq(
                lat=stop["lat"],
                lng=stop["lng"],
                headings=headings
            ))
        
        # Batch analyze
        try:
            results = await batch_analyze_locations(requests, self.max_concurrent)
            
            # Update stops with results
            for (i, stop), result in zip(to_analyze, results):
                enriched_stops[i]["access_score"] = result.access_score
                enriched_stops[i]["pred_service_time_sec"] = result.pred_service_time_sec
                enriched_stops[i]["access_findings"] = [f.model_dump() for f in result.findings]
                enriched_stops[i]["access_hazards"] = [h.model_dump() for h in result.hazards]
                
                # Cache the result
                if use_cache:
                    cache_key = self._cache_key(stop["lat"], stop["lng"], headings)
                    self._cache[cache_key] = {
                        "access_score": result.access_score,
                        "pred_service_time_sec": result.pred_service_time_sec,
                        "access_findings": [f.model_dump() for f in result.findings],
                        "access_hazards": [h.model_dump() for h in result.hazards]
                    }
        
        except Exception as e:
            # Fallback: add default access scores
            for i, stop in to_analyze:
                enriched_stops[i]["access_score"] = 50
                enriched_stops[i]["pred_service_time_sec"] = 240
                enriched_stops[i]["access_findings"] = []
                enriched_stops[i]["access_hazards"] = []
        
        return enriched_stops

# Global instance
access_enricher = AccessEnricher()

async def enrich_stops_with_access(stops: List[Dict[str, Any]], 
                                 headings: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Convenience function to enrich stops with access analysis."""
    return await access_enricher.enrich_stops(stops, headings)

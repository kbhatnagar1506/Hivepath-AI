"""
Google Maps API integration for real-world routing
"""
import os
import requests
import time
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
import json

class GoogleMapsClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not self.api_key:
            raise ValueError("Google Maps API key is required")
        
        self.base_url = "https://maps.googleapis.com/maps/api"
        self.rate_limit_delay = 0.1  # 100ms between requests
    
    @lru_cache(maxsize=1000)
    def get_distance_matrix(self, origins: Tuple, destinations: Tuple, mode: str = "driving") -> Dict[str, Any]:
        """
        Get distance matrix from Google Maps API
        origins: ((lat1, lng1), (lat2, lng2), ...)
        destinations: ((lat1, lng1), (lat2, lng2), ...)
        """
        origins_str = "|".join([f"{lat},{lng}" for lat, lng in origins])
        destinations_str = "|".join([f"{lat},{lng}" for lat, lng in destinations])
        
        url = f"{self.base_url}/distancematrix/json"
        params = {
            "origins": origins_str,
            "destinations": destinations_str,
            "mode": mode,
            "traffic_model": "best_guess",  # Use real-time traffic
            "departure_time": "now",
            "units": "metric",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "OK":
                raise Exception(f"Google Maps API error: {data.get('error_message', 'Unknown error')}")
            
            return data
        except Exception as e:
            print(f"Google Maps API error: {e}")
            # Fallback to haversine calculation
            return self._fallback_distance_matrix(origins, destinations)
    
    def _fallback_distance_matrix(self, origins: Tuple, destinations: Tuple) -> Dict[str, Any]:
        """Fallback to haversine calculation if Google Maps fails"""
        from .ortools_solver import _haversine_km
        
        rows = []
        for origin in origins:
            row = []
            for dest in destinations:
                distance = _haversine_km(origin, dest)
                row.append({
                    "distance": {"value": int(distance * 1000)},  # Convert to meters
                    "duration": {"value": int((distance / 40) * 3600)}  # Assume 40 km/h
                })
            rows.append(row)
        
        return {
            "rows": [{"elements": row} for row in rows],
            "status": "OK"
        }
    
    def get_route(self, origin: Tuple[float, float], destination: Tuple[float, float], 
                  mode: str = "driving", waypoints: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Get detailed route from Google Maps API
        """
        origin_str = f"{origin[0]},{origin[1]}"
        dest_str = f"{destination[0]},{destination[1]}"
        
        url = f"{self.base_url}/directions/json"
        params = {
            "origin": origin_str,
            "destination": dest_str,
            "mode": mode,
            "traffic_model": "best_guess",
            "departure_time": "now",
            "units": "metric",
            "key": self.api_key
        }
        
        if waypoints:
            waypoints_str = "|".join([f"{lat},{lng}" for lat, lng in waypoints])
            params["waypoints"] = waypoints_str
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data["status"] != "OK":
                raise Exception(f"Google Maps API error: {data.get('error_message', 'Unknown error')}")
            
            return data
        except Exception as e:
            print(f"Google Maps API error: {e}")
            return {"status": "ERROR", "error_message": str(e)}
    
    def get_place_details(self, place_id: str) -> Dict[str, Any]:
        """Get detailed information about a place"""
        url = f"{self.base_url}/place/details/json"
        params = {
            "place_id": place_id,
            "fields": "geometry,formatted_address,name,types",
            "key": self.api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Google Maps API error: {e}")
            return {"status": "ERROR", "error_message": str(e)}

def create_google_maps_distance_matrix(locations: List[Dict[str, Any]], 
                                     speed_kmph: float = 40.0) -> Tuple[List[List[float]], List[List[int]]]:
    """
    Create distance and time matrices using Google Maps API
    Returns: (distance_km, time_min)
    """
    try:
        client = GoogleMapsClient()
    except ValueError:
        print("Google Maps API key not found, using haversine fallback")
        return _create_haversine_matrices(locations, speed_kmph)
    
    n = len(locations)
    dist_km = [[0.0] * n for _ in range(n)]
    time_min = [[0] * n for _ in range(n)]
    
    # Convert locations to coordinate tuples
    coords = [(loc["lat"], loc["lng"]) for loc in locations]
    
    # Get distance matrix from Google Maps
    origins = tuple(coords)
    destinations = tuple(coords)
    
    try:
        matrix_data = client.get_distance_matrix(origins, destinations)
        
        if matrix_data["status"] == "OK":
            for i, row in enumerate(matrix_data["rows"]):
                for j, element in enumerate(row["elements"]):
                    if element["status"] == "OK":
                        # Distance in meters, convert to km
                        dist_km[i][j] = element["distance"]["value"] / 1000.0
                        # Duration in seconds, convert to minutes
                        time_min[i][j] = max(1, element["duration"]["value"] // 60)
                    else:
                        # Fallback to haversine for failed elements
                        from .ortools_solver import _haversine_km
                        dist_km[i][j] = _haversine_km(coords[i], coords[j])
                        time_min[i][j] = max(1, int((dist_km[i][j] / speed_kmph) * 60))
        else:
            print("Google Maps API failed, using haversine fallback")
            return _create_haversine_matrices(locations, speed_kmph)
            
    except Exception as e:
        print(f"Google Maps API error: {e}, using haversine fallback")
        return _create_haversine_matrices(locations, speed_kmph)
    
    return dist_km, time_min

def _create_haversine_matrices(locations: List[Dict[str, Any]], speed_kmph: float) -> Tuple[List[List[float]], List[List[int]]]:
    """Fallback to haversine calculation"""
    from .ortools_solver import _haversine_km
    
    n = len(locations)
    dist_km = [[0.0] * n for _ in range(n)]
    time_min = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d = _haversine_km((locations[i]["lat"], locations[i]["lng"]),
                                (locations[j]["lat"], locations[j]["lng"]))
                dist_km[i][j] = d
                time_min[i][j] = max(1, int((d / speed_kmph) * 60))
    
    return dist_km, time_min

def get_real_time_route(origin: Tuple[float, float], 
                       destination: Tuple[float, float],
                       waypoints: Optional[List[Tuple[float, float]]] = None) -> Dict[str, Any]:
    """
    Get real-time route with traffic information
    """
    try:
        client = GoogleMapsClient()
        return client.get_route(origin, destination, waypoints=waypoints)
    except ValueError:
        return {"status": "ERROR", "error_message": "Google Maps API key not configured"}

# Configuration helper
def configure_google_maps(api_key: str):
    """Set Google Maps API key"""
    os.environ["GOOGLE_MAPS_API_KEY"] = api_key
    print("Google Maps API key configured successfully")

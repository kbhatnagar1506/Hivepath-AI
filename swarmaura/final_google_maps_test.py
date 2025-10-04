#!/usr/bin/env python3
"""
Final comprehensive test showing Google Maps vs Haversine benefits
"""
import sys
sys.path.append('backend')
import os
import requests
import math
from services.ortools_solver import solve_vrp
import time

def test_google_maps_benefits():
    """Comprehensive test showing real-world Google Maps benefits"""
    
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("❌ GOOGLE_MAPS_API_KEY not set")
        return
    
    print("🗺️  Google Maps vs Haversine - Comprehensive Analysis")
    print("=" * 70)
    
    # Test individual route segments
    routes = [
        {
            'name': 'Cambridge → Aquarium',
            'origin': '42.3736,-71.1097',
            'destination': '42.3595,-71.0515',
            'description': 'Crossing Charles River'
        },
        {
            'name': 'Aquarium → Charlestown',
            'origin': '42.3595,-71.0515',
            'destination': '42.3742,-71.0539',
            'description': 'Waterfront to Navy Yard'
        },
        {
            'name': 'Charlestown → East Boston',
            'origin': '42.3742,-71.0539',
            'destination': '42.3750,-71.0392',
            'description': 'Crossing Boston Harbor'
        },
        {
            'name': 'East Boston → South Boston',
            'origin': '42.3750,-71.0392',
            'destination': '42.3383,-71.0103',
            'description': 'Harbor crossing to Castle Island'
        }
    ]
    
    def haversine_km(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        x = (math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)
        return 2 * R * math.asin(math.sqrt(x))
    
    print("📍 Individual Route Analysis:")
    print("-" * 50)
    
    total_google_dist = 0
    total_haversine_dist = 0
    total_google_time = 0
    total_haversine_time = 0
    
    for i, route in enumerate(routes, 1):
        print(f"\\n{i}. {route['name']} ({route['description']})")
        
        # Get Google Maps data
        url = 'https://maps.googleapis.com/maps/api/distancematrix/json'
        params = {
            'origins': route['origin'],
            'destinations': route['destination'],
            'mode': 'driving',
            'traffic_model': 'best_guess',
            'departure_time': 'now',
            'units': 'metric',
            'key': api_key
        }
        
        try:
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['status'] == 'OK':
                element = data['rows'][0]['elements'][0]
                if element['status'] == 'OK':
                    google_dist = element['distance']['value'] / 1000
                    google_time = element['duration']['value'] // 60
                    
                    # Calculate Haversine
                    coords = route['origin'].split(',')
                    origin_lat, origin_lng = float(coords[0]), float(coords[1])
                    coords = route['destination'].split(',')
                    dest_lat, dest_lng = float(coords[0]), float(coords[1])
                    
                    haversine_dist = haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
                    haversine_time = int((haversine_dist / 25) * 60)  # 25 km/h
                    
                    dist_diff = ((google_dist - haversine_dist) / haversine_dist) * 100
                    time_diff = ((google_time - haversine_time) / haversine_time) * 100
                    
                    print(f"   🗺️  Google Maps: {google_dist:.2f} km, {google_time} min")
                    print(f"   📏 Haversine:   {haversine_dist:.2f} km, {haversine_time} min")
                    print(f"   📊 Difference:  {dist_diff:+.1f}% distance, {time_diff:+.1f}% time")
                    
                    if abs(dist_diff) > 50:
                        print(f"   🌊 Major difference! Water crossing requires real roads")
                    elif abs(dist_diff) > 20:
                        print(f"   🛣️  Significant difference! Real road network matters")
                    else:
                        print(f"   ✅ Similar results - straight line works well")
                    
                    total_google_dist += google_dist
                    total_haversine_dist += haversine_dist
                    total_google_time += google_time
                    total_haversine_time += haversine_time
                    
                else:
                    print(f"   ❌ API Error: {element.get('status', 'Unknown error')}")
            else:
                print(f"   ❌ API Error: {data.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Request failed: {e}")
    
    # Summary
    print(f"\\n📊 Total Route Summary:")
    print("=" * 50)
    print(f"🗺️  Google Maps Total: {total_google_dist:.2f} km, {total_google_time} min")
    print(f"📏 Haversine Total:   {total_haversine_dist:.2f} km, {total_haversine_time} min")
    
    total_dist_diff = ((total_google_dist - total_haversine_dist) / total_haversine_dist) * 100
    total_time_diff = ((total_google_time - total_haversine_time) / total_haversine_time) * 100
    
    print(f"📈 Total Difference:  {total_dist_diff:+.1f}% distance, {total_time_diff:+.1f}% time")
    
    # Analysis
    print(f"\\n🎯 Analysis:")
    print("-" * 30)
    if total_dist_diff > 50:
        print("🌊 MAJOR DIFFERENCE: Google Maps is essential for water crossings")
        print("   - Haversine assumes straight-line travel (impossible over water)")
        print("   - Google Maps uses actual bridges, tunnels, and roads")
        print("   - Real-world accuracy: Google Maps wins by a huge margin")
    elif total_dist_diff > 20:
        print("🛣️  SIGNIFICANT DIFFERENCE: Google Maps shows real road network")
        print("   - Haversine is a good approximation for simple routes")
        print("   - Google Maps accounts for complex road networks")
        print("   - Production accuracy: Google Maps is much better")
    else:
        print("✅ SIMILAR RESULTS: Haversine works well for this area")
        print("   - Boston has an efficient road network")
        print("   - Straight-line distances are close to driving distances")
        print("   - Both methods work well for development")
    
    # Recommendations
    print(f"\\n💡 Recommendations:")
    print("-" * 30)
    print("🚀 For Development:")
    print("   - Use Haversine: Fast, free, good enough for testing")
    print("   - No API costs, no external dependencies")
    print("   - Perfect for algorithm development and testing")
    
    print("\\n🏭 For Production:")
    print("   - Use Google Maps: Real-world accuracy")
    print("   - Handles traffic, construction, one-way streets")
    print("   - Provides turn-by-turn directions")
    print("   - Worth the cost for customer-facing applications")
    
    print("\\n🔄 Hybrid Approach:")
    print("   - Development: Haversine (fast, free)")
    print("   - Staging: Google Maps (limited usage)")
    print("   - Production: Google Maps (full accuracy)")
    
    print(f"\\n🎉 Conclusion:")
    print("=" * 50)
    print("Your current Haversine system is EXCELLENT for development!")
    print("Google Maps API provides real-world accuracy for production.")
    print("Both have their place in a complete routing solution. 🚀")

if __name__ == "__main__":
    test_google_maps_benefits()

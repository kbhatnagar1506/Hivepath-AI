#!/usr/bin/env python3
"""
Test Google Maps API setup and show how to enable required APIs
"""
import os
import requests

def test_google_maps_setup():
    """Test if Google Maps API is properly configured"""
    
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("❌ GOOGLE_MAPS_API_KEY environment variable not set")
        return False
    
    print("🔑 Google Maps API Key Found")
    print(f"Key: {api_key[:10]}...{api_key[-4:]}")
    print()
    
    # Test with a simple Geocoding API call (usually enabled by default)
    print("🧪 Testing API access with Geocoding API...")
    
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        'address': 'Boston, MA',
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            print("✅ Geocoding API: Working")
            print("✅ API Key: Valid")
        else:
            print(f"❌ Geocoding API Error: {data.get('error_message', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    # Test Distance Matrix API
    print("\n🧪 Testing Distance Matrix API...")
    
    url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    params = {
        'origins': '42.3601,-71.0589',  # Faneuil Hall
        'destinations': '42.3467,-71.0972',  # Fenway Park
        'mode': 'driving',
        'key': api_key
    }
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        
        if data['status'] == 'OK':
            print("✅ Distance Matrix API: Working")
            element = data['rows'][0]['elements'][0]
            if element['status'] == 'OK':
                distance = element['distance']['value'] / 1000
                duration = element['duration']['value'] // 60
                print(f"   📏 Distance: {distance:.2f} km")
                print(f"   ⏰ Duration: {duration} min")
            else:
                print(f"❌ Distance Matrix Error: {element.get('status', 'Unknown error')}")
                return False
        else:
            print(f"❌ Distance Matrix API Error: {data.get('error_message', 'Unknown error')}")
            print("\n🔧 To fix this, you need to enable the Distance Matrix API:")
            print("   1. Go to https://console.cloud.google.com/")
            print("   2. Select your project")
            print("   3. Go to 'APIs & Services' > 'Library'")
            print("   4. Search for 'Distance Matrix API'")
            print("   5. Click 'Enable'")
            print("   6. Also enable 'Directions API' for full functionality")
            return False
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False
    
    print("\n🎉 All APIs are working! Your Google Maps integration is ready.")
    return True

if __name__ == "__main__":
    test_google_maps_setup()

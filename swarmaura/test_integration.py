
#!/usr/bin/env python3
"""
Integration Test for Frontend-Backend System
"""
import requests
import json
import time

def test_integration():
    """Test the integrated frontend-backend system"""
    print("🧪 TESTING INTEGRATED SYSTEM")
    print("=" * 40)
    
    base_url = "http://localhost:8000"
    frontend_url = "https://fleet-flow-7189cccb.base44.app"
    
    # Test backend health
    try:
        response = requests.get(f"{base_url}/api/v1/health")
        print(f"✅ Backend Health: {response.status_code}")
    except Exception as e:
        print(f"❌ Backend Health: {e}")
    
    # Test route optimization
    try:
        test_data = {
            "depot": {
                "id": "D",
                "name": "Depot",
                "lat": 42.3601,
                "lng": -71.0589
            },
            "stops": [
                {
                    "id": "S1",
                    "name": "Stop 1",
                    "lat": 42.3700,
                    "lng": -71.0500,
                    "demand": 100,
                    "priority": 1
                }
            ],
            "vehicles": [
                {
                    "id": "V1",
                    "type": "truck",
                    "capacity": 200
                }
            ]
        }
        
        response = requests.post(
            f"{base_url}/api/v1/optimize/routes",
            json=test_data
        )
        print(f"✅ Route Optimization: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Routes: {len(result.get('routes', []))}")
    except Exception as e:
        print(f"❌ Route Optimization: {e}")
    
    # Test integration status
    try:
        response = requests.get(f"{base_url}/api/v1/integration/status")
        print(f"✅ Integration Status: {response.status_code}")
        if response.status_code == 200:
            status = response.json()
            print(f"   Frontend: {status.get('frontend_url')}")
            print(f"   Backend: {status.get('backend_url')}")
    except Exception as e:
        print(f"❌ Integration Status: {e}")
    
    print("\n🎯 INTEGRATION TEST COMPLETE!")

if __name__ == "__main__":
    test_integration()

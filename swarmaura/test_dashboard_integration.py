#!/usr/bin/env python3
"""
Test script to verify the integrated dashboard works with our live API
"""

import requests
import json
import time
import subprocess
import sys
import os

def test_api_endpoints():
    """Test all API endpoints that the dashboard uses"""
    base_url = "http://localhost:8001"
    
    print("🧪 TESTING API ENDPOINTS FOR DASHBOARD INTEGRATION")
    print("=" * 60)
    
    endpoints = [
        ("/api/health", "Health Check"),
        ("/api/locations", "Locations"),
        ("/api/vehicles", "Vehicles"),
        ("/api/analytics/overview", "Analytics"),
        ("/api/predictions/service-times", "Predictions"),
        ("/api/accessibility", "Accessibility"),
        ("/api/environmental/weather", "Weather"),
        ("/api/environmental/traffic", "Traffic"),
        ("/api/bulk/all", "Bulk Data")
    ]
    
    results = {}
    
    for endpoint, name in endpoints:
        try:
            print(f"🔍 Testing {name}...")
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ {name}: {response.status_code} - {data.get('count', 'N/A')} items")
                results[endpoint] = {
                    "status": "success",
                    "status_code": response.status_code,
                    "data_count": data.get('count', 'N/A'),
                    "response_time": response.elapsed.total_seconds()
                }
            else:
                print(f"   ❌ {name}: {response.status_code}")
                results[endpoint] = {
                    "status": "error",
                    "status_code": response.status_code
                }
        except Exception as e:
            print(f"   ❌ {name}: Error - {str(e)}")
            results[endpoint] = {
                "status": "error",
                "error": str(e)
            }
    
    return results

def test_dashboard_data_flow():
    """Test the data flow that the dashboard expects"""
    print("\n🔄 TESTING DASHBOARD DATA FLOW")
    print("=" * 40)
    
    try:
        # Test bulk data endpoint (main dashboard data source)
        response = requests.get("http://localhost:8001/api/bulk/all", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Bulk data endpoint working")
            
            # Check required data structure
            required_keys = ['locations', 'vehicles', 'analytics', 'service_predictions']
            missing_keys = [key for key in required_keys if key not in data.get('data', {})]
            
            if missing_keys:
                print(f"⚠️  Missing keys in bulk data: {missing_keys}")
            else:
                print("✅ All required data keys present")
            
            # Check data counts
            bulk_data = data.get('data', {})
            print(f"   📊 Locations: {len(bulk_data.get('locations', []))}")
            print(f"   🚛 Vehicles: {len(bulk_data.get('vehicles', []))}")
            print(f"   🧠 Predictions: {len(bulk_data.get('service_predictions', []))}")
            print(f"   📈 Analytics: {bulk_data.get('analytics', {})}")
            
            return True
        else:
            print(f"❌ Bulk data endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Data flow test failed: {str(e)}")
        return False

def test_cors_headers():
    """Test CORS headers for browser access"""
    print("\n🌐 TESTING CORS HEADERS")
    print("=" * 30)
    
    try:
        response = requests.options("http://localhost:8001/api/health", 
                                  headers={"Origin": "http://localhost:3000"})
        
        cors_headers = {
            'Access-Control-Allow-Origin': response.headers.get('Access-Control-Allow-Origin'),
            'Access-Control-Allow-Methods': response.headers.get('Access-Control-Allow-Methods'),
            'Access-Control-Allow-Headers': response.headers.get('Access-Control-Allow-Headers')
        }
        
        print(f"✅ CORS Headers:")
        for header, value in cors_headers.items():
            print(f"   {header}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ CORS test failed: {str(e)}")
        return False

def generate_dashboard_config():
    """Generate configuration for the dashboard"""
    print("\n⚙️  GENERATING DASHBOARD CONFIG")
    print("=" * 35)
    
    config = {
        "api_base_url": "http://localhost:8001",
        "endpoints": {
            "health": "/api/health",
            "locations": "/api/locations",
            "vehicles": "/api/vehicles",
            "analytics": "/api/analytics/overview",
            "predictions": "/api/predictions/service-times",
            "accessibility": "/api/accessibility",
            "weather": "/api/environmental/weather",
            "traffic": "/api/environmental/traffic",
            "bulk": "/api/bulk/all"
        },
        "timeout": 5000,
        "retry_attempts": 3
    }
    
    with open("/Users/krishnabhatnagar/hackharvard/swarmaura/integrated_dashboard/.env.local", "w") as f:
        f.write(f"NEXT_PUBLIC_API_URL={config['api_base_url']}\n")
    
    print("✅ Dashboard config generated")
    print(f"   API Base URL: {config['api_base_url']}")
    print("   Environment file: .env.local")
    
    return config

def main():
    """Main test function"""
    print("🎯 SWARMAURA DASHBOARD INTEGRATION TEST")
    print("=" * 50)
    print("")
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:8001/api/health", timeout=3)
        if response.status_code != 200:
            print("❌ API is not running. Please start the API first:")
            print("   cd /Users/krishnabhatnagar/hackharvard/swarmaura")
            print("   python3 data_extraction_api.py")
            sys.exit(1)
    except:
        print("❌ API is not running. Please start the API first:")
        print("   cd /Users/krishnabhatnagar/hackharvard/swarmaura")
        print("   python3 data_extraction_api.py")
        sys.exit(1)
    
    print("✅ API is running")
    print("")
    
    # Run tests
    api_results = test_api_endpoints()
    data_flow_ok = test_dashboard_data_flow()
    cors_ok = test_cors_headers()
    config = generate_dashboard_config()
    
    # Summary
    print("\n📊 INTEGRATION TEST SUMMARY")
    print("=" * 35)
    
    successful_endpoints = sum(1 for result in api_results.values() if result.get('status') == 'success')
    total_endpoints = len(api_results)
    
    print(f"✅ API Endpoints: {successful_endpoints}/{total_endpoints} working")
    print(f"✅ Data Flow: {'Working' if data_flow_ok else 'Failed'}")
    print(f"✅ CORS Headers: {'Working' if cors_ok else 'Failed'}")
    print(f"✅ Dashboard Config: Generated")
    
    if successful_endpoints == total_endpoints and data_flow_ok and cors_ok:
        print("\n🎉 ALL TESTS PASSED! Dashboard integration is ready.")
        print("")
        print("🚀 TO START THE DASHBOARD:")
        print("   cd /Users/krishnabhatnagar/hackharvard/swarmaura/integrated_dashboard")
        print("   ./start_dashboard.sh")
        print("")
        print("🌐 Dashboard will be available at: http://localhost:3000")
        print("📡 API is running at: http://localhost:8001")
    else:
        print("\n❌ Some tests failed. Please check the API and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()

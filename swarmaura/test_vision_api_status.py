#!/usr/bin/env python3
"""
Test Google Cloud Vision API Status
"""

import requests
import json
import base64

GOOGLE_MAPS_API_KEY = "AIzaSyAUeiyRuSuKcnPBFmezWCFuUStl8unv7_0"
GOOGLE_CLOUD_VISION_API = "https://vision.googleapis.com/v1/images:annotate"

def test_vision_api_detailed():
    """Test Google Cloud Vision API with detailed error reporting"""
    
    print("🔍 DETAILED GOOGLE CLOUD VISION API TEST")
    print("=" * 50)
    
    # Test with a simple Street View image
    test_image_url = "https://maps.googleapis.com/maps/api/streetview?location=42.3503,-71.074&heading=0&pitch=0&fov=90&size=400x400&key=" + GOOGLE_MAPS_API_KEY
    
    print(f"📸 Test Image URL: {test_image_url[:80]}...")
    
    try:
        # Download test image
        print("📥 Downloading test image...")
        image_response = requests.get(test_image_url, timeout=10)
        print(f"   Status: {image_response.status_code}")
        
        if image_response.status_code != 200:
            print(f"❌ Failed to download test image: {image_response.status_code}")
            return False
        
        print(f"   Image size: {len(image_response.content)} bytes")
        
        # Encode image for Vision API
        print("🔄 Encoding image for Vision API...")
        image_content = base64.b64encode(image_response.content).decode('utf-8')
        print(f"   Encoded size: {len(image_content)} characters")
        
        # Prepare Vision API request
        print("📝 Preparing Vision API request...")
        vision_request = {
            "requests": [
                {
                    "image": {
                        "content": image_content
                    },
                    "features": [
                        {
                            "type": "LABEL_DETECTION",
                            "maxResults": 5
                        }
                    ]
                }
            ]
        }
        
        print("🚀 Calling Google Cloud Vision API...")
        print(f"   API Endpoint: {GOOGLE_CLOUD_VISION_API}")
        print(f"   API Key: {GOOGLE_MAPS_API_KEY[:20]}...")
        
        # Call Google Cloud Vision API
        vision_response = requests.post(
            f"{GOOGLE_CLOUD_VISION_API}?key={GOOGLE_MAPS_API_KEY}",
            headers={"Content-Type": "application/json"},
            json=vision_request,
            timeout=30
        )
        
        print(f"   Response Status: {vision_response.status_code}")
        print(f"   Response Headers: {dict(vision_response.headers)}")
        
        if vision_response.status_code == 200:
            result = vision_response.json()
            print("✅ Google Cloud Vision API: SUCCESS!")
            print(f"   Response: {json.dumps(result, indent=2)}")
            
            if "responses" in result and len(result["responses"]) > 0:
                response = result["responses"][0]
                if "labelAnnotations" in response:
                    labels = response["labelAnnotations"]
                    print(f"   Labels detected: {len(labels)}")
                    for i, label in enumerate(labels[:3], 1):
                        print(f"     {i}. {label['description']} (confidence: {label['score']:.2f})")
                else:
                    print("   No labels detected")
            
            return True
        else:
            print(f"❌ Google Cloud Vision API: FAILED")
            print(f"   Status Code: {vision_response.status_code}")
            print(f"   Response: {vision_response.text}")
            
            # Try to parse error details
            try:
                error_data = vision_response.json()
                if "error" in error_data:
                    error = error_data["error"]
                    print(f"   Error Code: {error.get('code', 'Unknown')}")
                    print(f"   Error Message: {error.get('message', 'Unknown')}")
                    print(f"   Error Status: {error.get('status', 'Unknown')}")
                    
                    if "details" in error:
                        for detail in error["details"]:
                            if "reason" in detail:
                                print(f"   Reason: {detail['reason']}")
                            if "domain" in detail:
                                print(f"   Domain: {detail['domain']}")
                            if "activationUrl" in detail:
                                print(f"   Activation URL: {detail['activationUrl']}")
            except:
                print("   Could not parse error details")
            
            return False
            
    except Exception as e:
        print(f"❌ Exception during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def check_billing_status():
    """Check if billing is enabled for the project"""
    print("\n💳 CHECKING BILLING STATUS")
    print("-" * 30)
    print("Google Cloud Vision API requires billing to be enabled.")
    print("To check billing status:")
    print("1. Visit: https://console.cloud.google.com/billing")
    print("2. Select your project: 105686176551")
    print("3. Ensure billing is enabled")
    print("4. Add a payment method if needed")

def check_api_quotas():
    """Check API quotas and limits"""
    print("\n📊 CHECKING API QUOTAS")
    print("-" * 25)
    print("To check API quotas:")
    print("1. Visit: https://console.cloud.google.com/apis/api/vision.googleapis.com/quotas")
    print("2. Select your project: 105686176551")
    print("3. Check if you have available quota")
    print("4. Request quota increase if needed")

if __name__ == "__main__":
    success = test_vision_api_detailed()
    
    if not success:
        print("\n🔧 TROUBLESHOOTING STEPS")
        print("=" * 30)
        check_billing_status()
        check_api_quotas()
        
        print("\n⏰ WAIT AND RETRY")
        print("-" * 20)
        print("If you just enabled the API:")
        print("1. Wait 5-10 minutes for activation to propagate")
        print("2. Re-run this test")
        print("3. Check the Google Cloud Console for any errors")
    else:
        print("\n🎉 SUCCESS! Google Cloud Vision API is working!")
        print("You can now run the full AI integration test.")

#!/usr/bin/env python3
"""
Heroku CLI Integration Setup
"""

import subprocess
import json
import os
import time

def run_heroku_command(command):
    """Run a Heroku CLI command"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout.strip()
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)

def check_heroku_cli():
    """Check if Heroku CLI is installed"""
    success, output = run_heroku_command("heroku --version")
    if success:
        print(f"✅ Heroku CLI: {output}")
        return True
    else:
        print("❌ Heroku CLI not found. Please install it first:")
        print("   https://devcenter.heroku.com/articles/heroku-cli")
        return False

def check_heroku_auth():
    """Check if user is logged into Heroku"""
    success, output = run_heroku_command("heroku auth:whoami")
    if success:
        print(f"✅ Heroku Auth: Logged in as {output}")
        return True
    else:
        print("❌ Not logged into Heroku. Please run:")
        print("   heroku login")
        return False

def list_heroku_apps():
    """List Heroku apps"""
    success, output = run_heroku_command("heroku apps")
    if success:
        print("📱 Heroku Apps:")
        print(output)
        return True
    else:
        print("❌ Failed to list apps")
        return False

def get_app_url(app_name):
    """Get the URL of a Heroku app"""
    success, output = run_heroku_command(f"heroku info --app {app_name}")
    if success:
        # Extract web URL from heroku info output
        lines = output.split('\n')
        for line in lines:
            if 'Web URL:' in line:
                url = line.split('Web URL:')[1].strip()
                return True, url
        return False, "URL not found"
    else:
        return False, output

def setup_heroku_integration():
    """Setup Heroku integration for geographic intelligence API"""
    
    print("🚀 Heroku CLI Integration Setup")
    print("=" * 40)
    
    # Step 1: Check Heroku CLI
    print("\n1. Checking Heroku CLI...")
    if not check_heroku_cli():
        return False
    
    # Step 2: Check authentication
    print("\n2. Checking Heroku authentication...")
    if not check_heroku_auth():
        return False
    
    # Step 3: List apps
    print("\n3. Listing Heroku apps...")
    if not list_heroku_apps():
        return False
    
    # Step 4: Get app URL
    print("\n4. Getting app URL...")
    app_name = input("Enter your Heroku app name: ").strip()
    
    if not app_name:
        print("❌ No app name provided")
        return False
    
    success, url = get_app_url(app_name)
    if success:
        print(f"✅ App URL: {url}")
        
        # Update the API configuration
        api_url = f"{url}/api/agents/swarm"
        print(f"✅ API Endpoint: {api_url}")
        
        # Create configuration file
        config = {
            "heroku_app_name": app_name,
            "heroku_url": url,
            "api_endpoint": api_url,
            "google_maps_api_key": "AIzaSyAUeiyRuSuKcnPBFmezWCFuUStl8unv7_0",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open("heroku_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("\n✅ Configuration saved to heroku_config.json")
        print("\n🚀 Next steps:")
        print("   1. Deploy your geographic intelligence app to Heroku")
        print("   2. Run: python3 test_live_google_integration.py")
        print("   3. Your API will be available at:", api_url)
        
        return True
    else:
        print(f"❌ Failed to get app URL: {url}")
        return False

def test_heroku_connection():
    """Test connection to Heroku app"""
    try:
        with open("heroku_config.json", "r") as f:
            config = json.load(f)
        
        api_url = config["api_endpoint"]
        
        print(f"🔌 Testing connection to: {api_url}")
        
        import requests
        response = requests.get(f"{api_url}?action=health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Connection successful!")
            print(f"   Service: {data.get('data', {}).get('service', 'Unknown')}")
            print(f"   Status: {data.get('data', {}).get('status', 'Unknown')}")
            return True
        else:
            print(f"❌ Connection failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Setup Heroku integration")
    print("2. Test Heroku connection")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        setup_heroku_integration()
    elif choice == "2":
        test_heroku_connection()
    else:
        print("Invalid choice")

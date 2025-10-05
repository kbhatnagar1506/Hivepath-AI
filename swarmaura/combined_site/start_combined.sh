#!/bin/bash

echo "🚀 STARTING COMBINED SITE"
echo "========================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python3."
    exit 1
fi

# Install requirements if needed
if [ ! -d "venv" ]; then
    echo "🔄 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔄 Activating virtual environment..."
source venv/bin/activate

echo "🔄 Installing requirements..."
pip install -r requirements.txt

echo "🚀 Starting combined site server..."
echo "🌐 Frontend: http://localhost:8080"
echo "📡 API: http://localhost:8080/api"
echo ""

python3 combined_server.py

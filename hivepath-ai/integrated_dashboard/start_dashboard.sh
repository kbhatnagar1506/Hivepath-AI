#!/bin/bash

echo "🚀 STARTING SWARMAURA INTEGRATED DASHBOARD"
echo "=========================================="
echo ""

# Check if API is running
echo "🔍 Checking API status..."
if curl -s http://localhost:8001/api/health > /dev/null; then
    echo "✅ API is running on http://localhost:8001"
else
    echo "❌ API is not running. Please start the API first:"
    echo "   cd /Users/krishnabhatnagar/hackharvard/swarmaura"
    echo "   python3 data_extraction_api.py"
    echo ""
    echo "Then run this script again."
    exit 1
fi

echo ""
echo "📦 Installing dependencies..."
if command -v pnpm &> /dev/null; then
    pnpm install
elif command -v npm &> /dev/null; then
    npm install
else
    echo "❌ Neither pnpm nor npm found. Please install Node.js first."
    exit 1
fi

echo ""
echo "🌐 Starting dashboard..."
echo "Dashboard will be available at: http://localhost:3000"
echo "API is running at: http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

# Set environment variable for API URL
export NEXT_PUBLIC_API_URL=http://localhost:8001

# Start the development server
if command -v pnpm &> /dev/null; then
    pnpm dev
else
    npm run dev
fi

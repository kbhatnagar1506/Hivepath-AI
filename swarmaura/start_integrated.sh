#!/bin/bash

echo "🚀 STARTING INTEGRATED FRONTEND-BACKEND SYSTEM"
echo "=============================================="

# Check if backend is running
if ! curl -s http://localhost:8000/api/v1/health > /dev/null; then
    echo "🔄 Starting backend..."
    cd backend
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
    BACKEND_PID=$!
    cd ..
    sleep 5
else
    echo "✅ Backend already running"
fi

# Check if frontend proxy is running
if ! curl -s http://localhost:8080 > /dev/null; then
    echo "🔄 Starting frontend proxy..."
    cd frontend-proxy
    npm install
    npm start &
    PROXY_PID=$!
    cd ..
    sleep 5
else
    echo "✅ Frontend proxy already running"
fi

echo ""
echo "🎯 INTEGRATED SYSTEM STATUS:"
echo "============================"
echo "Backend API: http://localhost:8000"
echo "Frontend: https://fleet-flow-7189cccb.base44.app"
echo "Integrated: http://localhost:8080"
echo ""
echo "📡 API Endpoints:"
echo "  - Health: http://localhost:8000/api/v1/health"
echo "  - Routes: http://localhost:8000/api/v1/optimize/routes"
echo "  - Multi-location: http://localhost:8000/api/v1/multi-location/routes"
echo "  - Integration: http://localhost:8000/api/v1/integration/status"
echo ""
echo "🌐 Frontend Integration:"
echo "  - Main App: https://fleet-flow-7189cccb.base44.app"
echo "  - Proxy: http://localhost:8080"
echo ""
echo "✅ Integrated system is running!"
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap 'echo "🛑 Stopping services..."; kill $BACKEND_PID $PROXY_PID 2>/dev/null; exit' INT
wait

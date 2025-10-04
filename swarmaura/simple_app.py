from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="SwarmAura API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "SwarmAura AI-Powered Routing System",
        "status": "online",
        "version": "1.0.0",
        "features": [
            "AI-powered routing optimization",
            "ML models for service time prediction", 
            "Risk assessment and accessibility evaluation",
            "Google Maps integration",
            "Real-time agent coordination"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "swarmaura"}

@app.get("/api/v1/status")
def api_status():
    return {
        "api": "operational",
        "redis": "connected" if os.getenv("REDIS_URL") else "not_configured",
        "environment": "production"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("simple_app:app", host="0.0.0.0", port=port)

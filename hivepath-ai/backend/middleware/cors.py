
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

def setup_cors(app: FastAPI):
    """Setup CORS middleware for frontend integration"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://fleet-flow-7189cccb.base44.app",
            "http://localhost:3000",
            "http://localhost:8080"
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    return app

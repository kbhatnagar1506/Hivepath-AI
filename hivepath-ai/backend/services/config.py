import os
SERVICE_NAME = os.getenv("SERVICE_NAME", "routeloom")
REDIS_URL    = os.getenv("REDIS_URL", "redis://localhost:6379/0")
LOG_LEVEL    = os.getenv("LOG_LEVEL", "INFO")
BASE_BACKEND = os.getenv("BASE_BACKEND_URL", "http://localhost:8000")
GOOGLE_KEY   = os.getenv("GOOGLE_MAPS_API_KEY", "")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY", "")
VLM_MODEL    = os.getenv("VLM_MODEL", "gpt-4o-mini")

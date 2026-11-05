"""Settings from .env. The service has no database settings and no cloud LLM fallback."""
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class Settings(BaseModel):
    backend_url: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8080")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    jwt_secret: str = os.getenv("JWT_SECRET", "")
    frontend_origin: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:4200")
    model_timeout_seconds: float = float(os.getenv("MODEL_TIMEOUT_SECONDS", "180"))
    max_applicants: int = int(os.getenv("MAX_APPLICANTS", "20"))
    max_campaigns: int = int(os.getenv("MAX_CAMPAIGNS", "8"))


settings = Settings()

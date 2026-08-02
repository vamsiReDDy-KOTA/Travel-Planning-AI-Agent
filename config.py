import os
from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Intelligent Travel Planning AI Agent"
    VERSION: str = "1.0.0"
    
    # Gemini API Config
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    # Qdrant Vector DB Config
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "http://localhost:6333")
    QDRANT_COLLECTION_PREFERENCES: str = "user_preferences"
    QDRANT_COLLECTION_RAG: str = "travel_guides"
    
    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    
    # Guardrail Limits
    MAX_BUDGET_THRESHOLD: float = 100000.0  # Max realistic budget USD
    MIN_DAYS: int = 1
    MAX_DAYS: int = 7
    DEFAULT_DAYS: int = 2

    model_config = ConfigDict(env_file=".env", extra="ignore")

settings = Settings()

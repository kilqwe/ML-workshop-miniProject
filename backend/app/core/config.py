from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    APP_NAME: str = "Football Analyzer API"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379"
    
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://ml-workshop-mini-project.vercel.app"
    ]

    model_config = {"env_file": ".env"}
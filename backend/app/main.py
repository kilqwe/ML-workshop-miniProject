from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from loguru import logger
import joblib

from app.core.config import settings
from app.api.v1.routes import predictions, auth, health
from app.services.prediction_service import load_and_preprocess

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML assets on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Loading ML pipeline assets...")
    try:
        app.state.pipeline = joblib.load("models/fifa_models.pkl")
        app.state.df, _, _ = load_and_preprocess("data/cleaned_fifa23.csv")
        logger.info("ML assets loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML assets: {e}")
        app.state.pipeline = None
        app.state.df = None

app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(health.router, prefix="/api", tags=["health"])
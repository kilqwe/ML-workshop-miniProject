from fastapi import APIRouter
from app.core.cache import redis_client

router = APIRouter()

@router.get("/health")
async def health_check():
    try:
        await redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unavailable"
    
    return {
        "status": "healthy",
        "redis": redis_status,
        "version": "2.0.0"
    }
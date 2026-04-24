import redis.asyncio as redis
from app.core.config import settings
import json

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)

async def get_cache(key: str):
    val = await redis_client.get(key)
    return json.loads(val) if val else None

async def set_cache(key: str, value: dict, ttl: int = 3600):
    await redis_client.setex(key, ttl, json.dumps(value))
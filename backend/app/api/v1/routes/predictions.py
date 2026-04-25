from fastapi import APIRouter, Request, Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session
import json

from app.core.cache import get_cache, set_cache
from app.db.session import get_db
from app.db.models import PredictionHistory
from app.services.prediction_service import predict_player, find_similar_players

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

class PlayerStats(BaseModel):
    pace_total: Optional[float] = Field(None, alias="Pace Total")
    shooting_total: Optional[float] = Field(None, alias="Shooting Total")
    passing_total: Optional[float] = Field(None, alias="Passing Total")
    dribbling_total: Optional[float] = Field(None, alias="Dribbling Total")
    defending_total: Optional[float] = Field(None, alias="Defending Total")
    physicality_total: Optional[float] = Field(None, alias="Physicality Total")
    heading_accuracy: Optional[float] = Field(None, alias="Heading Accuracy")
    jumping: Optional[float] = Field(None, alias="Jumping")
    long_passing: Optional[float] = Field(None, alias="LongPassing")
    goalkeeper_diving: Optional[float] = Field(None, alias="Goalkeeper Diving")
    goalkeeper_handling: Optional[float] = Field(None, alias="Goalkeeper Handling")
    goalkeeper_kicking: Optional[float] = Field(None, alias="Goalkeeper Kicking")
    goalkeeper_positioning: Optional[float] = Field(None, alias="Goalkeeper Positioning")
    goalkeeper_reflexes: Optional[float] = Field(None, alias="Goalkeeper Reflexes")

    class Config:
        populate_by_name = True

@router.post("/predictions")
@limiter.limit("10/minute")
async def predict(request: Request, stats: PlayerStats, db: Session = Depends(get_db)):
    pipeline = request.app.state.pipeline
    df = request.app.state.df

    if not pipeline or df is None:
        raise HTTPException(status_code=503, detail="Model assets not loaded")

    user_input = {k: v for k, v in stats.model_dump(by_alias=True).items() if v is not None}
    if not user_input:
        raise HTTPException(status_code=400, detail="No player stats provided")

    # Check Redis cache first
    cache_key = f"prediction:{json.dumps(user_input, sort_keys=True)}"
    cached = await get_cache(cache_key)
    if cached:
        return {**cached, "cached": True}

    # Run prediction
    predicted_group, predicted_rating, predicted_exact_position = predict_player(user_input, pipeline)

    similar_players_df = find_similar_players(
        df, user_input, predicted_group,
        pipeline["core_stats"], pipeline["gk_stats"], top_n=3
    )

    stat_columns = pipeline["gk_stats"] if predicted_group == "GK" else pipeline["core_stats"]
    response_columns = ["Full Name", "Overall", "Best Position", "Club Name"] + stat_columns
    similar_players = similar_players_df[response_columns].to_dict(orient="records")
    ideal_profile = pipeline["raw_centroids"].get(predicted_exact_position, None)

    result = {
        "predicted_rating": predicted_rating,
        "predicted_group": predicted_group,
        "predicted_exact_position": predicted_exact_position,
        "similar_players": similar_players,
        "ideal_profile": ideal_profile,
        "cached": False
    }

    # Save to cache and DB
    await set_cache(cache_key, result)
    db.add(PredictionHistory(
        input_stats=user_input,
        predicted_rating=predicted_rating,
        predicted_group=predicted_group,
        predicted_position=predicted_exact_position
    ))
    db.commit()

    return result

@router.get("/predictions/history")
async def get_history(db: Session = Depends(get_db)):
    history = db.query(PredictionHistory).order_by(
        PredictionHistory.created_at.desc()
    ).limit(20).all()
    return history

@router.get("/predictions/metrics")
async def get_model_metrics(request: Request):
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "regressor_metrics": pipeline.get("reg_metrics", {}),
        "description": {
            "r2": "R² score (1.0 = perfect, higher is better)",
            "mae": "Mean Absolute Error in rating points (lower is better)"
        }
    }
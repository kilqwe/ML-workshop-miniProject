from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
# --- MODIFIED IMPORTS ---
from models import predict_player, find_similar_players, load_and_preprocess

# 1. Initialize the FastAPI App
app = FastAPI(
    title="FIFA Player Rating API",
    description="An API to predict a player's rating, position, and find similar players.",
    version="1.1.0"
)
origins = [
    "http://localhost:3000",  # The default port for Next.js
    "https://ml-workshop-miniproject.onrender.com",
    "https://ml-workshop-mini-project.vercel.app"

    

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)
# 2. Load the Pipeline and the DataFrame
try:
    pipeline_path = "models/fifa_models.pkl"
    trained_pipeline = joblib.load(pipeline_path)
    
    # NEW: Load the dataframe needed for the similarity search
    df, _, _ = load_and_preprocess("data/cleaned_fifa23.csv") 
    
    print("✅ Pipeline and DataFrame loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    trained_pipeline = None
    df = None

# 3. Define the Input Data Model (no changes here)
class PlayerStats(BaseModel):
    pace_total: Optional[float] = Field(None, alias="Pace Total")
    # ... (include all the other stats as before) ...
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

# 4. Define the Prediction Endpoint (MODIFIED)
@app.post("/predict")
async def predict(stats: PlayerStats):
    if not trained_pipeline or df is None:
        raise HTTPException(status_code=503, detail="Model assets are not loaded.")

    try:
        user_input = {k: v for k, v in stats.model_dump(by_alias=True).items() if v is not None}
        if not user_input:
             raise HTTPException(status_code=400, detail="No player stats provided.")


        # Step 1: Get the main prediction
        predicted_group, predicted_rating, predicted_exact_position = predict_player(user_input, trained_pipeline)
        
        # Step 2: Find similar players
        similar_players_df = find_similar_players(
            df,
            user_input,
            predicted_group,
            trained_pipeline["core_stats"],
            trained_pipeline["gk_stats"],
            top_n=3
    )
        ideal_profile = trained_pipeline["raw_centroids"].get(predicted_exact_position, None)
        if predicted_group == "GK":
            stat_columns = trained_pipeline["gk_stats"]
        else:
            stat_columns = trained_pipeline["core_stats"]
        # Step 3: Format the similar players into a clean list of dictionaries
        response_columns = ["Full Name", "Overall", "Best Position", "Club Name"] + stat_columns
        similar_players = similar_players_df[response_columns].to_dict(orient="records")

        # Step 4: Return the combined result
        return {
            "predicted_rating": predicted_rating,
            "predicted_group": predicted_group,
            "predicted_exact_position": predicted_exact_position, 
            "similar_players": similar_players,
            "ideal_profile": ideal_profile,
        }
 

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "FIFA Player Rating API is running."}
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained models and columns
reg = joblib.load("reg_model.pkl")
clf_exact = joblib.load("clf_exact.pkl")
xgb = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
le_exact = joblib.load("le_exact.pkl")
le_group = joblib.load("le_group.pkl")
training_columns = joblib.load("training_columns.pkl")

df = pd.read_csv("data/cleaned_fifa23.csv")

class Features(BaseModel):
    features: dict

@app.post("/predict")
async def predict(data: Features):
    X = pd.DataFrame([data.features])
    X = X[training_columns]
    X_scaled = scaler.transform(X)

    predicted_ovr = float(reg.predict(X_scaled)[0])
    predicted_pos_exact = le_exact.inverse_transform(clf_exact.predict(X_scaled))[0]
    predicted_pos_group = le_group.inverse_transform(xgb.predict(X_scaled))[0]

    # --- THIS IS THE CHANGE ---
    # 1. Calculate the difference in Overall rating
    df["ovr_diff"] = (df["Overall"] - predicted_ovr).abs()
    
    # 2. Sort by the difference, get the top 3, select their names, and convert to a list
    closest_players = df.sort_values("ovr_diff").head(3)["Full Name"].tolist()
    # ------------------------

    return {
        "predicted_ovr": round(predicted_ovr, 1),
        "predicted_position_exact": predicted_pos_exact,
        "predicted_position_group": predicted_pos_group,
        "closest_players": closest_players # Return the list of 3 players
    }

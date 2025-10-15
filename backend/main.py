import joblib
from models import train_full_pipeline, predict_player

# ==================================================
# Main Training and Saving Block
# ==================================================
if __name__ == "__main__":
    # 1. DEFINE DATA PATH
    # --- IMPORTANT: REPLACE WITH THE ACTUAL PATH TO YOUR CSV FILE ---
    DATA_PATH = "data/cleaned_fifa23.csv"
    
    # 2. RUN THE ENTIRE TRAINING PIPELINE
    print("🚀 Starting model training pipeline...")
    df, trained_pipeline = train_full_pipeline(DATA_PATH)
    print("\n✅ Pipeline training complete!")

    # 3. SAVE THE PIPELINE ARTIFACTS
    save_path = "models/fifa_models.pkl"
    joblib.dump(trained_pipeline, save_path)
    print(f"✅ All models, scalers, and encoders saved to {save_path}")

    # 4. RUN A DEMONSTRATION
    print("\n--- Running a demonstration ---")
    loaded_pipeline = joblib.load(save_path)
    print("   - Successfully loaded saved pipeline.")

    # --- Demo 1: Player with only a few key stats ---
    player_with_partial_stats = {
        "Shooting Total": 98,
        "Dribbling Total": 95,
        "Pace Total": 92
    }
    
    predicted_group, predicted_rating,predicted_exact_position = predict_player(player_with_partial_stats, loaded_pipeline)
    
    print("\n--- 📈 Demo Prediction Result (Partial Stats) ---")
    print(f"Predicted Position Group: {predicted_group}")
    print(f"Predicted Overall Rating: {predicted_rating}")
    print(f"Predicted Exact Position: {predicted_exact_position}")

    # --- Demo 2: Player with full stats ---
    super_player_stats = {
        "Pace Total": 95, "Shooting Total": 95, "Passing Total": 90,
        "Dribbling Total": 98, "Defending Total": 45, "Physicality Total": 85,
        "Goalkeeper Diving": 15, "Goalkeeper Handling": 12, "Goalkeeper Kicking": 10,
        "Goalkeeper Positioning": 11, "Goalkeeper Reflexes": 13
    }
    
    group_full, rating_full, exact_posi = predict_player(super_player_stats, loaded_pipeline)
    
    print("\n--- 📈 Demo Prediction Result (Full Stats) ---")
    print(f"Predicted Position Group: {group_full}")
    print(f"Predicted Overall Rating: {rating_full}")
    print(f"Predicted Exact Position: {exact_posi}")
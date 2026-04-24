import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1) Load saved models
# ----------------------------
models_dict = joblib.load("models/fifa_models.pkl")

reg_models = models_dict["reg_models"]
gk_classifier = models_dict["gk_classifier"]
xgb_classifier = models_dict["xgb_classifier"]
training_columns = models_dict["training_columns"]
label_encoders = models_dict["label_encoders"]
centroids = models_dict["centroids"]
core_stats = models_dict["core_stats"]
gk_stats = models_dict["gk_stats"]
raw_means = models_dict["raw_means"]
core_scaler = models_dict["core_scaler"]
gk_scaler = models_dict["gk_scaler"]

le_binary = label_encoders["binary"]
le_group = label_encoders["group"]

# ----------------------------
# Normalize columns & features
# ----------------------------
training_columns = [col.strip() for col in training_columns]
core_stats = [col.strip() for col in core_stats]
gk_stats = [col.strip() for col in gk_stats]
raw_means = {k.strip(): v for k, v in raw_means.items()}

# Build a mapping for fast + safe index lookups
col_index_map = {col: i for i, col in enumerate(training_columns)}


# ----------------------------
# 2) Build full player vector
# ----------------------------
def build_full_vector(player_input):
    """
    Build a full-length feature vector using training_columns,
    filling missing values from raw_means.
    """
    return np.array([
        player_input.get(col, raw_means.get(col, 0))
        for col in training_columns
    ]).reshape(1, -1)


# ----------------------------
# 3) Assign exact position via centroid
# ----------------------------
def assign_position_centroid(player_vec, pos_group):
    group_map = {
        "DEF": ["CB", "LB", "RB", "LWB", "RWB"],
        "MID": ["CDM", "CM", "CAM", "LM", "RM"],
        "FWD": ["ST", "CF", "LW", "RW"],
        "GK": ["GK"]
    }
    candidates = [p for p in group_map.get(pos_group, []) if p in centroids]
    if not candidates:
        return list(centroids.keys())[0]

    dists = {}
    for pos in candidates:
        features = gk_stats if pos == "GK" else core_stats
        try:
            idx = [col_index_map[f] for f in features]
        except KeyError as e:
            raise ValueError(f"Feature {e} not found in training_columns")
        player_sub = player_vec[0, idx]
        centroid_sub = centroids[pos]
        dists[pos] = np.linalg.norm(player_sub - centroid_sub)
    return min(dists, key=dists.get)


# ----------------------------
# 4) Radar chart
# ----------------------------
def load_radar_chart(position, input_stats, core_stats, gk_stats):
    df = pd.read_csv("data/cleaned_fifa23.csv")
    df.columns = df.columns.str.strip()  # <-- strip spaces in CSV too

    features = gk_stats if position == "GK" else core_stats

    centroid_df = df[df["Best Position"] == position]
    if centroid_df.empty:
        raise ValueError(f"No players found for position '{position}'")

    centroid_values = centroid_df[features].mean().values
    input_values = np.array([input_stats.get(f, centroid_values[i]) for i, f in enumerate(features)])

    centroid_values = np.concatenate([centroid_values, [centroid_values[0]]])
    input_values = np.concatenate([input_values, [input_values[0]]])

    angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, centroid_values, color='blue', linewidth=2, label=f"{position} Avg")
    ax.fill(angles, centroid_values, color='blue', alpha=0.25)
    ax.plot(angles, input_values, color='red', linewidth=2, label="Player Input")
    ax.fill(angles, input_values, color='red', alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f"{position} Radar Chart", fontsize=14)
    plt.legend(loc='upper right')
    plt.show()



# ----------------------------
# 5) Predict single player
# ----------------------------
def predict_single_player(player_input):
    full_vector = build_full_vector(player_input)

    # GK detection: use GK classifier with GK stats subset
    gk_idx = [col_index_map[f] for f in gk_stats]
    gk_input = full_vector[:, gk_idx]
    gk_pred = gk_classifier.predict(gk_input)[0]
    is_gk = (gk_pred == le_binary.transform(["GK"])[0])

    # Group classifier: uses only core stats
    if is_gk:
        group_pred_label = "GK"
    else:
        group_idx = [col_index_map[f] for f in core_stats]
        group_input = full_vector[:, group_idx]
        group_pred = xgb_classifier.predict(group_input)[0]
        group_pred_label = le_group.inverse_transform([group_pred])[0]

    # Assign exact position using centroid distance
    exact_pos = assign_position_centroid(full_vector, group_pred_label)

    # Scale only relevant features
    if exact_pos == "GK":
        reg_idx = [col_index_map[f] for f in gk_stats]
        reg_vector = gk_scaler.transform(full_vector[:, reg_idx])
    else:
        reg_idx = [col_index_map[f] for f in core_stats]
        reg_vector = core_scaler.transform(full_vector[:, reg_idx])

    # Predict OVR using regressor for group label
    predicted_ovr = reg_models[group_pred_label].predict(reg_vector)[0]

    # Return summary
    return {
        "OVR": round(predicted_ovr, 1),
        "is_GK": is_gk,
        "Group": group_pred_label,
        "Exact": exact_pos
    }


# ----------------------------
# 6) Batch prediction function
# ----------------------------
def predict_players(player_inputs):
    results = []
    for p in player_inputs:
        result = predict_single_player(p)
        results.append(result)
        print("\n--- Player Prediction ---")
        print("Predicted OVR:", result["OVR"])
        print("Is GK:", result["is_GK"])
        print("Group Position:", result["Group"])
        print("Exact Position:", result["Exact"])

        # Radar Chart
        load_radar_chart(result["Exact"], p, core_stats, gk_stats)
    return results


# ----------------------------
# 7) Example usage
# ----------------------------
players = [
    {
        "Pace Total": 90,
        "Shooting Total": 85,
        "Passing Total": 75,
        "Dribbling Total": 80,
        "Defending Total": 40,
        "Physicality Total": 70
    },
    {
        "Goalkeeper Diving": 88,
        "Goalkeeper Handling": 85,
        "Goalkeeper Kicking": 78,
        "Goalkeeper Positioning": 90,
        "Goalkeeper Reflexes": 86
    }
]

if __name__ == "__main__":
    results = predict_players(players)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score
from sklearn.metrics.pairwise import euclidean_distances
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# ==================================================
# 1) Helper Functions
# ==================================================
def simplify_position(pos: str) -> str:
    """Groups detailed positions into broader categories."""
    if pos in ["CB", "LB", "RB", "RWB", "LWB"]:
        return "DEF"
    elif pos in ["CM", "CDM", "CAM", "LM", "RM"]:
        return "MID"
    elif pos in ["ST", "CF", "LW", "RW"]:
        return "FWD"
    elif pos == "GK":
        return "GK"
    return "OTHER"

def simplify_gk(pos: str) -> str:
    """Identifies a position as either 'GK' or 'OUTFIELD'."""
    if pos == "GK":
        return "GK"
    else:
        return "OUTFIELD"

# ==================================================
# 2) Load & Preprocess Data
# ==================================================
def load_and_preprocess(path: str):
    """Loads CSV, cleans column names, and defines feature sets."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    core_stats = [
        "Pace Total", "Shooting Total", "Passing Total", "Dribbling Total",
        "Defending Total", "Physicality Total"
    ]
    gk_stats = [
        "Goalkeeper Diving", "Goalkeeper Handling", "Goalkeeper Kicking",
        "Goalkeeper Positioning", "Goalkeeper Reflexes"
    ]

    df["Group"] = df["Best Position"].apply(simplify_position)
    return df, core_stats, gk_stats

# ==================================================
# 3) Training Functions (with Scaling)
# ==================================================
def train_gk_classifier(df, gk_stats):
    """Trains a classifier to distinguish GKs from Outfield players."""
    X = df[gk_stats].values
    y = df["Best Position"].apply(simplify_gk).values
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test_scaled))
    print(f"GK Classifier Accuracy: {acc:.3f}")
    return clf, le, scaler

def train_grouped_position_classifier(df, core_stats):
    """Trains a classifier to predict FWD, MID, or DEF for outfield players."""
    outfield_df = df[df["Group"] != "GK"]
    X = outfield_df[core_stats].values
    y = outfield_df["Group"].values
    
    le_group = LabelEncoder()
    y_enc = le_group.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                        random_state=42, eval_metric="mlogloss", use_label_encoder=False)
    xgb.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, xgb.predict(X_test_scaled))
    print(f"Grouped Position Accuracy: {acc:.3f}")
    return xgb, le_group, scaler

def train_position_specific_regressors(df, core_stats, gk_stats):
    """Trains a separate Overall rating predictor for each position group."""
    reg_models = {}
    scalers = {}
    feature_map = {"FWD": core_stats, "MID": core_stats, "DEF": core_stats, "GK": gk_stats}
    
    print("\n--- Training Position-Specific Regressors ---")
    for group, feats in feature_map.items():
        subset = df[df["Group"] == group]
        if subset.empty: continue
            
        X = subset[feats].values
        y = subset["Overall"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        reg = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        reg.fit(X_train_scaled, y_train)
        
        y_pred = reg.predict(X_test_scaled)
        print(f"  - {group} Regressor → R²: {r2_score(y_test, y_pred):.3f}, MAE: {mean_absolute_error(y_test, y_pred):.2f}")
              
        reg_models[group] = reg
        scalers[group] = scaler
        
    return reg_models, scalers

# ==================================================
# 4) Similarity and Centroid Functions
# ==================================================
def build_position_centroids(df, core_stats, gk_stats):
    scaler = StandardScaler()
    scaled_core_df = pd.DataFrame(scaler.fit_transform(df[core_stats]), columns=core_stats)
    centroids = {}
    for pos in df["Best Position"].unique():
        if pos == "GK":
            centroids[pos] = df[df["Best Position"] == pos][gk_stats].mean().to_dict()
        else:
            indices = df.index[df["Best Position"] == pos].tolist()
            centroids[pos] = scaled_core_df.loc[indices].mean().to_dict()
    return centroids, scaler

def find_similar_players(df, user_input: dict, position_group: str,
                         core_stats, gk_stats, top_n=5):
    """Finds real players with the most similar stats to the user's input."""
    features = gk_stats if position_group == "GK" else core_stats
    
    columns_to_select = ["Full Name", "Overall", "Best Position", "Club Name"] + features
    df_pos = df[df["Group"] == position_group][columns_to_select].dropna()
    
    if df_pos.empty:
        return pd.DataFrame(columns=columns_to_select)

    input_vector = np.array([user_input.get(f, df_pos[f].mean()) for f in features]).reshape(1, -1)
    player_vectors = df_pos[features].values
    
    distances = euclidean_distances(input_vector, player_vectors)[0]
    df_pos["Similarity"] = distances
    
    return df_pos.sort_values("Similarity", ascending=True).head(top_n)

# ==================================================
# 5) Master Pipeline Functions
# ==================================================
def train_full_pipeline(path: str):
    """Loads data and trains all models, returning all artifacts."""
    df, core_stats, gk_stats = load_and_preprocess(path)
    
    all_stats = core_stats + gk_stats
    default_stats = df[all_stats].mean().to_dict()
    
    raw_centroids = build_raw_centroids(df, core_stats, gk_stats)
    print("--- Training Classifiers ---")
    gk_clf, gk_le, gk_scaler = train_gk_classifier(df, gk_stats)
    group_clf, group_le, core_scaler = train_grouped_position_classifier(df, core_stats)
    reg_models, reg_scalers = train_position_specific_regressors(df, core_stats, gk_stats)
    exact_pos_models, exact_pos_les = train_exact_position_classifiers(df, core_stats)
    pipeline_artifacts = {
        "gk_classifier": gk_clf, "gk_label_encoder": gk_le, "gk_scaler": gk_scaler,
        "group_classifier": group_clf, "group_label_encoder": group_le, "core_scaler": core_scaler,
        "regressors": reg_models, "reg_scalers": reg_scalers,
        "core_stats": core_stats, "gk_stats": gk_stats,
        "default_stats": default_stats,
        "exact_pos_models": exact_pos_models,
        "exact_pos_label_encoders": exact_pos_les,
        "raw_centroids": raw_centroids,
    }
    return df, pipeline_artifacts

# In models.py, add this new function

def train_exact_position_classifiers(df, core_stats):
    """Trains a separate classifier for each group (FWD, MID, DEF) to predict the exact position."""
    pos_models = {}
    pos_label_encoders = {}
    
    print("\n--- Training Exact Position Classifiers ---")
    for group in ["FWD", "MID", "DEF"]:
        subset = df[df["Group"] == group]
        if subset.empty or len(subset["Best Position"].unique()) < 2:
            continue
            
        X = subset[core_stats].values
        y = subset["Best Position"].values
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # A simple RandomForest is good for this multi-class task
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_scaled, y_train)
        
        acc = accuracy_score(y_test, clf.predict(X_test_scaled))
        print(f"  - {group} Exact Position Classifier Accuracy: {acc:.3f}")
        
        pos_models[group] = {"model": clf, "scaler": scaler}
        pos_label_encoders[group] = le
        
    return pos_models, pos_label_encoders



def build_raw_centroids(df, core_stats, gk_stats):
    """Calculates the raw, unscaled average stats for each exact position."""
    centroids = {}
    # Ensure 'Best Position' is a column that exists
    if "Best Position" not in df.columns:
        return centroids

    for position in df["Best Position"].unique():
        subset = df[df["Best Position"] == position]
        if subset.empty:
            continue

        # Determine which stats to average based on the position group
        group = subset["Group"].iloc[0]
        if group == "GK":
            centroids[position] = subset[gk_stats].mean().to_dict()
        else:
            centroids[position] = subset[core_stats].mean().to_dict()
    return centroids

def predict_player(user_input: dict, pipeline: dict):
    """Predicts group, rating, and exact position."""
    complete_input = pipeline["default_stats"].copy()
    complete_input.update(user_input)
    input_df = pd.DataFrame([complete_input])
    
    # Unpack models for Step 1
    gk_clf, gk_le, gk_scaler = pipeline["gk_classifier"], pipeline["gk_label_encoder"], pipeline["gk_scaler"]
    
    # Step 1: Predict GK or Outfield
    input_gk_scaled = gk_scaler.transform(input_df[pipeline["gk_stats"]])
    is_gk_pred = gk_clf.predict(input_gk_scaled)
    player_type = gk_le.inverse_transform(is_gk_pred)[0]

    # Step 2: Predict Position Group
    if player_type == "OUTFIELD":
        group_clf, group_le, core_scaler = pipeline["group_classifier"], pipeline["group_label_encoder"], pipeline["core_scaler"]
        input_core_scaled = core_scaler.transform(input_df[pipeline["core_stats"]])
        group_pred = group_clf.predict(input_core_scaled)
        position_group = group_le.inverse_transform(group_pred)[0]
    # --- ADD THIS ELSE BLOCK ---
    else:
        position_group = "GK"
    # ---------------------------

    # Step 3: Predict the EXACT position
    predicted_exact_position = position_group
    if position_group in ["FWD", "MID", "DEF"]:
        pos_model_pack = pipeline["exact_pos_models"][position_group]
        pos_model = pos_model_pack["model"]
        pos_scaler = pos_model_pack["scaler"]
        pos_le = pipeline["exact_pos_label_encoders"][position_group]
        
        input_scaled = pos_scaler.transform(input_df[pipeline["core_stats"]])
        exact_pos_pred_encoded = pos_model.predict(input_scaled)
        predicted_exact_position = pos_le.inverse_transform(exact_pos_pred_encoded)[0]

    # Step 4: Predict Overall Rating
    regressor = pipeline["regressors"][position_group]
    reg_scaler = pipeline["reg_scalers"][position_group]
    
    features = pipeline["gk_stats"] if position_group == "GK" else pipeline["core_stats"]
    input_reg_scaled = reg_scaler.transform(input_df[features])
    overall_prediction = regressor.predict(input_reg_scaled)
    
    return position_group, round(overall_prediction[0]), predicted_exact_position
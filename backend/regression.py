import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# --- 1. Load and Preprocess Data ---
print("1. Loading and preprocessing data...")
df = pd.read_csv('data/cleaned_fifa23.csv')

player_names = df["Full Name"]
actual_ovr = df["Overall"]
actual_pos = df["Best Position"]

drop_cols = ["Full Name","Overall", "Best Position", "Club Name"]
features = df.drop(columns=[c for c in drop_cols if c in df.columns])

# ✅ ADD THIS LINE to capture the exact list of columns used for training
training_columns = features.columns.tolist() 
print(training_columns)

scaler = StandardScaler()
X = scaler.fit_transform(features)
print("Data loaded and preprocessed.")


def train_ovr_regressor(X, y_ovr):
    """Trains a RandomForestRegressor to predict the 'Overall' rating."""
    print("\n2. Training OVR Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_ovr, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(random_state=42, n_estimators=50)
    reg.fit(X_train, y_train)
    
    y_pred_ovr = reg.predict(X_test)
    print(f"  - Regression R²: {r2_score(y_test, y_pred_ovr):.4f}")
    print(f"  - Regression MAE: {mean_absolute_error(y_test, y_pred_ovr):.4f}")
    print("Regressor training complete.")
    return reg


def train_gk_classifier(X, df):
    """Trains a binary classifier to distinguish Goalkeepers from other players."""
    print("\n3. Training GK vs. Not-GK Classifier...")
    le_binary = LabelEncoder()
    y_gk = le_binary.fit_transform(df["Best Position"].apply(lambda x: "GK" if x == "GK" else "NOT_GK"))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_gk, test_size=0.2, random_state=42)
    gk_clf = RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced")
    gk_clf.fit(X_train, y_train)
    
    y_pred_gk = gk_clf.predict(X_test)
    print(f"  - GK Classifier Accuracy: {accuracy_score(y_test, y_pred_gk):.4f}")
    print("GK classifier training complete.")
    return gk_clf, le_binary


def train_specific_position_classifier(X, df, position_group, group_name):
    """Trains a classifier for a specific group of positions (DEF, MID, FWD)."""
    print(f"\n4. Training {group_name} Position Classifier...")
    mask = df["Best Position"].isin(position_group)
    
    if not mask.any():
        print(f"  - No players found for group {group_name}. Skipping.")
        return None, None, mask

    le = LabelEncoder()
    y = le.fit_transform(df.loc[mask, "Best Position"])
    X_group = X[mask]

    X_train, X_test, y_train, y_test = train_test_split(X_group, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(random_state=42, n_estimators=50, class_weight="balanced")
    clf.fit(X_train, y_train)
    
    print(f"  - {group_name} Classifier Accuracy: {accuracy_score(y_test, clf.predict(X_test)):.4f}")
    print(f"{group_name} classifier training complete.")
    return clf, le, mask


def train_grouped_position_classifier(X, df):
    """Trains an XGBoost classifier for general position groups (DEF, MID, FWD, GK)."""
    print("\n5. Training Grouped Position Classifier (XGBoost)...")
    
    def simplify_position(pos):
        if pos in ["CB", "LB", "RB", "RWB", "LWB"]: return "DEF"
        elif pos in ["CM", "CDM", "CAM", "LM", "RM"]: return "MID"
        elif pos in ["ST", "CF", "LW", "RW"]: return "FWD"
        elif pos == "GK": return "GK"
        return "OTHER"

    pos_group = df["Best Position"].apply(simplify_position)
    le_group = LabelEncoder()
    y_pos_group = le_group.fit_transform(pos_group)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_pos_group, test_size=0.2, random_state=42)
    xgb = XGBClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=8, 
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        use_label_encoder=False, eval_metric='mlogloss'
    )
    xgb.fit(X_train, y_train)
    
    y_pred_pos_group = xgb.predict(X_test)
    print(f"  - Grouped Position Accuracy: {accuracy_score(y_test, y_pred_pos_group):.4f}")
    print("XGBoost training complete.")
    return xgb, le_group, pos_group

# The generate_final_predictions and display_evaluation_metrics functions are not
# strictly needed to just train and save, but we can leave them for evaluation purposes.

# --- SCRIPT EXECUTION ---

# 2. Train Models
reg_model = train_ovr_regressor(X, actual_ovr)
gk_classifier, le_binary = train_gk_classifier(X, df)
def_classifier, le_def, def_mask = train_specific_position_classifier(X, df, ["CB", "LB", "RB", "LWB", "RWB"], "Defender")
mid_classifier, le_mid, mid_mask = train_specific_position_classifier(X, df, ["CM", "CDM", "CAM", "LM", "RM"], "Midfielder")
fwd_classifier, le_fwd, fwd_mask = train_specific_position_classifier(X, df, ["ST", "CF", "LW", "RW"], "Forward")
xgb_classifier, le_group, pos_group = train_grouped_position_classifier(X, df)

# ... (Prediction and evaluation calls can be commented out if you only want to train) ...

# 6. Save models and encoders for future use
print("\n8. Saving models...")
joblib.dump(reg_model, 'reg_model.joblib')
joblib.dump(gk_classifier, 'gk_classifier.joblib')
joblib.dump(def_classifier, 'def_classifier.joblib')
joblib.dump(mid_classifier, 'mid_classifier.joblib')
joblib.dump(fwd_classifier, 'fwd_classifier.joblib')
joblib.dump(xgb_classifier, 'xgb_classifier.joblib')
joblib.dump(scaler, 'scaler.joblib')

all_encoders = {
    'le_binary': le_binary, 'le_def': le_def, 'le_mid': le_mid, 
    'le_fwd': le_fwd, 'le_group': le_group
}
joblib.dump(all_encoders, 'all_encoders.joblib')

# ✅ ADD THIS LINE to save the column list file
joblib.dump(training_columns, 'training_columns.joblib')

print("Models, encoders, and columns saved successfully.")
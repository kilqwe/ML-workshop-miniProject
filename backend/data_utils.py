import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(path: str):
    """
    Loads FIFA dataset and preprocesses features.
    Returns:
        - df: raw dataframe
        - X: scaled feature matrix
        - _: placeholder for GK split (deprecated)
        - scaler: fitted MinMaxScaler
        - _: placeholder for GK scaler (deprecated)
        - training_columns: list of feature column names
        - _: placeholder for GK columns
        - player_names: Series of player names
        - actual_ovr: Series of overall ratings
        - actual_pos: Series of actual positions
        - raw_means: dict of raw column means
        - _: placeholder for GK means
    """
    print("1. Loading and preprocessing data...")

    # Load CSV
    df = pd.read_csv(path)
    print("Columns in CSV:", df.columns.tolist())

    # Extract key info
    player_names = df["Full Name"]
    actual_ovr = df["Overall"]
    actual_pos = df["Best Position"]

    # Drop non-feature columns
    drop_cols = ["Full Name", "Overall", "Best Position", "Club Name"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    features = df[feature_cols].copy()

    # Fit scaler
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(features)
    X = scaler.transform(features)

    # Compute raw means for imputing missing stats later
    raw_means = features.mean().to_dict()

    print("Data loaded and preprocessed successfully.")

    return (
        df,           # full dataframe
        X,            # scaled features
        None,         # placeholder for X_gk
        scaler,       # fitted scaler
        None,         # placeholder for scaler_gk
        feature_cols, # feature column names
        None,         # placeholder for gk_cols
        player_names, # player names
        actual_ovr,   # overall ratings
        actual_pos,   # positions
        raw_means,    # raw means
        None          # placeholder for gk_means
    )

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from xgboost import XGBClassifier

# Read the original full dataset
df = pd.read_csv('data/cleaned_fifa23.csv')


def run_full_feature_clustering(dataframe):
    """
    This function performs clustering and finds similar players based on ALL player attributes.
    It is self-contained and does not interfere with the main 6-feature model.
    """
    print("--- Running Full-Feature Clustering and Recommendation Analysis ---")
    
    # 1. Select features for clustering (all detailed stats)
    cluster_df = dataframe.copy()
    features_for_clustering = cluster_df.drop(columns=[
        "Full Name", "Overall", "Best Position", "Club Name", "pos_group"
    ], errors="ignore")

    # Use a new, separate scaler for this task
    clustering_scaler = StandardScaler()
    X_scaled_cluster = clustering_scaler.fit_transform(features_for_clustering)

    # 2. Perform KMeans Clustering
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(X_scaled_cluster)
    print("\nClustering complete.")

    # 3. Define the recommendation function within this scope
    def recommend_similar_players(player_name, n=5):
        if player_name not in cluster_df["Full Name"].values:
            return f"{player_name} not found in dataset"
        
        player_row_index = cluster_df[cluster_df["Full Name"] == player_name].index[0]
        player_cluster = cluster_df.loc[player_row_index, "cluster"]
        player_vector = X_scaled_cluster[player_row_index]

        # Filter for players in the same cluster
        same_cluster_df = cluster_df[cluster_df["cluster"] == player_cluster].copy()
        same_cluster_indices = same_cluster_df.index
        X_same_cluster = X_scaled_cluster[same_cluster_indices]
        
        # Calculate distances
        distances = euclidean_distances([player_vector], X_same_cluster)[0]
        same_cluster_df["distance"] = distances
        
        recommendations = same_cluster_df.sort_values("distance").head(n + 1)
        return recommendations[["Full Name", "Overall", "Best Position", "distance"]].iloc[1:]

    # 4. Run an example
    print("\nExample: Similar players to 'Heung Min Son':")
    print(recommend_similar_players("Heung Min Son", n=5))
    
    # 5. Save the clustering-specific models
    joblib.dump(kmeans, "kmeans_model.pkl")
    joblib.dump(clustering_scaler, "clustering_scaler.pkl")
    print("\nClustering models saved as kmeans_model.pkl and clustering_scaler.pkl")


##  Main Script for Regression and classification predictions##


player_names = df["Full Name"]
actual_ovr = df["Overall"]
actual_pos = df["Best Position"]


feature_cols = [
    "Pace Total", "Shooting Total", "Passing Total",
    "Dribbling Total", "Defending Total", "Physicality Total"
]
features = df[feature_cols]
training_columns = features.columns.tolist()

# Scale the 6 features
scaler = StandardScaler()
X = scaler.fit_transform(features)


print("\n Training Regression and classification model")


y_ovr = actual_ovr
X_train_ovr, _, y_train_ovr, _ = train_test_split(X, y_ovr, test_size=0.2, random_state=42)
reg = RandomForestRegressor(random_state=42, n_estimators=100)
reg.fit(X_train_ovr, y_train_ovr)
print("Regression model trained.")

le_exact = LabelEncoder()
y_pos_exact = le_exact.fit_transform(actual_pos)
X_train_pos, _, y_train_pos, _ = train_test_split(X, y_pos_exact, test_size=0.2, random_state=42)
clf_exact = RandomForestClassifier(random_state=42, n_estimators=100)
clf_exact.fit(X_train_pos, y_train_pos)
print("Exact Classification position model trained.")


def simplify_position(pos):
    if pos in ["CB", "LB", "RB", "RWB", "LWB"]: return "DEF"
    elif pos in ["CM", "CDM", "CAM", "LM", "RM"]: return "MID"
    elif pos in ["ST", "CF", "LW", "RW"]: return "FWD"
    elif pos == "GK": return "GK"
    else: return "OTHER"

df["pos_group"] = df["Best Position"].apply(simplify_position)
le_group = LabelEncoder()
y_pos_group = le_group.fit_transform(df["pos_group"])
X_train_grp, _, y_train_grp, _ = train_test_split(X, y_pos_group, test_size=0.2, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train_grp, y_train_grp)
print("Grouped position model trained.(regression)")

# 4. SAVE THE 6-FEATURE MODELS
joblib.dump(reg, "reg_model.pkl")
joblib.dump(clf_exact, "clf_exact.pkl")
joblib.dump(xgb, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_exact, "le_exact.pkl")
joblib.dump(le_group, "le_group.pkl")
joblib.dump(training_columns, "training_columns.pkl")

print("\nAll models saved successfully!")

#clustering implementation
# run_full_feature_clustering(df)

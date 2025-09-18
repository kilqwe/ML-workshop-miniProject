import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('data/cleaned_fifa23.csv')

# %% [markdown]
# ### Feature Selection

# %%
player_names = df["Full Name"]
actual_ovr = df["Overall"]
actual_pos = df["Best Position"]

#features 
drop_cols = ["Full Name", "Overall", "Best Position", "Club Name"]
features = df.drop(columns=[c for c in drop_cols if c in df.columns])

#scale numeric features (optional but good for ML)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# %% [markdown]
# ### Regression for OVR and Position

# %%
y_ovr = actual_ovr

X_train, X_test, y_train, y_test = train_test_split(X, y_ovr, test_size=0.2, random_state=42)

reg = RandomForestRegressor(random_state=42, n_estimators=200)
reg.fit(X_train, y_train)

y_pred_ovr = reg.predict(X_test)

print("Regression R²:", r2_score(y_test, y_pred_ovr))
print("Regression MAE:", mean_absolute_error(y_test, y_pred_ovr))

# %% [markdown]
# ### Classification for Position

# %%
le_exact = LabelEncoder()
y_pos_exact = le_exact.fit_transform(actual_pos)

X_train, X_test, y_train, y_test = train_test_split(X, y_pos_exact, test_size=0.2, random_state=42)

clf_exact = RandomForestClassifier(random_state=42, n_estimators=300, class_weight="balanced")
clf_exact.fit(X_train, y_train)

y_pred_pos_exact = clf_exact.predict(X_test)

print("\nExact Position Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred_pos_exact))
print("Report:\n", classification_report(y_test, y_pred_pos_exact, target_names=le_exact.classes_))

cm = confusion_matrix(y_test, y_pred_pos_exact)
# sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le_exact.classes_, yticklabels=le_exact.classes_)
# plt.title("Confusion Matrix - Exact Positions")
# plt.show()

# %% [markdown]
# ### Position Predcition

# %%
def simplify_position(pos):
    if pos in ["CB", "LB", "RB", "RWB", "LWB"]:
        return "DEF"
    elif pos in ["CM", "CDM", "CAM", "LM", "RM"]:
        return "MID"
    elif pos in ["ST", "CF", "LW", "RW"]:
        return "FWD"
    elif pos == "GK":
        return "GK"
    else:
        return "OTHER"

df["pos_group"] = df["Best Position"].apply(simplify_position)

le_group = LabelEncoder()
y_pos_group = le_group.fit_transform(df["pos_group"])

X_train, X_test, y_train, y_test = train_test_split(X, y_pos_group, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)

y_pred_pos_group = xgb.predict(X_test)

print("\nGrouped Position Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred_pos_group))
print("Report:\n", classification_report(y_test, y_pred_pos_group, target_names=le_group.classes_))

cm = confusion_matrix(y_test, y_pred_pos_group)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=le_group.classes_, yticklabels=le_group.classes_)
# plt.title("Confusion Matrix - Grouped Positions")
# plt.show()

# %% [markdown]
# #### Final Output

# %%
predicted_ovr = reg.predict(X)
predicted_pos_exact = le_exact.inverse_transform(clf_exact.predict(X))
predicted_pos_group = le_group.inverse_transform(xgb.predict(X))

results = pd.DataFrame({
    "Full Name": player_names,
    "Predicted OVR": predicted_ovr.round(1),
    "Actual OVR": actual_ovr,
    "Predicted Position (Exact)": predicted_pos_exact,
    "Actual Position": actual_pos,
    "Predicted Position (Grouped)": predicted_pos_group,
    "Actual Position (Grouped)": df["pos_group"]
})

print("\nSample Results:")
print(results.head())

results.to_csv("predictions.csv", index=False)

# %% [markdown]
# # Clustering

# %% [markdown]
# ### Attributes

# %%
# (exclude identifiers, overall, position)
cluster_features = df.drop(columns=["Full Name", "Overall", "Best Position", "Club Name", "pos_group"], errors="ignore")
X_scaled = scaler.fit_transform(cluster_features)

# Choose k using elbow + silhouette if you want (here we fix k=6 for demo)
kmeans = KMeans(n_clusters=6, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# %% [markdown]
# ### PCA for visualization

# %%
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["pca1"], df["pca2"] = X_pca[:, 0], X_pca[:, 1]

plt.figure(figsize=(8,6))
# sns.scatterplot(data=df, x="pca1", y="pca2", hue="cluster", palette="tab10", alpha=0.7)
# plt.title("FIFA 23 Player Clusters (PCA 2D)")
# plt.show()

# %% [markdown]
# ### Profiling

# %%
print("\nAverage OVR per cluster:")
print(df.groupby("cluster")["Overall"].mean().round(1))

print("\nMost common positions per cluster:")
print(df.groupby("cluster")["Best Position"].agg(lambda x: x.value_counts().index[0]))

# %% [markdown]
# ### Similar players

# %%
df["Predicted OVR"] = predicted_ovr.round(1)
df["Predicted Position (Exact)"] = predicted_pos_exact

def recommend_similar_players(player_name, n=5):
    """Return top-n most similar players to the given player (same cluster)."""
    if player_name not in df["Full Name"].values:
        return f"{player_name} not found in dataset"
    
    player_row = df[df["Full Name"] == player_name].iloc[0]
    cluster_id = player_row["cluster"]
    
    cluster_df = df[df["cluster"] == cluster_id].copy()
    X_cluster = X_scaled[df["cluster"] == cluster_id]
    
    # distances within the cluster
    distances = euclidean_distances([X_scaled[df["Full Name"] == player_name][0]], X_cluster)[0]
    cluster_df["distance"] = distances
    
    # sort by distance
    recommendations = cluster_df.sort_values("distance").head(n+1)  # +1 to include self
    return recommendations[["Full Name", "Overall", "Best Position", "Predicted OVR", "Predicted Position (Exact)", "distance"]].iloc[1:]

# Example usage:
print("\nSimilar players to 'Heung Min Son':")
print(recommend_similar_players("Heung Min Son", n=5))

# joblib.dump(reg, "reg_model.pkl")
# joblib.dump(clf_exact, "clf_exact.pkl")
# joblib.dump(xgb, "xgb_model.pkl")
# joblib.dump(scaler, "scaler.pkl")
# joblib.dump(le_exact, "le_exact.pkl")
# joblib.dump(le_group, "le_group.pkl")
# print("\nModels saved using pickle.")

print(type(X_train))


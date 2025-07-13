import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load CSV
df = pd.read_csv("/Users/erikarispe/googleMachineLearning/nfl_team_game_stats_2024.csv")

# Print columns to verify
print("Columns:", df.columns.tolist())

# Handle missing values
df["REDZONETD%"] = df["REDZONETD%"].fillna(0)

# Set feature and label columns
features = [
    "HOME",
    "PASSINGYARDS",
    "RUSHINGYARDS",
    "TURNOVERS",
    "THIRDDOWN%",
    "REDZONETD%",
    "PENALTYYARDS",
    "OPPONENTPAG"
]
label = "POINTSSCORED"

X = df[features]
y = df[label]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

### --- No Hyperparameters Model --- ###
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n--- Base Model ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 4))

# Feature importance
features_df = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Importance"])
print("\nTop Features:")
print(features_df.sort_values("Importance", ascending=False).head(15))

### --- Tuned Hyperparameters Model --- ###
rf2 = RandomForestRegressor(
    n_estimators=1000,
    criterion='squared_error',
    min_samples_split=10,
    max_depth=14,
    random_state=42
)

rf2.fit(X_train, y_train)
y_pred2 = rf2.predict(X_test)

print("\n--- Tuned Model ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred2), 2))
print("R2 Score:", round(r2_score(y_test, y_pred2), 4))

# Feature importance
features2_df = pd.DataFrame(rf2.feature_importances_, index=X.columns, columns=["Importance"])
print("\nTop Features (Tuned):")
print(features2_df.sort_values("Importance", ascending=False).head(15))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load updated CSV
df = pd.read_csv("/Users/erikarispe/googleMachineLearning/nfl_2024_team_game_stats.csv")

# Print columns to verify
print("Columns:", df.columns.tolist())

# Handle missing values
df["REDZONETD%"] = df["REDZONETD%"].fillna(0)
df["EPA_per_play"] = df["EPA_per_play"].fillna(0)
df["PASS_EPA"] = df["PASS_EPA"].fillna(0)
df["RUSH_EPA"] = df["RUSH_EPA"].fillna(0)
df["SUCCESS_RATE"] = df["SUCCESS_RATE"].fillna(0)

# Set feature and label columns (now with advanced stats)
features = [
    "HOME",
    "PASSINGYARDS",
    "RUSHINGYARDS",
    "TURNOVERS",
    "THIRDDOWN%",
    "REDZONETD%",
    "PENALTYYARDS",
    "OPPONENTPAG",
    "EPA_per_play",
    "PASS_EPA",
    "RUSH_EPA",
    "SUCCESS_RATE"
]
label = "POINTSSCORED"

X = df[features]
y = df[label]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=17, test_size=0.2)

### --- Base Random Forest Model --- ###
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

### --- Tuned Random Forest Model --- ###
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

# Feature importance for tuned model
features2_df = pd.DataFrame(rf2.feature_importances_, index=X.columns, columns=["Importance"])
print("\nTop Features (Tuned):")
print(features2_df.sort_values("Importance", ascending=False).head(15))

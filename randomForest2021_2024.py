import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load the combined dataset (update path if needed)
df = pd.read_csv("/Users/erikarispe/googleMachineLearning/nfl_team_game_stats_2021_2024_combined.csv")

# Fill missing values and replace inf values
df.fillna(0, inplace=True)
df.replace([float('inf'), float('-inf')], 0, inplace=True)

# Define feature columns (core + EPA + rolling + adjusted + interaction)
features = [
    # "HOME",  # Uncomment if this column exists in your CSV
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
    "SUCCESS_RATE",
    "EPA_per_play_last3",
    "REDZONETD%_last3",
    "POINTSSCORED_last3",
    "OPPONENTPAG_last3",
    "Adj_EPA",
    "Adj_REDZONETD%",
    "Adj_POINTS_last3",
    "EPA_x_REDZONETD",
    "YARD_RATIO"
]

label = "POINTSSCORED"

# Prepare training data
X = df[features]
y = df[label]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

# --- Base Model ---
rf_base = RandomForestRegressor()
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

print("\n--- Base Model ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred_base), 2))
print("R2 Score:", round(r2_score(y_test, y_pred_base), 4))

base_importance_df = pd.DataFrame(rf_base.feature_importances_, index=X.columns, columns=["Importance"])
print("\nTop Features (Base):")
print(base_importance_df.sort_values("Importance", ascending=False).head(15))

# --- Tuned Model ---
rf_tuned = RandomForestRegressor(
    n_estimators=1000,
    max_depth=14,
    min_samples_split=10,
    criterion="squared_error",
    random_state=42
)
rf_tuned.fit(X_train, y_train)
y_pred_tuned = rf_tuned.predict(X_test)

print("\n--- Tuned Model ---")
print("MAE:", round(mean_absolute_error(y_test, y_pred_tuned), 2))
print("R2 Score:", round(r2_score(y_test, y_pred_tuned), 4))

tuned_importance_df = pd.DataFrame(rf_tuned.feature_importances_, index=X.columns, columns=["Importance"])
print("\nTop Features (Tuned):")
print(tuned_importance_df.sort_values("Importance", ascending=False).head(15))

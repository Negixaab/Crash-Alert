import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("data/realroaddata.csv")

# Features
X = df[[
    "Road Accidents - Cases",
    "Road Accidents - Injured",
    "Road Accidents - Died"
]]

# 🎯 LEVEL 3 RISK (non-linear + noise)
risk = (
    0.5 * df["Road Accidents - Cases"] +
    1.5 * df["Road Accidents - Injured"] +
    3.0 * df["Road Accidents - Died"] +
    0.0001 * df["Road Accidents - Cases"] * df["Road Accidents - Died"] +  # interaction
    np.random.normal(0, 50, len(df))  # noise
)

y = risk

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Level 3 ML model trained")
print("Features:", X.columns.tolist())
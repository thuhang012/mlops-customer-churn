#!/usr/bin/env python3
"""Create dummy/mock artifacts for CI/CD testing."""

import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Create artifact directories
Path("artifacts/models").mkdir(parents=True, exist_ok=True)
Path("artifacts/preprocessors").mkdir(parents=True, exist_ok=True)

# Create dummy model
dummy_model = LogisticRegression(random_state=42)
model_path = Path("artifacts/models/Netflix_Prediction_final.pkl")
joblib.dump(dummy_model, model_path)
print(f"✅ Created dummy model: {model_path}")

# Create dummy preprocessor
feature_columns = [
    "age_group", "gender", "country", "subscription_plan", "monthly_fee",
    "device_type", "watch_time_minutes", "session_count",
    "completion_percentage", "rating", "days_since_last_watch"
]

dummy_preprocessor = {
    "pipeline": ColumnTransformer([
        ("scaler", StandardScaler(), feature_columns)
    ]),
    "feature_columns": feature_columns,
}

preprocessor_path = Path("artifacts/preprocessors/preprocessor.pkl")
joblib.dump(dummy_preprocessor, preprocessor_path)
print(f"✅ Created dummy preprocessor: {preprocessor_path}")

print("\n⚠️  IMPORTANT: These are dummy artifacts for CI/CD testing only!")
print("   Replace with real artifacts after training the model.")

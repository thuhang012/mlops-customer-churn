import joblib
import pandas as pd
from src.mlops_project.api.schema import CustomerInput

model = joblib.load("artifacts/models/dummy_model.pkl")

def predict(data: CustomerInput):
    df = pd.DataFrame([data.dict()])
    y_pred = model.predict(df)
    return int(y_pred[0])
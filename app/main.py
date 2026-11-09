from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("models/cybershield_xgb.pkl")
scaler = joblib.load("models/scaler.pkl")
cols = joblib.load("models/feature_columns.pkl")

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload]).reindex(columns=cols, fill_value=0)
    prob = float(model.predict_proba(scaler.transform(df))[0, 1])
    return {"prediction": int(prob > 0.5), "threat_probability": prob}
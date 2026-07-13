from fastapi import FastAPI
import pandas as pd
from pydantic import create_model
import joblib
import json
import numpy as np

app = FastAPI(title="Fraud Detection API")

model = joblib.load("isolation_forest_fraud.joblib")
with open("feature_order.json") as f:
    FEATURE_ORDER = json.load(f)
    
field_definitions = {name: (float, ...) for name in FEATURE_ORDER}
Transaction = create_model("Transaction", **field_definitions)

THRESHOLDS = {
    "critical": 0.7039,
    "high" : 0.6402,
    "medium" : 0.5583
}

def risk_tier(score: float) -> str:
    if score >= THRESHOLDS["critical"]:
        return "critical"
    elif score >= THRESHOLDS["high"]:
        return "high"
    elif score >= THRESHOLDS["medium"]:
        return "medium"
    return "low"


@app.post("/predict")
def predict(transaction: Transaction):
    data = transaction.model_dump()
    feature_df = pd.DataFrame([[data[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
    score = float(-model.score_samples(feature_df)[0])
    return {"anomaly_score": score, "risk_tier": risk_tier(score)}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
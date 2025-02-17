from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load models and scaler
scaler_path = "models/scaler.pkl"
gbm_model_path = "models/gbm_model.pkl"
lstm_model_path = "models/lstm_model.h5"

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

with open(gbm_model_path, "rb") as f:
    gbm_model = pickle.load(f)

lstm_model = load_model(lstm_model_path)

app = FastAPI(title="TrackVisionAI API")

@app.get("/")
def home():
    return {"message": "TrackVisionAI API is running!"}

@app.post("/predict_race/")
def predict_race(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_lstm = features_scaled.reshape((features_scaled.shape[0], features_scaled.shape[1], 1))
        prediction = lstm_model.predict(features_lstm)[0][0]
        return {"race_outcome": "Winner" if prediction > 0.5 else "Loser", "probability": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_payout/")
def predict_payout(data: dict):
    try:
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        payout_prediction = gbm_model.predict(features_scaled)[0]
        return {"predicted_payout": float(payout_prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

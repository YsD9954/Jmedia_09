from fastapi import APIRouter, HTTPException
import pandas as pd
import pickle
import os
from prophet import Prophet

router = APIRouter()

# Ensure model exists
model_path = "models/prophet_model.pkl"
if not os.path.exists(model_path):
    raise HTTPException(status_code=500, detail="Model file not found. Train the model first.")

# Load Prophet model
with open(model_path, "rb") as f:
    prophet_model = pickle.load(f)

@router.get("/")
def home():
    return {"message": "TrackVisionAI API is running!"}

@router.post("/predict/")
def predict(data: dict):
    try:
        days = data.get("days", 10)  # Default to 10 days
        future = pd.DataFrame({"ds": pd.date_range(start=pd.Timestamp.today(), periods=days, freq="D")})
        forecast = prophet_model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return forecast.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

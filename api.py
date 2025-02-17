from fastapi import APIRouter, HTTPException, FastAPI
import pandas as pd
import pickle
import os
from prophet import Prophet
from api import router

app = FastAPI(title="TrackVisionAI API")

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

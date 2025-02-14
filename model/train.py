from prophet import Prophet
import pandas as pd
import pickle
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Load dataset
df = pd.read_csv("trackvisionai_race_data.csv")

# Prepare data for Prophet
df.rename(columns={"date": "ds", "value": "y"}, inplace=True)

# Train the model
model = Prophet()
model.fit(df)

# Save the trained model correctly
model_path = "models/prophet_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"✅ Model trained and saved at {model_path}")

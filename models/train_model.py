from prophet import Prophet
import pandas as pd
import pickle
import os

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Generate a sample dataset
data = pd.DataFrame({
    'ds': pd.date_range(start='2024-01-01', periods=200, freq='D'),
    'y': [i + (i % 10) * 2 for i in range(200)]
})

# Train Prophet model
model = Prophet()
model.fit(data)

# Save model
with open("models/prophet_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved successfully!")

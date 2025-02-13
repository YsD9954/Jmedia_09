import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import xgboost as xgb

# Load Dataset
df = pd.read_csv("datasets/race_data.csv")

# Encode Categorical Variables
label_encoder = LabelEncoder()
df["weather"] = label_encoder.fit_transform(df["weather"])
df["track_type"] = label_encoder.fit_transform(df["track_type"])
df["outcome"] = label_encoder.fit_transform(df["outcome"])  # Win = 1, Lose = 0

# Save Label Encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

# Feature Selection
X = df[["post_position", "weather", "track_type", "jockey_win_rate", "trainer_win_rate"]].values
y = df["outcome"].values

# Normalize Features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save Scaler
joblib.dump(scaler, "models/scaler.pkl")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Reshape Data for LSTM (3D)
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ---- Train LSTM Model ----
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train, epochs=50, batch_size=16, validation_data=(X_test_lstm, y_test))

# Save LSTM Model
lstm_model.save("models/lstm_model.h5")

# ---- Train XGBoost Model for Betting Optimization ----
gbm_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm_model.fit(X_train, y_train)

# Save XGBoost Model
joblib.dump(gbm_model, "models/gbm_model.pkl")

print("✅ Models trained and saved successfully!")

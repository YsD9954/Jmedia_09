import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
data_path = "trackvisionai_race_data.csv"
df = pd.read_csv(data_path)

# Features and target variables
features = ["Jockey_Experience", "Horse_Speed", "Previous_Wins", "Odds"]
target_win = "Winner_Status"
target_payout = "Payout"

# Convert categorical target to binary (Winner = 1, Loser = 0)
df[target_win] = df[target_win].apply(lambda x: 1 if x == "Winner" else 0)

# Scale features
scaler = MinMaxScaler()
X = scaler.fit_transform(df[features])
y_win = df[target_win].values
y_payout = df[target_payout].values

# Split dataset
X_train, X_test, y_train_win, y_test_win = train_test_split(X, y_win, test_size=0.2, random_state=42)
_, _, y_train_payout, y_test_payout = train_test_split(X, y_payout, test_size=0.2, random_state=42)

# LSTM Model for race outcome prediction
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    LSTM(50, return_sequences=False),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_win, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test_win))

# GBM Model for payout prediction
gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbm_model.fit(X_train, y_train_payout)

# Save models
os.makedirs("models", exist_ok=True)
with open("models/gbm_model.pkl", "wb") as f:
    pickle.dump(gbm_model, f)

lstm_model.save("models/lstm_model.h5")

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Training complete. Models saved!")

import pandas as pd
import joblib
from pymongo import MongoClient
from datetime import timedelta
import os

HOURS_AHEAD = 72

def load_latest_features():
    client = MongoClient(os.getenv("MONGO_URI"))
    col = client["Pearls_aqi_feature_store"]["karachi_air_qualityIndex"]

    df = pd.DataFrame(list(col.find({}, {"_id": 0})))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    return df.iloc[-1:].copy()

##change this part in github 
def load_latest_model():
    client = MongoClient(os.getenv("MONGO_URI"))
    col = client["Pearls_aqi_model_registry"]["models"]

    latest = col.find_one(
        sort=[("created_at", -1)]
    )

    if not latest:
        raise ValueError("❌ No model found in registry")

    return joblib.load(latest["artifact_path"])
# till 
def forecast_3_days():
    model = load_latest_model()
    current = load_latest_features()

    predictions = []

    for step in range(HOURS_AHEAD):
        X = current.drop(columns=["timestamp", "location", "pm25_next_hour"])
        pred_pm25 = model.predict(X)[0]

        predictions.append(pred_pm25)

        # ⏩ Advance time
        next_time = current["timestamp"].iloc[0] + timedelta(hours=1)

        # 🔁 Shift lag features
        current["pm25_lag_6h"] = current["pm25_lag_3h"]
        current["pm25_lag_3h"] = current["pm25_lag_1h"]
        current["pm25_lag_1h"] = pred_pm25

        # Update rolling proxy
        current["pm25"] = pred_pm25

        # Update time features
        current["timestamp"] = next_time
        current["hour"] = next_time.hour
        current["day"] = next_time.day
        current["month"] = next_time.month
        current["day_of_week"] = next_time.dayofweek
        current["is_weekend"] = int(next_time.dayofweek >= 5)
        current["is_rush_hour"] = int(next_time.hour in [7,8,9,17,18,19])

    return predictions

if __name__ == "__main__":
    preds = forecast_3_days()
    print("✅ 3-Day AQI Forecast Generated")



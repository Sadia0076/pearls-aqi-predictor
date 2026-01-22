import pandas as pd


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw AQI data and engineers industry-grade features
    for pollutant forecasting and AQI calculation.
    """

    # --------------------------------------------------
    # 1. Basic cleaning (time-series standard)
    # --------------------------------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward + backward fill (acceptable for hourly AQI sensors)
    df.fillna(method="ffill", inplace=True)
    df.fillna(method="bfill", inplace=True)

    # --------------------------------------------------
    # 2. Time-based features
    # --------------------------------------------------
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # --------------------------------------------------
    # 3. Behavioral & human-activity proxies
    # --------------------------------------------------
    # Traffic emission proxy
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Weekend proxy (lower traffic & industrial activity)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # --------------------------------------------------
    # 4. Change-rate features (pollution dynamics)
    # --------------------------------------------------
    df["pm25_change"] = df["pm25"].diff().fillna(0)
    df["pm10_change"] = df["pm10"].diff().fillna(0)
    df["co_change"] = df["co"].diff().fillna(0)
    df["no2_change"] = df["no2"].diff().fillna(0)
    df["so2_change"] = df["so2"].diff().fillna(0)
    df["o3_change"] = df["o3"].diff().fillna(0)

    # --------------------------------------------------
    # 5. Lag features (VERY IMPORTANT FOR FORECASTING)
    # --------------------------------------------------
    df["pm25_lag_1h"] = df["pm25"].shift(1)
    df["pm25_lag_3h"] = df["pm25"].shift(3)
    df["pm25_lag_6h"] = df["pm25"].shift(6)

    df["pm10_lag_1h"] = df["pm10"].shift(1)
    df["pm10_lag_3h"] = df["pm10"].shift(3)

    # --------------------------------------------------
    # 6. Rolling statistics (temporal context)
    # --------------------------------------------------
    df["pm25_rolling_3h"] = df["pm25"].rolling(3).mean()
    df["pm25_rolling_6h"] = df["pm25"].rolling(6).mean()
    df["pm25_rolling_12h"] = df["pm25"].rolling(12).mean()

    df["pm10_rolling_3h"] = df["pm10"].rolling(3).mean()
    df["pm10_rolling_6h"] = df["pm10"].rolling(6).mean()

    # --------------------------------------------------
    # 7. Rain / clean-air proxy (event-based feature)
    # --------------------------------------------------
    # Sudden drop in PM2.5 often caused by rain or wind
    df["post_rain_effect"] = (df["pm25_change"] < -5).astype(int)

    # --------------------------------------------------
    # 8. Targets (Predict pollutants, NOT AQI)
    # --------------------------------------------------
    df["pm25_next_hour"] = df["pm25"].shift(-1)
    df["pm10_next_hour"] = df["pm10"].shift(-1)

    # --------------------------------------------------
    # 9. Final cleanup
    # --------------------------------------------------
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


# --------------------------------------------------
# Run independently
# --------------------------------------------------
if __name__ == "__main__":
    df = pd.read_csv("raw_aqi_data.csv")
    clean_df = transform_features(df)
    clean_df.to_csv("clean_aqi_features1.csv", index=False)
    print("✅ Data cleaned, features engineered & targets created")

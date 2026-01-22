import requests
import pandas as pd
from config import config

def fetch_air_quality_data(start_date, end_date):
    params = {
        "latitude": config.LATITUDE,
        "longitude": config.LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone"
        ]
    }

    response = requests.get(config.AIR_QUALITY_API, params=params)
    response.raise_for_status()

    data = response.json()["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["time"]),
        "pm25": data["pm2_5"],
        "pm10": data["pm10"],
        "co": data["carbon_monoxide"],
        "no2": data["nitrogen_dioxide"],
        "so2": data["sulphur_dioxide"],
        "o3": data["ozone"],
        "location": config.LOCATION
    })

    return df


if __name__ == "__main__":
    df = fetch_air_quality_data(
        start_date=config.START_DATE,
        end_date=config.END_DATE
    )
    df.to_csv("data/raw/raw_aqi_data.csv", index=False)
    print("✅ Raw AQI data fetched & saved")

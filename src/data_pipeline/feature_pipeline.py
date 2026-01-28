#This does ONLY last-hour data. NO backfill loop
from datetime import datetime, timedelta
from fetch_data import fetch_air_quality_data
from clean_transform import transform_features
from upload_to_mongodb import upload_features
import sys
import os
sys.path.append(os.getcwd())

def run_hourly_pipeline():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)

    print(f"⏱ Fetching data from {start_time} to {end_time}")

    raw_df = fetch_air_quality_data(
        start_date=start_time.strftime("%Y-%m-%d"),
        end_date=end_time.strftime("%Y-%m-%d")
    )

    if raw_df.empty:
        print("⚠️ No new data fetched")
        return

    features_df = transform_features(raw_df)
    upload_features(features_df)

    print("✅ Hourly feature pipeline completed")

if __name__ == "__main__":
    run_hourly_pipeline()


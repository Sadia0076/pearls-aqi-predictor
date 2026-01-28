# src/data_pipeline/backfill.py
from fetch_data import fetch_air_quality_data
from clean_transform import transform_features
from upload_to_mongodb import upload_features

# Example: backfill by weekly ranges
date_ranges = [
    ("2024-01-01", "2024-01-07"),
    ("2024-01-08", "2024-01-14"),
    ("2024-01-15", "2024-01-21"),
]

for start, end in date_ranges:
    print(f"⏳ Processing {start} → {end}")

    # 1️⃣ Fetch raw AQI + weather data
    raw_df = fetch_air_quality_data(start, end)

    # 2️⃣ Clean + engineer features
    features_df = transform_features(raw_df)

    # 3️⃣ Upload to MongoDB
    upload_features(features_df)

print("✅ Backfill completed!")



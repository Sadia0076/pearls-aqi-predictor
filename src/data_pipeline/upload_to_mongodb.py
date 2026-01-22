# src/data_pipeline/upload_to_mongodb.py

import pandas as pd
from pymongo import MongoClient
import os


def upload_features(df: pd.DataFrame):
    """
    Uploads a DataFrame of AQI features to MongoDB Feature Store.
    """

    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("❌ MONGO_URI environment variable is not set")

    # Connect to MongoDB Atlas
    client = MongoClient(mongo_uri)

    # Feature Store DB & Collection
    db = client["Pearls_aqi_feature_store"]
    collection = db["karachi_air_quality_index"]

    # Convert DataFrame to list of dicts
    records = df.to_dict(orient="records")

    if not records:
        print("⚠️ No records to upload")
        return

    # Insert into MongoDB
    collection.insert_many(records)
    print(f"✅ Uploaded {len(records)} records to MongoDB Feature Store")


if __name__ == "__main__":
    df = pd.read_csv("clean_aqi_features1.csv")
    upload_features(df)


# src/data_pipeline/upload_to_mongodb.py
import pandas as pd
from pymongo import MongoClient

def upload_features(df: pd.DataFrame):
    """
    Uploads a DataFrame of AQI features to MongoDB Feature Store.
    Accepts a DataFrame directly (good for backfill loops).
    """
    # Connect to LOCAL MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["Pearls_aqi_feature_store"]
    collection = db["karachi_air_qualityIndex"]

    # Convert DataFrame to list of dicts
    records = df.to_dict(orient="records")

    # Insert into MongoDB
    collection.insert_many(records)
    print(f"✅ Uploaded {len(records)} records to MongoDB Feature Store")
    

# If you want to run independently from CSV
if __name__ == "__main__":
    df = pd.read_csv("clean_aqi_features1.csv")
    upload_features(df)

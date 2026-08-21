import os
import platform
import sqlite3

import pandas as pd
from dotenv import load_dotenv

from fetch_data import fetch_all_data
from parse_features import compute_features

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to local SQLite feature store (when running locally on Windows)
DB_PATH = os.path.join(BASE_DIR, "feature_store", "aqi_features.db")
CSV_PATH = os.path.join(BASE_DIR, "data", "aqi_features.csv")

def save_to_sqlite(df):
    
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    try:
        df.to_sql("aqi_features", conn, if_exists="append", index=False)
        total = pd.read_sql("SELECT COUNT(*) as count FROM aqi_features", conn)
    finally:
        conn.close()

    print(f"SQLite store: {total['count'].iloc[0]} total rows in feature store")

def read_from_sqlite():

    if not os.path.exists(DB_PATH):
        print("SQLite feature store does not exist yet.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_PATH)

    try:
        df = pd.read_sql("SELECT * FROM aqi_features ORDER BY timestamp", conn)
    finally:
        conn.close()

    return df

# Use Hopesworks when using a Linux System (GitHub Actions)
def save_to_hopsworks(df):  
    cert_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".hopsworks_certs"
    )
    os.makedirs(cert_folder, exist_ok=True)

    import hopsworks

    project = hopsworks.login(
        project='aqi_predictor_23',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        cert_folder=cert_folder
    )

    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="aqi_features",
        version=1,
        primary_key=["timestamp"],
        event_time="timestamp",
        online_enabled=False,
        description="Hourly AQI features - Karachi"
    )

    import time
    for attempt in range(1, 4):
        try:
            fg.insert(df)
            print("Data stored in Hopsworks successfully!")
            break
        except Exception as e:
            print(f"Insert attempt {attempt}/3 failed: {e}")
            if attempt < 3:
                print("Waiting 30s before retry...")
                time.sleep(30)
            else:
                print("Hopsworks insert failed after 3 attempts.")
                raise RuntimeError("Could not store features in Hopsworks.")

def run_pipeline():
    
    print("Fetching raw data...")
    raw_data = fetch_all_data()
    print("Raw data fetched successfully!")

    print("Parsing and computing features...")
    df = compute_features(raw_data)
    print("Features computed successfully!")
    print(df.to_string())

    if platform.system() == "Windows":
        print(f"Platfrom Detected: {platform.system()}")
        print("Using local SQLite feature store...")
        save_to_sqlite(df)
    else:
        print(f"Platform Detected {platform.system()}")
        print("Using Hopsworks feature store...")
        save_to_hopsworks(df)

    print(f"\nPipeline complete!")
    print(f"Current AQI = {df['aqi'].iloc[0]}")

if __name__ == "__main__":
    run_pipeline()
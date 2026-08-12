import os
import sys
import time
import sqlite3
import platform
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

LAT, LON = 24.860753, 67.029503 # co-ordinates of my location (Karachi)

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "feature_store", "aqi_features.db"
)

def fetch_historical_openmeteo_aqi(start_date: str, end_date: str) -> pd.DataFrame:
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "us_aqi",       
            "pm2_5",        
            "pm10",
            "ozone", # o3
            "nitrogen_dioxide", # no2
            "carbon_monoxide",  # co
            "sulphur_dioxide",  # so2
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC"
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "aqi": hourly["us_aqi"],
        "pm2_5": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "o3": hourly["ozone"],
        "no2": hourly["nitrogen_dioxide"],
        "co": hourly["carbon_monoxide"],
        "so2": hourly["sulphur_dioxide"],
    })

    return df

def backfill(days: int = 365):
    
    now = datetime.utcnow()
    start_dt = now - timedelta(days=days)

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = now.strftime("%Y-%m-%d")

    print(f"Fetching {days} days of historical AQI from Open-Meteo...")
    print(f"Date range: {start_date} to {end_date}")

    df = fetch_historical_openmeteo_aqi(start_date, end_date)

    df = df.dropna(subset=["aqi"])  # drop rows where AQI is null 
    df["aqi"] = df["aqi"].astype(int)

    # Fill missing pollutants with forward fill then 0
    pollution_data = ["pm2_5", "pm10", "o3", "no2", "co", "so2"]
    df[pollution_data] = df[pollution_data].ffill().fillna(0.0).astype(float)

    print(f"Open-Meteo rows fetched: {len(df)}")

    # Add historical weather columns as 0 since OpenWeather weather hidtory requires paid plan
    weather_data = ["temperature", "pressure", "humidity", "wind_speed", "wind_direction"]

    for col in weather_data:
        df[col] = 0.0

    # Add time features (reusing same logic as done in parse_features.py/compute_features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    def get_hour_category(hour):
        if 6  <= hour < 10:
            return 1
        if 10 <= hour < 17: 
            return 2
        if 17 <= hour < 21: 
            return 3
        return 0

    df["hour_category"] = df["hour"].apply(get_hour_category)

    df["pm2_5"] = df["pm2_5"].astype(float)
    df["pm10"] = df["pm10"].astype(float)
    df["o3"] = df["o3"].astype(float)
    df["no2"] = df["no2"].astype(float)
    df["co"] = df["co"].astype(float)
    df["so2"] = df["so2"].astype(float)

    df["temperature"] = df["temperature"].astype(float)
    df["pressure"] = df["pressure"].astype(int)
    df["humidity"] = df["humidity"].astype(int)
    df["wind_speed"] = df["wind_speed"].astype(float)
    df["wind_direction"] = df["wind_direction"].astype(int)

    df["hour"] = df["hour"].astype(int)
    df["day"] = df["day"].astype(int)
    df["day_of_week"] = df["day_of_week"].astype(int)
    df["month"] = df["month"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["hour_category"] = df["hour_category"].astype(int)

    # Final column order matching pipeline.py
    final_cols = ["timestamp", "aqi", "pm2_5", "pm10", "o3", "no2", "co", "so2",
    "temperature", "pressure", "humidity", "wind_speed", "wind_direction",
    "hour", "day", "day_of_week", "month", "is_weekend", "hour_category"]

    df = df[final_cols]

    print(f"\nBackfill complete.")
    print(f"Total rows: {len(df)}")
    print(f"AQI range: {df['aqi'].min()} to {df['aqi'].max()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(df[["timestamp","aqi","pm2_5","hour"]].head(5).to_string())

    if platform.system() == "Windows":
        _save_to_sqlite(df)
    else:
        _save_to_hopsworks(df)

def _save_to_sqlite(df: pd.DataFrame):

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    try:
        df["timestamp"] = df["timestamp"].astype(str)

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='aqi_features'"
        )

        if cursor.fetchone():
            existing = pd.read_sql("SELECT timestamp FROM aqi_features", conn)
            df = df[~df["timestamp"].isin(existing["timestamp"])]

        if len(df) == 0:
            print("All rows already in SQLite — nothing new to insert.")
            return

        df.to_sql("aqi_features", conn, if_exists="append", index=False)
        total = pd.read_sql("SELECT COUNT(*) as c FROM aqi_features", conn)

        print(f"SQLite: inserted {len(df)} rows. Total: {total['c'].iloc[0]}")
    
    finally:
        conn.close()

def _save_to_hopsworks(df: pd.DataFrame):

    cert_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".hopsworks_certs"
    )
    os.makedirs(cert_folder, exist_ok=True)

    import hopsworks

    project = hopsworks.login(
        project="aqi_predictor_23",
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

    fg.insert(df)

    print(f"Hopsworks: {len(df)} rows inserted.")

if __name__ == "__main__":
    backfill(days=365)
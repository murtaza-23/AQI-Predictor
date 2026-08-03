import numpy as np
import pandas as pd
from datetime import datetime

# convert data obtained from the APIs in form of a dictionary to a DataFrame (tabular form)
def compute_features(data: dict) -> pd.DataFrame: 

    df = pd.DataFrame([data])   # single row dataframe

    df["timestamp"] = pd.to_datetime(df["timestamp"])   # convert timestamp to proper date time

    # Time-based features for hourly and daily AQI patterns
    df["hour"] = df["timestamp"].dt.hour    # (0-23) hour of day
    df["day"] = df["timestamp"].dt.dayofweek # Mon-Fri (0-6)
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day"] >= 5).astype(int)

    def get_hour_category(hour):    # predict what part of day it is (Night, Morning, Midday etc)
        if 6 <= hour < 10:
            return 1    # Morning rush
        elif 10 <= hour < 17:
            return 2    # Midday
        elif 17 <= hour < 21:
            return 3    # Evening rush
        else:
            return 0    # Night

    df["hour_category"] = df["hour"].apply(get_hour_category)

    pollutant_data_columns = ["pm2_5", "pm10", "o3", "no2", "co", "so2"]
    weather_data_columns = ["temperature", "pressure", "humidity", "wind_speed", "wind_direction"]

    for col in pollutant_data_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in weather_data_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

    final_columns = ["timestamp", "aqi", "pm2_5", "pm10", "o3", "no2", "co", "so2",
    "temperature", "pressure", "humidity", "wind_speed", "wind_direction",
    "hour", "day", "month", "is_weekend", "hour_category"]

    return df[final_columns]


# Function to add AQI Change rate
def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_change_rate"] = df["aqi"].diff()    # current - previous

    df = df.dropna(subset=["aqi_change_rate"]).reset_index(drop=True)

    return df

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from feature_pipeline.fetch_data import fetch_all_data

    raw_data = fetch_all_data()
    df = compute_features(raw_data)
    print(df.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nNull values:\n{df.isnull().sum()}")